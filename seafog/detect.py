"""
seafog.detect contains Seafog class which provides several methods to detect seafog top height and seafog area.
"""

from time import ctime
from typing import Union, Tuple, List

import numpy as np
from xarray import DataArray, Dataset

from .utils import solar_altitude_zenith, logger


def remove_single_point(data: np.ndarray) -> np.ndarray:
    """
    remove single point which eight surrounding points' valid number is less than 4 or the 16 surround-surrounding points' valid number is less than 8.

    :param data: 2-D ndarray.
    :return: data with single points has been removed.
    """
    result = data.copy()

    # loop and check
    for i in range(2, data.shape[0] - 2):
        for j in range(2, data.shape[1] - 2):
            # if the point is np.nan, pass
            if np.isnan(data[i, j]):
                continue
            # check the innermost ring's 8 points and outermost ring's 16 points
            index_8 = np.where(~np.isnan(data[i - 1:i + 2, j - 1:j + 2]))
            index_16 = np.where(~np.isnan(data[i - 2:i + 3, j - 2:j + 3]))
            if index_8[0].size < 5:
                result[i, j] = np.nan
            elif index_16[0].size - 9 < 8:
                result[i, j] = np.nan

    return result


class Seafog:
    """
    ``Seafog`` class provides several methods to detect seafog top height and seafog area.
    seafog.detect module has exported a Seafog instance called ``detect_seafog``, use it directly.
    """

    def __init__(self):
        # place parameters setting here
        self._diff_range_night_IR: Tuple[float, float] = (-5.5, -3.5)
        self._diff_range_daytime_IR: Tuple[float, float, float] = (-2, 3, 45)
        self._beta: float = 0.0685
        self._ir_sst_diff_range: Tuple[float, float] = (-5, 10)
        # threshold value of QCLOUD
        self._qcloud_threshold = 0.02
        # datas which are higher than 500m will be removed
        self._maximum_height = 600

    # ############################################## parameters setting ##############################################

    @property
    def diff_range_night_IR(self) -> Tuple[float, float]:
        """
        the value range of ``IR4 - IR1`` at night.

        :return: (min_value, max_value)
        """
        return self._diff_range_night_IR

    @diff_range_night_IR.setter
    def diff_range_night_IR(self, value: Union[Tuple[float, float], List[float]]):
        assert len(value) == 2, f"Only two values will be accepted, but got {value}"
        assert isinstance(value[0], float) and isinstance(value[1], float), f"Only float value will be accepted, but got {value}"

        self._diff_range_night_IR = tuple(value)

    @property
    def diff_range_daytime_IR(self) -> Tuple[float, float, float]:
        """
        the value range of ``IR4 - IR1`` in the daytime, including three values of two range.

        :return: (start, end(start_two), end_two)
        """
        return self._diff_range_daytime_IR

    @diff_range_daytime_IR.setter
    def diff_range_daytime_IR(self, value: Union[Tuple[float, float, float], List[float]]):
        assert len(value) == 3, f"Only three values will be accepted, but got{value}"
        for _value in value:
            assert isinstance(_value, float) or isinstance(_value, int), f"Only int or float value will be accepted, but got {value}"

        value = tuple(float(x) for x in value)
        self._diff_range_daytime_IR = value

    @property
    def beta(self) -> float:
        """
        back scatter parameter.

        :return: back scatter parameter.
        """
        return self._beta

    @beta.setter
    def beta(self, value: float):
        assert isinstance(value, float), "Only float value will be accepted"
        self._beta = value

    @property
    def IR_sst_diff_range(self) -> Tuple[float, float]:
        """
        the value range of ``IR4 - SST``.

        :return: (min_value, max_value).
        """
        return self._ir_sst_diff_range

    @IR_sst_diff_range.setter
    def IR_sst_diff_range(self, value: Tuple[float, float]):
        assert isinstance(value, tuple), "Only float value will be accepted"
        assert len(value) == 2, f"Only two values will be accepted, but got{value}"
        self._ir_sst_diff_range = value

    @property
    def QCLOUD_threshold(self) -> float:
        """
        liquid water content of the cloud.
        any part of air which QCLOUD is bigger than this threshold will be regarded as seafog.

        :return: threshold value of QCLOUD.
        """
        return self._qcloud_threshold

    @QCLOUD_threshold.setter
    def QCLOUD_threshold(self, value: float):
        self._qcloud_threshold = value

    @property
    def maximum_height(self) -> int:
        """
        maximum height of seafog top.

        :return: threshold value.
        """
        return self._maximum_height

    @maximum_height.setter
    def maximum_height(self, value: int):
        self._maximum_height = value

    # ################################################################################################################

    def seafog_from_wrfout(self, wrfout: Dataset) -> Dataset:
        """
        calculate the seafog area and top height of seafog using WRF data.

        :param wrfout: WRF data
        :return: ``xarray.Dataset`` object.
        """
        # check data in dataset
        for field in ["QCLOUD", "XTIME", "HGT", "PH", "PHB", "XLONG", "XLAT"]:
            assert field in wrfout.keys(), f"Your WRF data doesn't have {field} data, which is necessary for calculation"

        # read QCLOUD, change unit to g/kg
        QCloud = wrfout["QCLOUD"].to_numpy() * 1000
        # read other data
        terrain_height = wrfout["HGT"].to_numpy()
        perturbation_geo = wrfout["PH"].to_numpy()
        base_geo = wrfout["PHB"].to_numpy()

        # expand terrain_height's dimension
        terrain_height = np.expand_dims(terrain_height, axis=1)
        terrain_height = np.tile(terrain_height, (1, perturbation_geo.shape[1], 1, 1))

        # calculate height from sea level
        height = (perturbation_geo + base_geo) / 9.8 - terrain_height
        # calculate half-height
        height = (height[:, 1:, :, :] + height[:, :-1, :, :]) / 2

        # search datas which are under maximum height and bigger that QCLOUD threshold
        height_index = height <= self._maximum_height
        QCloud_index = QCloud >= self._qcloud_threshold
        # combine and get seafog index
        seafog_index = np.logical_and(height_index, QCloud_index)

        # then we need to calculate the height where QCLOUD == QCLOUD_threshold, using linear interpolation
        # # copy height
        seafog = height.copy()
        # # change datas which index is False to nan
        seafog[~seafog_index] = np.nan
        # # find the largest height values
        valid_largest_height = np.nanmax(seafog, axis=1)
        # # expand dimension
        valid_largest_height = np.expand_dims(valid_largest_height, axis=1)
        # # get indexes of these values
        valid_index = seafog == valid_largest_height

        # create a new height array and new QCloud array to get values at a higher level
        array_shape = list(height.shape)
        array_shape[1] = 1
        fake_array = np.zeros(array_shape)
        new_height = np.concatenate([height[:, 1:, :, :], fake_array], axis=1)
        new_QCloud = np.concatenate([QCloud[:, 1:, :, :], fake_array], axis=1)

        # take out values with these two indexes
        valid_height_value = height[valid_index]
        valid_QCloud_value = QCloud[valid_index]
        invalid_height_value = new_height[valid_index]
        invalid_QCloud_value = new_QCloud[valid_index]

        # interpolate
        QCloud_gradient = (invalid_height_value - valid_height_value) / (invalid_QCloud_value - valid_QCloud_value)
        delta_height = (self._qcloud_threshold - valid_QCloud_value) * QCloud_gradient
        # if delta_height is larger than 50, change it to 50
        delta_height[delta_height > 50] = 50.

        # put it to the original height
        height[valid_index] = height[valid_index] + delta_height

        # change datas which index is False to nan
        height[~seafog_index] = np.nan

        # find the largest height value, which is seafog's top
        height = np.nanmax(height, axis=1)

        # get seafog's bottom
        QCloud_surface = QCloud[:, 0, :, :]

        # create DataArray and return
        longitude = wrfout["XLONG"].to_numpy()[0]
        latitude = wrfout["XLAT"].to_numpy()[0]
        time = wrfout["XTIME"].to_numpy()

        longitude_index = np.arange(longitude.shape[1])
        latitude_index = np.arange(longitude.shape[0])

        create_time = ctime()
        return Dataset(data_vars={
            "seafog": DataArray(name="seafog", data=height, dims=["time", "latitude_index", "longitude_index"], coords={
                "time": time,
                "latitude_index": latitude_index,
                "longitude_index": longitude_index
            }, attrs={
                "units": "m",
                "long_name": "height of seafog's top from WRF",
                "standard_name": "seafog",
                "description": f"height of seafog's top from WRF, calculated by package seafog at {create_time}"
            }),
            "seafog_surface": DataArray(name="seafog_surface", data=QCloud_surface, dims=["time", "latitude_index", "longitude_index"], coords={
                "time": time,
                "latitude_index": latitude_index,
                "longitude_index": longitude_index
            }, attrs={
                "units": "g/kg",
                "long_name": "QCLOUD at surface from WRF",
                "standard_name": "seafog_surface"
            }),

            "longitude": DataArray(name="longitude", data=longitude, dims=["latitude_index", "longitude_index"], coords={
                "latitude_index": latitude_index,
                "longitude_index": longitude_index
            }, attrs={
                "units": "degree",
                "long_name": "longitude",
                "standard_name": "longitude"
            }),
            "latitude": DataArray(name="latitude", data=latitude, dims=["latitude_index", "longitude_index"], coords={
                "latitude_index": latitude_index,
                "longitude_index": longitude_index
            }, attrs={
                "units": "degree",
                "long_name": "latitude",
                "standard_name": "latitude"
            })
        }, attrs={
            "description": f"height of seafog's top and area of seafog bottom from WRF, calculated by package seafog at {create_time}"
        })

    def seafog_top_daily(self, VIS: np.ndarray, zenith_cosine: np.ndarray) -> np.ndarray:
        """
        calculate the top height of seafog in the daytime.

        :param VIS: VIS band data.
        :param zenith_cosine: cosine value of the solar zenith angle.
        :return:
        """
        return -45.6 + 84.3 * np.power(VIS / 100 * zenith_cosine / ((1 - VIS / 100) * self._beta), 0.5)

    def daytime_seafog(self, IR4: DataArray, sst: DataArray, fillnan=True) -> DataArray:
        """
        detect seafog based on the difference between IR4 and sst data.

        :param fillnan: if remove non-seafog area.
        :param IR4: IR4 band data.
        :param sst: SST data.
        :return: difference value of IR4 and SST data.
                 if ``fillnan=True``, non-seafog area will be removed.
        """
        # check data dimensions
        assert (IR4.shape[0] == sst.shape[0]), "The shape of IR4 and SST data must be the same"
        assert (IR4.shape[1] == sst.shape[1]), "The shape of IR4 and SST data must be the same"
        IR_diff = IR4 - sst - 4.0 - 273.16
        if fillnan:
            index = np.logical_and(IR_diff >= self._ir_sst_diff_range[0], IR_diff <= self._ir_sst_diff_range[1])
            return IR_diff.where(index)
        else:
            return IR_diff

    def night_seafog(self, date: str, IR1: DataArray, IR4: DataArray) -> DataArray:
        """
        detect seafog area and thickness based on the IR1, IR4 and VIS satellite data and GOOS sst data.

        :param date: UTC date of the data, for example, '2020-05-19 09:13'.
        :param IR1: IR1 data.
        :param IR4: IR4 data.
        :return: detected seafog height.
        """
        # check data dimensions
        assert (IR1.shape[0] == IR4.shape[0]), "The shape of IR1 and IR4 data must be the same"
        assert (IR1.shape[1] == IR4.shape[1]), "The shape of IR1 and IR4 data must be the same"

        # get longitude and latitude
        longitude = IR1['longitude'].to_numpy()
        latitude = IR1['latitude'].to_numpy()
        # meshgrid
        longitude, latitude = np.meshgrid(longitude, latitude)
        # get solar altitude and zenith angle
        solar_altitude, solar_zenith = solar_altitude_zenith(date, longitude, latitude)
        # we only detect seafog at night
        if not (solar_zenith[-1, 0] >= 90 and solar_zenith[-1, -1] >= 90):
            logger.error(f"`night_seafog` can only detect seafog at night. Use `daytime_seafog` to detect diurnal seafog.")
            raise ValueError(f"`night_seafog` can only detect seafog at night. Use `daytime_seafog` to detect diurnal seafog.")
        # create an empty array to store seafog height, with nan means there is no seafog
        seafog = np.full(IR1.shape, np.nan)
        # diff IR4 and IR1 to take out seafog
        diff_IR4_IR1 = IR4.to_numpy() - IR1.to_numpy()

        index = np.logical_and(diff_IR4_IR1 > self.diff_range_night_IR[0],
                               diff_IR4_IR1 < self.diff_range_night_IR[1])
        seafog[index] = 100 * (-2.12 + 1.91 * np.abs(diff_IR4_IR1[index]) / 2)

        # the height of seafog should less than 600
        seafog[seafog > self._maximum_height] = np.nan
        # remove single point
        seafog = remove_single_point(seafog)
        # the seafog detection has been done

        return DataArray(name='seafog', data=seafog, dims=IR1.dims, coords=IR1.coords, attrs={
            'units': 'm',
            'long_name': "height of seafog's top",
            'standard_name': 'seafog',
            'description': f"height of seafog's top, calculated by package seafog at {ctime()}"
        })


detect_seafog = Seafog()

__all__ = ['detect_seafog']
