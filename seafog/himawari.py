"""
seafog.himawari provides methods to download Himawari-8/9 data from PTREE and to plot true color satellite image.
"""

from datetime import datetime
from os import makedirs
from os.path import exists
from typing import Callable, Tuple, Union

import numpy as np
from cartopy import crs
from cartopy.mpl.geoaxes import GeoAxes
from numpy import ndarray
from rich.progress import Progress
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
from xarray import Dataset

from .utils import download_ftp, logger


def convert_satellite_data(satellite_data: ndarray, min_value: float = 0, max_value: float = 1) -> ndarray:
    """
    transform satellite data from albedo value to RGB value (0-255) with segmentation mapping.
    
    :param satellite_data: satellite data
    :param min_value: the minimum albedo value that will be mapped to 0
    :param max_value: the maximum albedo value that will be mapped to 255
    :return: RGB values.

    >>> data = np.linspace(0, 1, 10)
    >>> convert_satellite_data(data)
    array([  0, 103, 154, 180, 204, 219, 231, 241, 248, 255])

    """
    satellite_data = (satellite_data - min_value) / (max_value - min_value) * 255
    # satellite_data = satellite_data.astype(int)

    # segmentation mapping
    x = [0, 30, 60, 120, 190, 255]
    y = [0, 110, 160, 210, 240, 255]
    interpolator = interp1d(x, y, bounds_error=False, fill_value=255)

    return interpolator(satellite_data).astype(int)


def up_sampling(data: ndarray, target_size: Tuple[int, int]) -> ndarray:
    """
    up sampling a image.
    
    :param data: image data.
    :param target_size: target shape.
    :return: new image data.
    """
    zoom_scale = []
    for raw_len, target_len in zip(data.shape, target_size):
        zoom_scale.append(target_len / raw_len)

    return zoom(data, zoom_scale, order=3)


def plot_himawari_vis(ax: GeoAxes, dataset: Dataset, area_range: Union[Tuple[float, ...], Tuple[int, ...], None] = None, *args, **kwargs):
    """
    plot VIS image with giving satellite data.
    
    :param ax: cartopy axes object.
    :param dataset: himawari data.
    :param area_range: area range, (lon_min, lon_max, lat_min, lat_max).
                       if None, plot the whole area.
    :param args: other params that will be passed to ``seafog.convert_satellite_data``.
    :param kwargs: other params that will be passed to ``seafog.convert_satellite_data``.
    :return:
    """
    # check axes
    assert isinstance(ax, GeoAxes), "`ax` must be cartopy.mpl.geoaxes.GeoAxes so we can draw VIS image."

    # check data bands
    for band in ["albedo_01", "albedo_02", "albedo_03"]:
        assert band in dataset.data_vars, f"We need `{band}` band to draw VIS image."

    # check area_range
    if area_range is not None:
        # extract area range
        lon_min, lon_max, lat_min, lat_max = area_range
    else:
        lon_min, lon_max, lat_min, lat_max = None, None, None, None

    # area index
    index = (slice(lat_max, lat_min), slice(lon_min, lon_max))

    # extract RGB bands
    blue_band = dataset["albedo_01"].loc[index]
    green_band = dataset["albedo_02"].loc[index]
    red_band = dataset["albedo_03"].loc[index]

    # extract longitude and latitude
    longitude = blue_band["longitude"]
    latitude = blue_band["latitude"]

    # create an RGB image array
    blue_value = convert_satellite_data(blue_band.to_numpy(), *args, **kwargs)
    green_value = convert_satellite_data(green_band.to_numpy(), *args, **kwargs)
    red_value = convert_satellite_data(red_band.to_numpy(), *args, **kwargs)
    vis_rgb = np.dstack((red_value, green_value, blue_value)) / 255

    # create a color map
    vis_color = vis_rgb.reshape(-1, 3)

    # plot
    ax.pcolormesh(longitude, latitude, vis_rgb, color=vis_color, shading="nearest", transform=crs.PlateCarree())


def himawari_find_data(date: str, save_path: str, resolution: str = "low", area: str = "full", band_num: str = "auto", ftp_setting: dict = None, progress_num=1,
                       progress: Progress = None, show_progress=True, callback: Callable = None, timeout=720, connection_timeout=30, force_download=False) -> str:
    """
    Download himawari-8/9 data from the official website.
    This function will search the data in the ``save_path`` first.
    If the data exists, it will skip the download.
    You can set `force_download=True` to force to re-download the data.

    :param date: Date string in format "%Y-%m-%d %H:%M".
    :type date: str
    :param save_path: The directory to store data.
    :type save_path: str
    :param resolution: Himawari data's resolution, valid value: ``["low", "high"]``.
                       The actual resolution will be different depending on the ``area`` you chose.
                       You can refer to the `seafog wiki <http://gitea.seafog.syize.cn/seafog/seafog-plot/wiki/himawari8#himawari8_find_data>`_ for more information.
    :type resolution: str
    :param area: Himawari data's area, valid value: ``["full", "japan"]``.
    :type area: str
    :param band_num: Band numbers in the data, valid value: ``["21", "04", "14", "auto"]``.
                     The band number will change with the ``area``, ``full`` is ``"21"``, ``japan`` is ``"04"`` or ``"14"``.
                     If ``band_num == "auto"``, the usable highest number will be used.
    :type band_num: str
    :param ftp_setting: A dict contains the following keys: ``["username", "password", "proxy_host", "proxy_port"]``, in which ``["proxy_host", "proxy_port"]`` are optional.
    :type ftp_setting: dict
    :param progress_num: The number of processes which will be used to download the data.
                         Increase this number can help make download speed faster.
                         It won't be larger than ``cpu_cores – 2``.
    :type progress_num: int
    :param progress: ``rich.progress.Progress`` object to display progress bar.
    :type progress: Progress
    :param show_progress: If True, display download progress in the console.
    :type show_progress: bool
    :param callback: Callable object that will be called every iteration during data download.
                     ``callback`` should accept two params, `total_size` and `step_size`.
    :type callback: Callable
    :param timeout: The max time which could be used to download the data.
                    Download will be terminated once the max time is reached.
                    Set it to -1 for no limitation.
                    Units: seconds.
    :type timeout: int
    :param connection_timeout: The max time which will be used to connect to the server.
                               Units: seconds.
    :type connection_timeout: int
    :param force_download: If ignore the existing data and re-download it.
    :type force_download: bool
    :return: Data file path.
    :rtype: str
    """
    # for example, NC_H08_20150707_0000_r04_FLDK.05401_05201.nc
    NAME_TEMPLATE = "NC_H{}_{}_{}_{}_FLDK.{}_{}.nc"
    # for example: ftp://ftp.ptree.jaxa.jp/jma/netcdf/201507/07/NC_H08_20150707_0010_R21_FLDK.06001_06001.nc
    URL_TEMPLATE = "ftp://ftp.ptree.jaxa.jp/jma/netcdf/{}/{}/"

    resolution_dict = {
        'full': {
            'low': {'x': '02401', 'y': '02401'},
            'high': {'x': '06001', 'y': '06001'}
        },
        'japan': {
            'low': {'x': '02701', 'y': '02601'},
            'high': {'x': '05401', 'y': '05201'}
        }
    }

    # check date
    if datetime.strptime(date, "%Y-%m-%d %H:%M") < datetime(2015, 7, 7, 0, 0):
        logger.error(f"Date is earlier than 2015-07-07 00:00")
        raise ValueError

    # check area
    if area not in ['full', 'japan']:
        logger.error(f"Unknown area: {area}")
        raise ValueError

    # check resolution
    if resolution not in ['low', 'high']:
        logger.error(f"Unknown resolution: {resolution}")
        raise ValueError

    # check save path
    if not exists(save_path):
        makedirs(save_path)

    # check band number
    if band_num == 'auto':
        if area == 'full':
            band_num = "R21"
        else:
            band_num = "r14"
    else:
        if area == "full":
            band_num = "R21"
        else:
            if band_num not in ["04", "14"]:
                logger.error(f"Error band number: should be '04' or '14', but is {band_num}")
                raise ValueError
            band_num = f"r{band_num}"

    x_res = resolution_dict[area][resolution]['x']
    y_res = resolution_dict[area][resolution]['y']

    # pares date
    date = datetime.strptime(date, "%Y-%m-%d %H:%M")

    # generate url and save path
    satellite_num = "08" if date <= datetime(2022, 12, 12) else "09"
    filename = NAME_TEMPLATE.format(satellite_num, date.strftime("%Y%m%d"), date.strftime("%H%M"), band_num, x_res, y_res)
    data_path = f"{save_path}/{filename}"
    # check the existing data
    if not force_download and exists(data_path):
        return data_path

    url = URL_TEMPLATE.format(date.strftime("%Y%m"), date.strftime("%d")) + filename

    # parse ftp setting
    user = None
    passwd = None
    proxy_host = None
    proxy_port = None

    if ftp_setting is not None:
        for key in ftp_setting:
            if key == "username":
                user = ftp_setting[key]
            elif key == "password":
                passwd = ftp_setting[key]
            elif key == "proxy_host":
                proxy_host = ftp_setting[key]
            elif key == "proxy_port":
                proxy_port = ftp_setting[key]

    # inform user to apply P-Tree account
    if None in [user, passwd]:
        logger.error(f"You doesn't give your username and password of PTREE, which is necessary to download satellite data. You can apply an account in this website: "
                     f"https://www.eorc.jaxa.jp/ptree/registration_top.html")
        raise KeyError

    if not download_ftp(url, save_path, filename, user=user, passwd=passwd, proxy_host=proxy_host, proxy_port=proxy_port,
                        show_progress=show_progress, progress=progress, callback=callback, timeout=timeout, connection_timeout=connection_timeout,
                        progress_num=progress_num, force_download=force_download):
        raise ConnectionError

    return f"{save_path}/{filename}"


__all__ = ['convert_satellite_data', 'up_sampling', 'himawari_find_data', 'plot_himawari_vis']
