"""
seafog.goos_sst provides methods to download NearGOOS data from `JMA <https://www.data.jma.go.jp/gmd/goos/data/pub/JMA-product/>`_ and reads it.
"""

from datetime import datetime
from os import makedirs
from os.path import exists
from typing import Tuple, Callable

import numpy as np
from rich.progress import Progress
from xarray import DataArray

from .utils import decompress_file, download_url, logger

# For 0.25 degree GOOS data
ROOT_URL = "https://www.data.jma.go.jp/gmd/goos/data/pub/JMA-product/"
# the {} for data year
GOOS_SST_URL = "mgd_sst_glb_D/{}/"
# the {} for data date.
# for example, mgd_sst_glb_D20220101.txt.gz
GOOS_RE_SST_NAME_TEMPLATE = "re_mgd_sst_glb_D{}.txt.gz"
GOOS_SST_NAME_TEMPLATE = "mgd_sst_glb_D{}.txt.gz"

# For 0.1 degree High-GOOS data
# the {} for data year
HIGH_GOOS_SST_URL = "him_sst_pac_D/{}/"
# the {} for data date.
# for example, him_sst_pac_D20220615.txt
HIGH_GOOS_NAME_TEMPLATE = "him_sst_pac_D{}.txt"


def _generate_data_info(date: str, save_path: str, resolution="low") -> Tuple[str, str, str]:
    """
    generate data filename, download url, and the save path.
    
    :param date: for example, "2022-05-19 12:00", UTC time
    :param save_path: path to save data.
    :param resolution: `low`: 0.25 degrees; `high`: 0.1 degree.
    :return: tuple(filename, download url, save path)
    """
    # generate date time
    date = datetime.strptime(date, "%Y-%m-%d %H:%M").strftime("%Y%m%d")

    # generate filename, url and save path
    if resolution == "low":
        if int(date[:4]) >= 2022:
            filename = GOOS_SST_NAME_TEMPLATE.format(date)
            url = ROOT_URL + GOOS_SST_URL.format(date[:4]) + filename
        else:
            filename = GOOS_RE_SST_NAME_TEMPLATE.format(date)
            url = ROOT_URL + GOOS_SST_URL.format(date[:4]) + filename
        data_path = save_path + filename

    elif resolution == "high":
        filename = HIGH_GOOS_NAME_TEMPLATE.format(date)
        url = ROOT_URL + HIGH_GOOS_SST_URL.format(date[:4]) + filename
        data_path = save_path + filename

    else:
        logger.error(f"Unknown resolution: {resolution}. Valid values: ['low', 'high']")
        raise ValueError

    return filename, url, data_path


def goos_sst_parser(sst_filename: str, resolution="low") -> DataArray:
    """
    read NearGOOS sst data.
    
    :param sst_filename: GOOS sst data filename
    :param resolution: `low`: 0.25 degree; `high`: 0.1 degree.
    :return: sst data stored in DataArray, with latitude and longitude and nan value (where is 999 and 88.8)
    """
    # check file
    assert exists(sst_filename), f"file {sst_filename} does not exist"

    # read all lines in input file
    with open(sst_filename, 'r') as f:
        inputs = f.readlines()
    # the first line is date
    inputs = inputs[1:]

    # parse data, every three characters is a number
    data = []
    for line in inputs:
        # remove the \n at the end of line
        line = line[:-1]
        data.append([int(line[i:i + 3]) for i in range(0, len(line), 3)])

    data = np.asarray(data).astype(float) / 10
    # remove invalid data which is larger than 80
    data[data > 80] = np.nan
    # turn data upside down
    data = data[::-1, :]

    # generate longitude and latitude
    if resolution == "low":
        longitude = np.linspace(0.125, 359.875, 1440)
        latitude = np.linspace(-89.875, 89.875, 720)

    elif resolution == "high":
        longitude = np.linspace(100.05, 179.95, 800)
        latitude = np.linspace(0.05, 59.95, 600)

    else:
        logger.error(f"Unknown resolution: {resolution}. Valid values: ['low', 'high']")
        raise ValueError

    # generate xarray data array
    data = DataArray(name='sst', data=data, dims=['latitude', 'longitude'], coords={
        'latitude': latitude,
        'longitude': longitude
    }, attrs={
        'units': 'degree'
    })

    return data


def goos_sst_find_data(date: str, save_path: str, resolution="low", proxy_host: str = None, proxy_port: int = None, progress: Progress = None, headers: dict = None,
                       show_progress=True, callback: Callable = None) -> str:
    """
    download NearGOOS SST data and save it to the specified path.
    
    :param date: for example, "2022-05-19 12:00", UTC time.
    :param save_path: path to save data.
    :param resolution: `low`: 0.25 degrees; `high`: 0.1 degrees.
    :param proxy_host: http and https proxy server address.
    :param proxy_port: http and https proxy server port.
    :param progress: ``rich.progress.Progress`` object to display progress bar.
    :param headers: self-defined http headers.
    :param show_progress: if True, display download progress in the console.
    :param callback: callable object that will be called every iteration during data download.
                     ``callback`` should accept two params, `total_size` and `step_size`
    :return: data path.
    
    download NearGOOS data:
    
    >>> goos_sst_find_data("2024-07-29 00:00", save_path="data", show_progress=False)
    'data/mgd_sst_glb_D20240729.txt'

    download HighGOOS data:
    
    >>> goos_sst_find_data("2024-07-29 00:00", save_path="data", resolution="high", show_progress=False)
    'data/him_sst_pac_D20240729.txt'

    """
    # check the save path, if it doesn't exist, download data directly
    if save_path[-1] != '/':
        save_path += '/'

    if not exists(save_path):
        makedirs(save_path)

    # parse date and determine data file name
    data_name, url, data_path = _generate_data_info(date, save_path, resolution=resolution)

    # check if file exists
    if exists(data_path[:-3]):
        return data_path[:-3]

    # download data and decompress it
    # logger.debug(f"Downloading file from {url}")
    code = download_url(url, save_path, data_name, proxy_host=proxy_host, proxy_port=proxy_port,
                        headers=headers, show_progress=show_progress, progress=progress, callback=callback)
    if code == 404:
        logger.warning(f"The file {data_name} doesn't exist in server (status 404), following download tasks will be terminated")
        # generate empty results
        return ""
    elif code != 200:
        logger.error(f"Failed to download file {data_name}, status code is {code}. May be you can try again later, or check if this url is right: {url}")
        raise ConnectionError

    # No need to decompress for High-GOOS
    if resolution == "low":
        decompress_file(data_path)
        data_path = data_path[:-3]

    return data_path


__all__ = ['goos_sst_parser', 'goos_sst_find_data']
