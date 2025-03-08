"""
Created on Thu Jun  1 16:20 2023
This Python file is used to read data from pgm picture and calculate the corresponding brightness temperature with the loop up table file
"""

from datetime import datetime
from os import makedirs
from os.path import exists
from time import sleep
from typing import Tuple, Callable, Iterable, Union

import numpy as np
from pgm_reader import Reader
from rich.progress import Progress
from xarray import DataArray

from .res import get_default_table_file, VIS_table_file
from .utils import decompress_file, download_url, logger

ROOT_URL = "http://weather.is.kochi-u.ac.jp/sat"
# GAME/{year}/{month}/{data-type}
PGM_FILE_URL = "GAME/{}/{}/{}"
# {header}{time-string}{data-type}.pgm.gz, i.e. MT1R10122010IR1.pgm.gz
PGM_FILE_NAME = "{}{}{}.pgm.gz"
# CAL/{year}/{month}
TABLE_FILE_URL = "CAL/{}/{}"
# {header}{time-string}CAL.dat.gz, i.e. MTS211050106CAL.dat.gz
TABLE_FILE_NAME = "{}{}CAL.dat.gz"
# from now on the header change from MT1R to MTS2
HEADER_CHANGE_1 = datetime(2010, 12, 22, 3, 0)
# from now on the header change from MTS2 to HMW8
HEADER_CHANGE_2 = datetime(2015, 7, 1, 2)

# month dict
MONTH = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun', '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}


def _generate_data_info(date: datetime, header: str, data_type: str, save_path: str) -> Tuple[str, str, str]:
    """
    generate data filename, download url and save path.

    :param date: for example, 2022-05-19 12:00, UTC time
    :param header: filename header
    :param data_type: valid value: IR1, IR2, IR3, IR4, VIS
    :param save_path: data save path
    :return: tuple(filename, download url, save path)
    """
    pre_url = PGM_FILE_URL.format(date.strftime("%Y"), MONTH[date.strftime("%m")], data_type)
    filename = PGM_FILE_NAME.format(header, date.strftime("%Y%m%d%H")[2:], data_type)
    url = f"{ROOT_URL}/{pre_url}/{filename}"
    save_path = f"{save_path}/{filename}"

    return filename, url, save_path


def _generate_table_info(date: datetime, header: str, save_path: str) -> Tuple[str, str, str]:
    """
    generate table filename, download url and save path.

    :param date: for example, 2022-05-19 12:00, UTC time
    :param header: filename header
    :param save_path: data save path
    :return:tuple(filename, download url, save path)
    """
    pre_url = TABLE_FILE_URL.format(date.strftime("%Y"), MONTH[date.strftime("%m")])
    filename = TABLE_FILE_NAME.format(header, date.strftime("%Y%m%d%H")[2:])
    url = f"{ROOT_URL}/{pre_url}/{filename}"
    save_path = f"{save_path}/{filename}"

    return filename, url, save_path


def get_file_header(date: datetime) -> str:
    """
    get the data file header based on data date.

    :param date: data date
    :return: data filename header
    """
    if date < HEADER_CHANGE_1:
        return "MT1R"
    elif HEADER_CHANGE_1 <= date < HEADER_CHANGE_2:
        return "MTS2"
    else:
        return "HMW8"


def pgm_parser(pgm_filename: str, table_filename: str, data_type='IR1') -> DataArray:
    """
    Parse the pgm file and calculate the corresponding brightness temperature with the loop up table file.
    For VIS table calculation, VIS data is calibrated with 0:=0% and 255:=100% (albedo) and can be linearly interpolated.
    See http://weather.is.kochi-u.ac.jp/archive-e.html

    :param pgm_filename: PGM filename
    :param table_filename: table filename
    :param data_type: 'IR1', 'IR4', 'VIS' or 'VIS-b'.
    Note that if the data type is 'VIS', you need to give specific VIS temperature loop up table, or you can use 'VIS-b' instead to use built-in table.
    :return: PGM data.
    """
    assert exists(pgm_filename), f"File Not Exists: {pgm_filename}"
    if data_type != "VIS-b":
        assert exists(table_filename), f"File Not Exists: {table_filename}"

    # if the data type is VIS-b, use built-in table file
    if data_type == 'VIS-b':
        logger.debug(f'Use built in VIS look up table which is in: {VIS_table_file}, you can change it as you like')
        table_filename = VIS_table_file

    # check if table file needs to process
    with open(table_filename, 'r') as f:
        table_data = f.readlines()
    # try to extract the table value
    try:
        table_value = {i: float(table_data[i][:-1]) for i in range(len(table_data))}
    except ValueError:
        # raw table file, need to get the corresponding value
        logger.debug('Raw table file, need to be processed')
        logger.debug(f"Your data type is {data_type}")
        # check if the data type is VIS
        if 'VIS' not in data_type:
            # get the corresponding data type line
            table_data = [x for x in table_data if data_type in x]
            # remove the first useless line
            table_data = table_data[1:]
            # split and get the latest string as the value
            table_data = [x.split()[-1] for x in table_data]
            table_value = {i: float(table_data[i]) for i in range(len(table_data))}
        else:
            logger.debug(f'Use built in VIS look up table which is in: {VIS_table_file}, you can change it as you like')
            # read pre-calculated value
            with open(VIS_table_file, 'r') as f:
                table_data = f.readlines()
                table_value = {i: float(table_data[i][:-1]) for i in range(len(table_data))}

    # read pgm file
    reader = Reader()
    data = reader.read_pgm(pgm_filename)
    new_data = data.copy().astype(float)
    # get the corresponding temperature
    keys = list(table_value.keys())
    for key in keys:
        new_data[data == key] = table_value[key]

    # turn data upside down
    new_data = new_data[::-1, :]

    # generate longitude and latitude
    longitude = np.linspace(70.15, 160.15, 1800)
    latitude = np.linspace(-20.15, 69.85, 1800)
    # generate xarray DataArray object
    new_data = DataArray(name=data_type, data=new_data, dims=['latitude', 'longitude'], coords={
        'latitude': latitude,
        'longitude': longitude
    }, attrs={
        'units': 'K'
    })

    return new_data


def pgm_find_data(date: str, save_path: str, data_type: Union[str, Iterable[str]] = 'default', built_in=True, proxy_host: str = None, proxy_port: int = None,
                  progress: Progress = None, headers: dict = None, show_progress=True, callback: Callable = None) -> Tuple[str, ...]:
    """
    Use this function to find data in the specify data path.
    If data is not find, this will download it automatically.

    :param date: for example, 2022-05-19 12:00, UTC time
    :param save_path: data save path
    :param data_type: valid value: IR1, IR2, IR3, IR4, VIS, all, default (IR1, IR4 and VIS), or a list contains multiple data types.
    :param built_in: if table file doesn't exist in server, use built-in table file
    :param proxy_host: http and https proxy server address
    :param proxy_port: http and https proxy server port
    :param progress: Progress object to display download progress
    :param headers: Self-defined http headers.
    :param show_progress: If True, display download progress in console.
    :param callback: Callable object that will be called every iteration of `res.iter_content`.
    `callback` should accept two params: `total_size` and `step_size`
    :return: data paths, including the data of your input type and their table file

    Download IR1, IR4 and VIS data:

    >>> pgm_find_data("2024-07-29 00:00", "data", show_progress=False)
    ('data/HMW824072900IR1.pgm', 'data/HMW824072900IR4.pgm', 'data/HMW824072900VIS.pgm', 'data/HMW824072900CAL.dat')

    Download IR 1-4 data:

    >>> pgm_find_data("2024-07-29 00:00", "data", data_type=["IR1", "IR2", "IR3", "IR4"], show_progress=False)
    ('data/HMW824072900IR1.pgm', 'data/HMW824072900IR2.pgm', 'data/HMW824072900IR3.pgm', 'data/HMW824072900IR4.pgm', 'data/HMW824072900CAL.dat')

    """
    # check data path, if it doesn't exist, download data directly
    if not exists(save_path):
        makedirs(save_path)

    # get data type and corresponding data header
    date = datetime.strptime(date, "%Y-%m-%d %H:%M")
    if data_type == 'default':
        data_type_list = ['IR1', 'IR4', 'VIS']
    elif data_type == 'all':
        data_type_list = ['IR1', 'IR2', 'IR3', 'IR4', 'VIS']
    elif isinstance(data_type, Iterable):
        data_type_list = data_type
    else:
        data_type_list = [data_type]
    header = get_file_header(date)

    # store data path
    data_path_list = []
    # store missing data name and url
    miss_data_info = {}

    # find exist data
    for data_type in data_type_list:
        filename, file_url, file_path = _generate_data_info(date, header, data_type, save_path)
        if not exists(file_path[:-3]):
            miss_data_info[filename] = file_url
        else:
            data_path_list.append(file_path[:-3])

    # download missing data
    if len(miss_data_info) > 0:
        for filename in miss_data_info:
            # download file and decompress it
            # logger.debug(f"Downloading file from {miss_data_info[filename]}")
            code = download_url(miss_data_info[filename], save_path, filename, proxy_host=proxy_host, proxy_port=proxy_port, headers=headers,
                                show_progress=show_progress, progress=progress, callback=callback)
            if code == 404:
                logger.warning(f"The file {filename} doesn't exist in server (status 404), following download tasks will be terminated")
                # generate empty results
                return tuple(["" for _ in data_type_list] + [""])
            elif code != 200:
                logger.error(f"Failed to download file {filename}, status code is {code}. "
                             f"May be you can try again later, or check if this url is right: {miss_data_info[filename]}")
                raise ConnectionError

            decompress_file(f"{save_path}/{filename}")

            data_path_list.append(f"{save_path}/{filename[:-3]}")

            # sleep one second for humanitarianism
            sleep(0.5)

    data_path_list.sort()

    # find exist table file
    filename, table_url, table_path = _generate_table_info(date, header, save_path)
    # download table file
    if not exists(table_path[:-3]):
        # logger.debug(f"Downloading file from {table_url}")
        code = download_url(table_url, save_path, filename, proxy_host=proxy_host, proxy_port=proxy_port, headers=headers,
                            show_progress=show_progress, progress=progress, callback=callback)
        if code == 404 or code == -1:
            if built_in:
                logger.warning(f"The file {filename} doesn't exist in server (status {code}), use built-in table file by default. "
                               f"You can pass `built_in=False` to disable this feature")
                table_path = get_default_table_file(header)
                data_path_list.append(table_path)
        elif code != 200:
            logger.error(f"Failed to download file {filename}, status code is {code}. May be you can try again later, or check if this url is right: {table_url}")
            raise ConnectionError
        else:
            decompress_file(table_path)
            data_path_list.append(table_path[:-3])
    else:
        data_path_list.append(table_path[:-3])

    return tuple(data_path_list)


__all__ = ['pgm_parser', 'pgm_find_data']
