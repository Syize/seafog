"""
seafog.avhrr contains the function to download AVHRR daily SST (Sea Surface Temperature) data
from the `website <https://coastwatch.pfeg.noaa.gov/erddap/griddap/nceiPH53sstd1day.html>`_
"""

from datetime import datetime, timedelta
from json import loads, dumps
from os import makedirs
from os.path import exists
from time import time
from typing import Callable

from rich.progress import Progress

from .res import AVHRR_CACHE_FOLDER
from .utils import download_url, logger


def avhrr_find_data(date: str, save_path: str, proxy_host: str = None, proxy_port: int = None, progress: Progress = None, headers: dict = None,
                    show_progress=True, callback: Callable = None) -> str:
    """
    download avhrr daily sst data from the `website <https://coastwatch.pfeg.noaa.gov/erddap/griddap/nceiPH53sstd1day.html>`_.

    :param date: UTC time, for example, "2020-01-01".
    :param save_path: the directory path to store downloaded data.
    :param proxy_host: host address of proxy server.
    :param proxy_port: port number of proxy server.
    :param progress: rich.progress.Progress object to display download progress.
    :param headers: self-defined http headers.
    :param show_progress: if True, display download progress in console.
    :param callback: callable object that will be called during data download.
                     it can be a function or any other Python object.
                     `callback` should accept two params: `total_size` and `step_size`.
    :return: path of downloaded data.

    download data:

    >>> avhrr_find_data("2020-01-01", "data", show_progress=False) # set show_progress=False will disable the progress bar.
    'data/20200101141457-NCEI-L3C_GHRSST-SSTskin-AVHRR_Pathfinder-PFV5.3_NOAA19_G_2020001_day-v02.0-fv01.0.nc'

    """
    # for example, https://coastwatch.pfeg.noaa.gov/erddap/files/nceiPH53sstd1day/2014/data/
    ROOT_URL = "https://coastwatch.pfeg.noaa.gov/erddap/files/nceiPH53sstd1day/{}/data/"
    # check cache folder
    if not exists(AVHRR_CACHE_FOLDER):
        makedirs(AVHRR_CACHE_FOLDER)
    # parse date
    date = datetime.strptime(date, "%Y-%m-%d")
    # read file lists
    cache_file = f"{AVHRR_CACHE_FOLDER}/{date.year}.json"
    # if not exists, download cache
    exist_flag = exists(cache_file)
    # if cache is out of date, refresh it
    refresh_flag = False
    # # store update_time
    update_time = {}
    # # expire time
    expire_time = timedelta(hours=1)
    if not exists(f"{AVHRR_CACHE_FOLDER}/update_time"):
        refresh_flag = True
    else:
        with open(f"{AVHRR_CACHE_FOLDER}/update_time", "r") as f:
            update_time = loads(f.read())
        if str(date.year) not in update_time:
            refresh_flag = True
        elif datetime.fromtimestamp(time()) - datetime.strptime(update_time[str(date.year)], "%Y-%m-%d %H") > expire_time:
            refresh_flag = True
    if not exist_flag or refresh_flag:
        url = ROOT_URL.format(date.year) + ".json"
        logger.debug(f"Download {date.year}'s file list cache")
        status = download_url(url, AVHRR_CACHE_FOLDER, f"{date.year}.json", proxy_host=proxy_host, proxy_port=proxy_port,
                              headers=headers, show_progress=show_progress, progress=progress)
        if status != 200:
            logger.warn(f"Fail to download file lists! Use old file list.")
        else:
            # update update_time
            update_time[str(date.year)] = datetime.fromtimestamp(time()).strftime("%Y-%m-%d %H")
            with open(f"{AVHRR_CACHE_FOLDER}/update_time", "w") as f:
                f.write(dumps(update_time))
    # read file lists from json
    # # check the cache file
    if not exists(cache_file):
        return ""
    with open(cache_file, 'r') as f:
        file_list = loads(f.read())['table']['rows']
    # find the corresponding filename and its size
    filename = None
    size = None
    for file in file_list:
        if file[0][:8] == date.strftime("%Y%m%d"):
            filename = file[0]
            size = file[2]
    # check
    if filename is None:
        logger.error(f"Data at {date.strftime('%Y-%m-%d')} not found")
        return ""
    # download data
    if not exists(save_path):
        makedirs(save_path)
    # if the file exists, return
    if exists(f"{save_path}/{filename}"):
        return f"{save_path}/{filename}"

    url = ROOT_URL.format(date.year) + filename
    # logger.debug(f"Download data from {url}")
    status = download_url(url, save_path, filename, proxy_host=proxy_host, proxy_port=proxy_port, size=size, headers=headers,
                          show_progress=show_progress, progress=progress, callback=callback)
    if status != 200:
        logger.error(f"Fail to download data from {url}")
        return ""

    return f"{save_path}/{filename}"


__all__ = ['avhrr_find_data']
