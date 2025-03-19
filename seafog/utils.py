"""
This Python file contains some utility functions for the project.
"""

import gzip
import logging
import sys
import warnings
from datetime import datetime
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from os import cpu_count, listdir, makedirs, remove, rename
from os.path import abspath, basename, dirname, exists, getsize
from shutil import unpack_archive
from time import sleep
from typing import Callable, Optional, Tuple, Union

import ephem
import numpy as np
import pycurl
import xarray as xr
from global_land_mask import is_land
from requests import get
from requests.exceptions import ConnectTimeout, ProxyError, RequestException, SSLError
from rich.logging import RichHandler
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

# close numpy RuntimeWarning
warnings.filterwarnings("ignore")

# init a logger
logger = logging.getLogger("seafog")
formatter = logging.Formatter("%(name)s :: %(message)s", datefmt="%m-%d %H:%M:%S")
# use rich handler
handler = RichHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.WARNING)
# remove handler in pgm reader and set handler
_logger = logging.getLogger("pgm reader")
for _handler in _logger.handlers:
    _logger.removeHandler(_handler)
_logger.addHandler(handler)

# define progress
PROGRESS = Progress(
    TextColumn("{task.description}"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•", DownloadColumn(),
    "•", TransferSpeedColumn(),
    "•", TimeRemainingColumn()
)

sun = ephem.Sun()  # type: ignore
observer = ephem.Observer()

# disguise requests.get as wget
HEADERS = {
    "User-Agent": "Wget/1.12 (linux-gnu)"
}


def _multi_process_env_check() -> bool:
    """
    On Windows and macOS, we need something like ``if __name__ == "__main__"`` to avoid the RuntimeError.
    So check the main entry file first.

    :return:
    :rtype:
    """
    if sys.platform == "linux":
        return True

    main_entry_file = sys.argv[0]
    with open(main_entry_file, 'r') as f:
        contents = f.read()

    if "if __name__ == '__main__':" not in contents or 'if __name__ == "__main__"' not in contents:
        return False
    else:
        return True


class CurlSingleProcessCallback:
    """
    This callback is used to update progress when download the data in the single process.

    """

    def __init__(self, curl_handle: pycurl.Curl, progress: Progress = None, filename: str = "", callback: Callable = None):
        """
        A callback class that will be used to trace the progress of FTP download.

        :param curl_handle: The curl object to download files.
                            We need to access this to stop downloading.
        :param progress: A Progress object to display download progress
        :param filename: Save filename.
        :param callback: Callable object that will be called every iteration of `res.iter_content`.
        `callback` should accept two params: `total_size` and `step_size`.
        `callback` can return a non-zero and non-None value to stop download.
        """
        self.curl = curl_handle
        self.progress: Progress = progress
        self.task_id = None
        self.callback: Callable = callback
        self.filename = filename
        self.last_downloaded_total = 0
        self.start_size = 0

    def set_progress(self, progress: Progress, filename=""):
        """
        Set progress info.
        :param progress:
        :type progress:
        :param filename:
        :type filename:
        :return:
        :rtype:
        """
        self.progress = progress
        self.filename = filename

    def set_start_size(self, size: int):
        """
        If we continue to download from .part file, we need to tell progress the size of download part.

        :param size: Download part's size.
        :return:
        """
        self.start_size = size

    def __call__(self, download_total: int, downloaded_total: int, upload_total: int, uploaded_total: int):
        # Check download_total and downloaded_total
        # If both are 0, download doesn't begin
        if download_total == 0 and downloaded_total == 0:
            return
        # Transform to kb
        download_total = download_total
        downloaded_total = downloaded_total
        # Calculate step size
        step_size = downloaded_total - self.last_downloaded_total
        self.last_downloaded_total = downloaded_total

        # Check task id. If it's None, create a new task
        if self.task_id is None and isinstance(self.progress, Progress):
            # create a task
            self.task_id = self.progress.add_task(f"[red]{self.filename}[red]", total=download_total)
            # set start size
            self.progress.update(self.task_id, advance=self.start_size)
            # update progress
            self.progress.update(self.task_id, advance=step_size)
        elif self.task_id is not None:
            self.progress.update(self.task_id, advance=step_size)
        if self.callback is not None:
            return_code = self.callback(download_total, step_size)
            # check return code
            if return_code is not None and return_code != 0:
                logger.info(f"Stop downloading by callback")
                # pause at first
                self.curl.pause(pycurl.PAUSE_ALL)
                # just return non-zero value can stop downloading
                return return_code


class CurlMultiProcessCallback:
    """
    This callback is used to send download progress to the main progress when download the file in multiple processes.

    """

    def __init__(self, send_pipe: Connection, downloaded_size=0, restart=False):
        self.send_pipe = send_pipe
        self.last_downloaded = downloaded_size
        if downloaded_size != 0 and not restart:
            self.send_pipe.send(downloaded_size)

    def __call__(self, download_total: int, downloaded_total: int, upload_total: int, uploaded_total: int):
        if download_total == 0 and downloaded_total == 0:
            return
        else:
            step_size = downloaded_total - self.last_downloaded
            self.last_downloaded = downloaded_total
            self.send_pipe.send(step_size)


class CurlDownloader:
    """
    A pycurl wrapper class to download files in single or multiple processes.

    """

    def __init__(self, url: str, save_path: str, filename: str, proxy_host: Optional[str] = None, proxy_port: Optional[int] = None, timeout=720, connection_timeout=30):
        self.url = url
        self.save_path = save_path
        self.filename = filename
        self.proxy_host = proxy_host
        self.proxy_port = proxy_port
        self.timeout = timeout
        self.connection_timeout = connection_timeout

    def multiple_process(self, part_filename: str, start: int, end: int, send_pipe: Connection, force_download: bool):
        """
        This function should be called in another process, and it is used to download part of the file.
        It can automatically restart the task up to five times if download is failed.

        :param part_filename: Filename of the temporal file.
        :type part_filename: str
        :param start: Start point to download.
        :type start: int
        :param end: End point of the part.
        :type end: int
        :param send_pipe: The pipe to send downloaded size.
        :type send_pipe: Connection
        :param force_download: If ignore the existing data and re-download it.
        :type force_download: bool
        :return:
        :rtype:
        """
        try_time = 1
        success_flag = 0
        restart = False

        while try_time <= 5 and not success_flag:
            try:
                self._multiple_process(part_filename, start, end, send_pipe, force_download, restart)
                success_flag = 1
            except pycurl.error as e:
                restart = True
                error_code = e.args[0]
                if error_code in [23, 42]:
                    logger.error(f"Stop download by the user")
                    try_time = 5
                else:
                    logger.warning(f"Subprocess exits because of '{e.args[1]}' (error code {e.args[0]}), retry ({try_time} times)")

            try_time += 1

        if not success_flag:
            logger.error(f"Failed to download the file")
            raise ConnectionError

    def _multiple_process(self, part_filename: str, start: int, end: int, send_pipe: Connection, force_download: bool, restart=False):
        """
        This function should be called in another process, and it is used to download part of the file.

        :param part_filename: Filename of the temporal file.
        :type part_filename: str
        :param start: Start point to download.
        :type start: int
        :param end: End point of the part.
        :type end: int
        :param send_pipe: The pipe to send downloaded size.
        :type send_pipe: Connection
        :param force_download: If ignore the existing data and re-download it.
        :type force_download: bool
        :param restart: If True, doesn't report the downloaded file size.
        :type restart: bool
        :return: True if download successfully, else False.
        :rtype: bool
        """
        # curl settings
        curl = pycurl.Curl()
        curl.setopt(pycurl.CONNECTTIMEOUT, self.connection_timeout)
        if self.timeout >= 0:
            curl.setopt(pycurl.TIMEOUT, self.timeout)
        curl.setopt(pycurl.URL, self.url)
        curl.setopt(pycurl.NOPROGRESS, False)
        if self.proxy_host is not None and self.proxy_port is not None:
            if not self.proxy_host.startswith("socks5"):
                logger.error(f"Only support `socks5` protocol")
                raise ValueError
            else:
                curl.setopt(pycurl.PROXY, self.proxy_host)
                curl.setopt(pycurl.PROXYPORT, self.proxy_port)
                curl.setopt(pycurl.PROXYTYPE, pycurl.PROXYTYPE_SOCKS5)

        save_path = f"{self.save_path}/{part_filename}"
        if not force_download and exists(save_path):
            downloaded_size = getsize(save_path)
            if downloaded_size > (end - start):
                open_type = "wb"
            else:
                open_type = "ab"
                start = start + downloaded_size
        else:
            downloaded_size = 0
            open_type = "wb"

        curl.setopt(pycurl.RANGE, f"{start}-{end}")
        curl.setopt(pycurl.XFERINFOFUNCTION, CurlMultiProcessCallback(send_pipe, downloaded_size, restart))

        with open(save_path, open_type) as f:
            curl.setopt(pycurl.WRITEDATA, f)
            curl.perform()

    def single_process(self, progress: Progress = None, callback: Callable = None, force_download=False) -> bool:
        """
        This function can be called directly to download file.
        It can automatically restart the task up to 5 times if download is failed.

        :param progress: 
        :type progress: Progress
        :param callback: 
        :type callback: Callable
        :param force_download: If ignore the existing data and re-download it.
        :type force_download: bool
        :return: True if download successfully, else False.
        :rtype: bool
        """
        try_time = 1
        success_flag = 0
        curl = pycurl.Curl()

        while try_time <= 5 and not success_flag:
            curl.reset()
            curl.setopt(pycurl.CONNECTTIMEOUT, self.connection_timeout)
            if self.timeout >= 0:
                curl.setopt(pycurl.TIMEOUT, self.timeout)
            curl.setopt(pycurl.URL, self.url)
            curl.setopt(pycurl.NOPROGRESS, False)
            progress_callback = CurlSingleProcessCallback(curl_handle=curl, progress=progress, filename=self.filename, callback=callback)
            curl.setopt(pycurl.XFERINFOFUNCTION, progress_callback)

            if self.proxy_host is not None and self.proxy_port is not None:
                if not self.proxy_host.startswith("socks5"):
                    logger.error(f"Only support `socks5` protocol")
                    raise ValueError
                else:
                    curl.setopt(pycurl.PROXY, self.proxy_host)
                    curl.setopt(pycurl.PROXYPORT, self.proxy_port)
                    curl.setopt(pycurl.PROXYTYPE, pycurl.PROXYTYPE_SOCKS5)

            save_path = f"{self.save_path}/{self.filename}"
            temp_save_path = f"{save_path}.part"
            if not force_download and exists(temp_save_path):
                downloaded_size = getsize(temp_save_path)
                curl.setopt(pycurl.RESUME_FROM, downloaded_size)
                progress_callback.set_start_size(downloaded_size)
                open_type = "ab"
            else:
                open_type = "wb"

            with open(temp_save_path, open_type) as f:
                curl.setopt(pycurl.WRITEDATA, f)

                try:
                    curl.perform()
                    success_flag = 1
                except pycurl.error as e:
                    error_code = e.args[0]
                    # pycurl's error is ugly
                    if error_code in [23, 42]:
                        logger.error(f"Stop download by the user")
                        try_time = 5
                    else:
                        logger.warning(f"Failed to download because of {e}, retry ({try_time + 1} times)")

            if success_flag:
                rename(temp_save_path, save_path)

            try_time += 1

        if success_flag:
            return True
        else:
            return False

    def start(self, process_num=1, progress: Progress = None, callback: Callable = None, force_download=False) -> bool:
        """
        Start to download the file in the single process or multiple processes.

        :param process_num: The number of processes used to download the file.
        :type process_num: int
        :param progress:
        :type progress: Progress
        :param callback:
        :type callback: Callable
        :param force_download: If ignore the existing data and re-download it.
        :type force_download: bool
        :return: True if download successfully, else False.
        :rtype: bool
        """
        if process_num < 2:
            return self.single_process(progress=progress, callback=callback)

        else:
            if not _multi_process_env_check():
                logger.error(
                    "If you are about to download the file with multiple processes, you need to wrap your code in the block: `if __name__ == '__main__':` "
                    "to avoid the `RuntimeError` from the multiprocessing package. For more information, please check the official documentation: "
                    "https://docs.python.org/3/library/multiprocessing.html#the-process-class"
                )
                raise RuntimeError(
                    "If you are about to download the file with multiple processes, you need to wrap your code in the block: `if __name__ == '__main__':` "
                    "to avoid the `RuntimeError` from the multiprocessing package. For more information, please check the official documentation: "
                    "https://docs.python.org/3/library/multiprocessing.html#the-process-class"
                )

            logger.info(f"You are downloading the file in Multi-Process mode")
            cpu_cores = cpu_count()
            if process_num > cpu_cores:
                process_num = cpu_cores - 2
                if process_num <= 0:
                    process_num = 1
                logger.warning(f"You machine has {cpu_cores} CPUs, so reduce process number to {process_num}")

            part_filenames = [f"{self.filename}.part.{i}" for i in range(process_num)]

            # check the old cache file
            res = []
            for _cache_file in part_filenames:
                if exists(f"{self.save_path}/{_cache_file}"):
                    res.append(True)
                else:
                    res.append(False)

            if exists(f"{self.save_path}/{self.filename}.part.{process_num}"):
                res.append(False)
            else:
                res.append(True)

            if True in res and False in res:
                logger.warning(f"Process number has changed after the last running, clear old cache file for the fresh download")
                cache_file_list = [x for x in listdir(self.save_path) if x.startswith(f"{self.filename}.part")]
                for _cache_file in cache_file_list:
                    remove(f"{self.save_path}/{_cache_file}")

            # get file size
            curl = pycurl.Curl()
            curl.setopt(pycurl.CONNECTTIMEOUT, self.connection_timeout)
            if self.timeout >= 0:
                curl.setopt(pycurl.TIMEOUT, self.timeout)
            curl.setopt(pycurl.URL, self.url)
            curl.setopt(pycurl.NOPROGRESS, True)
            curl.setopt(pycurl.NOBODY, 1)
            curl.perform()
            file_size = int(curl.getinfo(pycurl.CONTENT_LENGTH_DOWNLOAD))

            if isinstance(progress, Progress):
                pid = progress.add_task(f"[red]{self.filename}[red]", total=file_size)
            else:
                pid = None

            chunk_size = file_size // process_num
            process_pool = []
            recv_pipe, send_pipe = Pipe(False)

            for (i, _part_filename) in enumerate(part_filenames[:-1]):
                process_pool.append(
                    Process(
                        target=self.multiple_process, args=[_part_filename, chunk_size * i, chunk_size * (i + 1) - 1, send_pipe, force_download]
                    )
                )
                process_pool[i].start()

            process_pool.append(
                Process(
                    target=self.multiple_process, args=[part_filenames[-1], chunk_size * (process_num - 1), file_size, send_pipe, force_download]
                )
            )
            process_pool[-1].start()

            loop_flag = 1
            stop_flag = 0
            while loop_flag:
                status_code = []
                for _process in process_pool:
                    # update progress here
                    if pid is not None:
                        if recv_pipe.poll():
                            step_size = recv_pipe.recv()
                            progress.update(pid, advance=step_size)
                            # call user callback function
                            if callback is not None:
                                return_code = callback(file_size, step_size)
                                if return_code != 0:
                                    raise KeyboardInterrupt

                    # 0: the process is alive
                    # 1: the process exits abnormally
                    # 2: the process exits normally
                    if _process.is_alive():
                        status_code.append(0)
                    elif _process.exitcode != 0:
                        status_code.append(1)
                    else:
                        status_code.append(2)

                if 1 in status_code:
                    loop_flag = 0
                    stop_flag = 1
                elif 0 not in status_code:
                    loop_flag = 0

            if stop_flag:
                for _process in process_pool:
                    if _process.is_alive():
                        _process.terminate()

                return False
            else:
                # concat files
                temp_file = f"{self.save_path}/{self.filename}.part"
                with open(temp_file, "wb") as f:
                    for _file in part_filenames:
                        with open(f"{self.save_path}/{_file}", "rb") as f2:
                            f.write(f2.read())
                        remove(f"{self.save_path}/{_file}")

                rename(temp_file, f"{self.save_path}/{self.filename}")

                return True


def solar_altitude_zenith_formula(
    date: str,
    longitude: Union[float, np.ndarray],
    latitude: Union[float, np.ndarray]
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    calculate the solar altitude and zenith angle with math formula.

    :param date: date, for example, '2020-05-29 09:13', UTC time
    :param longitude: longitudes. single value or numpy array. units: degree
    :param latitude: latitudes. single value or numpy array. units: degree
    :return: solar altitude and zenith angle. units: degree
    """
    # calculate the hour angle
    hour = datetime.strptime(date, '%Y-%m-%d %H:%M').hour
    minute = datetime.strptime(date, '%Y-%m-%d %H:%M').minute
    hour = hour + minute / 60 - 12
    hour_angle = hour * 180 / 12 + longitude
    # print(hour_angle)

    # calculate the solar declination
    day = datetime.strptime(date, '%Y-%m-%d %H:%M').strftime("%j")
    day = int(day)
    solar_declination = -23.44 * np.cos(np.deg2rad(360 / 365) * (day + 10))
    # print(solar_declination)

    # calculate the solar altitude
    res = np.cos(np.deg2rad(hour_angle)) * np.cos(np.deg2rad(solar_declination)) * np.cos(np.deg2rad(latitude)) + \
          np.sin(np.deg2rad(solar_declination)) * np.sin(np.deg2rad(latitude))

    return np.rad2deg(np.arcsin(res)), np.rad2deg(np.arccos(res))


def solar_altitude_zenith_ephem(
    date: str,
    longitude: Union[float, np.ndarray],
    latitude: Union[float, np.ndarray]
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    calculate the solar altitude and zenith angle with ``ephem`` library.

    :param date: date, for example, '2020-05-29 09:13', UTC time
    :param longitude: longitudes. single value or numpy array. units: degree
    :param latitude: latitudes. single value or numpy array. units: degree
    :return: solar altitude and zenith angle. units: degree
    """
    # transform date
    date = datetime.strptime(date, '%Y-%m-%d %H:%M').strftime("%Y/%m/%d %H:%M:%S")
    # transform from degree to rad
    longitude = np.deg2rad(longitude)
    latitude = np.deg2rad(latitude)
    # check if is an array or a single value
    if isinstance(longitude, np.ndarray) or isinstance(latitude, np.ndarray):
        shape = longitude.shape
        # transform N-D array to 1-D
        longitude = longitude.flatten()
        latitude = latitude.flatten()
        result = [list(_solar_altitude_zenith_ephem(date, lon, lat)) for lon, lat in zip(longitude, latitude)]
        result = np.asarray(result)
        # # transform 1-D array back to N-D
        return result[:, 0].reshape(shape), result[:, 1].reshape(shape)
    else:
        return _solar_altitude_zenith_ephem(date, longitude, latitude)


def _solar_altitude_zenith_ephem(date: str, longitude: float, latitude: float) -> Tuple[float, float]:
    """
    calculate the solar altitude and zenith angle with ``ephem`` library.

    :param date: date, for example, '2020-05-29 09:13', UTC time
    :param longitude: longitude. single value, units: rad
    :param latitude: latitude. single value, units: rad
    :return: solar altitude and zenith angle. units: degree
    """
    # use global object
    global sun
    global observer

    observer.lon = longitude
    observer.lat = latitude
    observer.date = date
    sun.compute(observer)

    return sun.alt / ephem.degree, 90 - sun.alt / ephem.degree


def solar_altitude_zenith(
    date: str,
    longitude: Union[float, np.ndarray],
    latitude: Union[float, np.ndarray],
    method='formula'
) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    calculate the solar altitude and zenith angle based on the date, longitude and latitude.

    :param date: UTC date, for example, '2020-05-29 09:13'
    :param longitude: longitude for a single point or an array. units: degree
    :param latitude: latitude for a single point or an array. units: degree
    :param method: method for calculating solar altitude and zenith angle, including 'formula' and 'ephem'. Defaults to 'formula'.
    :return: solar altitude and zenith angle. units: degree
    """
    # if input is ndarray, calculate multiple points
    if method == 'formula':
        return solar_altitude_zenith_formula(date, longitude, latitude)
    elif method == 'ephem':
        return solar_altitude_zenith_ephem(date, longitude, latitude)
    else:
        raise Exception(f"Unknown method: {method}")


def atmosphere_visibility(temperature: Union[float, np.ndarray], solar_zenith: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    calculate the atmosphere visibility based on the temperature and solar zenith angle.

    :param temperature: temperature, units: K
    :param solar_zenith: solar zenith, units: degree
    :return: atmosphere visibility, units: meter
    """
    # define constant variables
    beta = 0.0685
    sigma = 0.02

    return 45 * np.power(
        (1 - temperature / 100) * beta / (temperature / 100 * np.cos(np.deg2rad(solar_zenith))),
        1 / 3
    ) * np.log(1 / sigma)


def seafog_top_height(temperature: Union[float, np.ndarray], solar_zenith: Union[float, np.ndarray], beta: float = 0.0685) -> Union[float, np.ndarray]:
    """
    Calculate the seafog thickness based on the brightness temperature of the VIS band and solar zenith angle.
    This function only works at daytime.

    :param temperature: Brightness temperature of the VIS band. Units: K.
    :param solar_zenith: Solar zenith angle. Units: degree
    :param beta: Backscatter coefficient of the atmosphere.
    :return: Height of the marine fog top. Units: meter
    """

    return 45 * np.power(
        temperature / 100 * np.cos(np.deg2rad(solar_zenith)) / ((1 - temperature / 100) * beta),
        2 / 3
    )


def decompress_file(file_path: str, save_path: str = None, remove_raw: bool = True, file_format="gz"):
    """
    decompress a file.

    :param remove_raw: Whether to remove raw compressed files.
    :param file_path: Compressed file path.
    :param save_path: The save path of decompressed files, pass None to save the file in the same directory as the compressed file. Default is None
    :param file_format: File format of the compressed file.
    :return:
    """
    # get save path
    if save_path is not None and not exists(save_path):
        makedirs(save_path)
    elif save_path is None:
        save_path = abspath(dirname(file_path))

    if save_path[-1] != '/':
        save_path += '/'

    if file_format == "gz":
        # get filename
        filename = basename(file_path).replace(".gz", "")

        # decompress file
        # unpack_archive(file_path, save_path, format="")
        g_file = gzip.GzipFile(file_path)
        with open(save_path + filename, 'wb') as f:
            f.write(g_file.read())
        g_file.close()
    else:
        unpack_archive(file_path, save_path, format=file_format)

    # remove .gz file
    if remove_raw:
        remove(file_path)


def download_url(
    url: str, save_path: str, filename: str, proxy_host: str = None, proxy_port: int = None, size: int = None, headers: dict = None,
    show_progress=True, progress: Progress = None, callback: Callable = None
) -> int:
    """
    download the file from the url.

    :param url: file url
    :param save_path: save path
    :param filename: save filename.
    :param proxy_host: proxy server address
    :param proxy_port: proxy server port
    :param size: file size in kb. some websites may not return the content size, in which case you should give size to make progress show correctly.
    :param headers: Self-defined http headers.
    :param show_progress: If True, display download progress in console
    :param progress: Progress object to display download progress
    :param callback: Callable object that will be called every iteration of `res.iter_content`. `callback` should accept two params: `total_size` and `step_size`.
                     `callback` can return a non-zero and non-None value to stop download.
    :return: res status code
    """
    if show_progress:
        if isinstance(progress, Progress):
            return _download_url(
                url, save_path, filename, proxy_host=proxy_host, proxy_port=proxy_port, size=size, headers=headers,
                progress=progress, callback=callback
            )
        else:
            with PROGRESS:
                # remove finished task
                for task in PROGRESS.task_ids:
                    PROGRESS.remove_task(task)
                return _download_url(
                    url, save_path, filename, proxy_host=proxy_host, proxy_port=proxy_port, size=size, headers=headers,
                    progress=PROGRESS, callback=callback
                )
    else:
        return _download_url(url, save_path, filename, proxy_host=proxy_host, proxy_port=proxy_port, size=size, headers=headers, callback=callback)


def _download_url(
    url: str, save_path: str, filename: str, proxy_host: str = None, proxy_port: int = None, size: int = None,
    progress: Progress = None, headers: dict = None, callback: Callable = None
) -> int:
    """
    Download the file from the url.

    :param url: File url
    :param save_path: Save path
    :param filename: Save filename.
    :param proxy_host: Proxy server port
    :param proxy_port: Proxy server address
    :param size: File size in kb. Some websites may not return the content size, in which case you should give the size to make progress show correctly.
    :param progress: A Progress object to display download progress
    :param headers: Self-defined http headers.
    :param callback: Callable object that will be called every iteration of `res.iter_content`. `callback` should accept two params: `total_size` and `step_size`.
                     `callback` can return a non-zero and non-None value to stop download.
    :return: res status code
    """
    # check the save path
    if not exists(save_path):
        makedirs(save_path)

    if save_path[-1] != '/':
        save_path += '/'

    # generate proxy setting
    if proxy_host is None or proxy_port is None:
        proxy_setting = None
    else:
        proxy_setting = {
            "http": f"{proxy_host}:{proxy_port}",
            "https": f"{proxy_host}:{proxy_port}"
        }

    # check headers
    if not isinstance(headers, dict):
        headers = HEADERS

    # use debug to log url because usually user only cares if download successfully.
    logger.debug(f"Downloading file from: {url}")

    # loop 5 times
    step = 0
    res = False
    task_id = None
    while step < 4:
        try:
            res = get(url, headers=headers, allow_redirects=True, proxies=proxy_setting, stream=True)
            # check code, if it isn't 200, retry
            if res.status_code == 404:
                return 404
            elif res.status_code != 200:
                sleep(1)
                step += 1
            else:
                # set progress bar
                if size is None:
                    size = res.headers.get('content-length')
                if size is None:
                    size = 1000
                if progress is not None:
                    task_id = progress.add_task(f"[red]{filename}[red]", total=int(size))
                # save data to a temp file in case download is terminated
                temp_filename = f"{filename}.part"
                with open(save_path + temp_filename, 'wb') as f:
                    for data in res.iter_content(chunk_size=4096):
                        f.write(data)
                        if progress is not None:
                            progress.update(task_id, advance=len(data))
                        if callback is not None:
                            return_value = callback(int(size), len(data))
                            # check return code of callback
                            if return_value is not None and return_value != 0:
                                logger.info(f"Stop downloading by callback")
                                return -1
                # rename
                if exists(save_path + filename):
                    # remove the old file
                    remove(save_path + filename)
                rename(save_path + temp_filename, save_path + filename)
                break
        except (ConnectTimeout, ProxyError, SSLError) as error:
            logger.error(f"Error \"{error}\" occurred, retry...")
            sleep(1)
            step += 1
            continue
        except RequestException as error:
            logger.error(f"Unexpected error occurred: {error}, stop download")
            # other exception
            return -1

    # check if download successfully
    if isinstance(res, bool):
        # raise Exception(f"Fail to download data after retrying 5 times from {url}")
        return -1
    # if the user gives the progress, maybe we are in a loop. so remove finished progress tasks
    if progress is not None and task_id is not None:
        progress.stop_task(task_id)
        progress.remove_task(task_id)

    return res.status_code


def download_ftp(
    ftp_url: str, save_path: str, filename: str, user: str = None, passwd: str = None, proxy_host: str = None, proxy_port: int = None, progress_num=1,
    show_progress=True, progress: Progress = None, callback: Callable = None, timeout=720, connection_timeout=30, force_download=False
) -> bool:
    """
    download file from ftp server.

    :param ftp_url: file ftp url
    :param save_path: data save path
    :param filename: filename
    :param user: ftp server username
    :param passwd: ftp server password
    :param proxy_host: proxy setting.
    :param proxy_port: proxy setting.
    :param progress_num: The number of progresses that is used to download the file.
    :param show_progress: If True, display download progress in console
    :param progress: Progress object to display download progress
    :param callback: Callable object that will be called every iteration of `res.iter_content`. `callback` should accept two params: `total_size` and `step_size`.
                     `callback` can return a non-zero and non-None value to stop download.
    :param timeout: The max time which could be used to download file. Download will be terminated once the max time reached. Set it to -1 for no limitation. Unit: seconds.
    :param connection_timeout:
    :param force_download: If ignore the existing data and re-download it.
    :type force_download: bool
    :return: bool value, True if download successfully
    """
    if show_progress:
        if isinstance(progress, Progress):
            return _download_ftp(
                ftp_url, save_path, filename, user=user, passwd=passwd, proxy_host=proxy_host, proxy_port=proxy_port,
                progress=progress, callback=callback, timeout=timeout, connection_timeout=connection_timeout, progress_num=progress_num,
                force_download=force_download
            )
        else:
            with PROGRESS:
                # remove finished task
                for task in PROGRESS.task_ids:
                    PROGRESS.remove_task(task)
                return _download_ftp(
                    ftp_url, save_path, filename, user=user, passwd=passwd, proxy_host=proxy_host, proxy_port=proxy_port,
                    progress=PROGRESS, callback=callback, timeout=timeout, connection_timeout=connection_timeout, progress_num=progress_num,
                    force_download=force_download
                )
    else:
        return _download_ftp(
            ftp_url, save_path, filename, user=user, passwd=passwd, proxy_host=proxy_host, proxy_port=proxy_port, callback=callback,
            timeout=timeout, connection_timeout=connection_timeout, progress_num=progress_num,
            force_download=force_download
        )


def _download_ftp(
    ftp_url: str, save_path: str, filename: str, user: str = None, passwd: str = None, proxy_host: str = None, proxy_port: int = None, progress_num=1,
    progress: Progress = None, callback: Callable = None, timeout: int = 720, connection_timeout=30, force_download=False
) -> bool:
    """
    Download file from ftp server.

    :param ftp_url: File url
    :param save_path: Data save path
    :param filename: Filename
    :param user: FTP server username.
    :param passwd: FTP server password.
    :param proxy_host: Proxy host. Only support socks5.
    :param proxy_port: Proxy port.
    :param progress_num: The number of progresses that is used to download the file.
    :param progress: Progress object to display download progress.
    :param callback: Callable object that will be called every iteration of `res.iter_content`. `callback` should accept two params: `total_size` and `step_size`.
                     `callback` can return a non-zero and non-None value to stop download.
    :param timeout: The max time which could be used to download file. Download will be terminated once the max time reached. Set it to -1 for no limitation. Unit: seconds.
    :param connection_timeout:
    :param force_download: If ignore the existing data and re-download it.
    :type force_download: bool
    :return:  value, True if download successfully
    """
    # If username and password are given, we need to generate new url
    if user is not None and passwd is not None:
        url = ftp_url.split("ftp://")[1]
        ftp_url = f"ftp://{user}:{passwd}@{url}"

    downloader = CurlDownloader(ftp_url, save_path, filename, proxy_host, proxy_port, timeout, connection_timeout)
    return downloader.start(progress=progress, callback=callback, process_num=progress_num, force_download=force_download)


def mask_land(data: xr.DataArray, landmask: Union[str, xr.DataArray, None] = None) -> xr.DataArray:
    """
    Mask the land value in input data.

    :param data: DataArray data
    :param landmask: NetCDF landmask data file or a DataArray object.
           If None, use Python package `global-land-mask` to generate land mask.
    :return: Data with land masked
    """
    # check data, we require data contains dimensions called "latitude" and "longitude"
    if "latitude" not in data.dims or "longitude" not in data.dims:
        logger.error(f"Can't found dimension `latitude` and `longitude` in data.")
        logger.error(f"If your data contains coordinates, considering rename it to `latitude` and `longitude`")
        raise KeyError(f"Can't found dimension `latitude` and `longitude` in data.")

    if isinstance(landmask, str):
        assert exists(landmask), f"landmask file {landmask} does not exist"

        # read landmask file
        landmask = xr.open_dataset(landmask)
        landmask = landmask['landmask']

    if isinstance(landmask, xr.DataArray):
        # check the length of each dimension in both files
        assert data['latitude'].size == landmask['latitude'].size, "latitude dimension length does not match"
        assert data['longitude'].size == landmask['longitude'].size, "longitude dimension length does not match"
        landmask = landmask.to_numpy()

    else:
        # generate land mask use `global-land-mask`
        latitude = data["latitude"]
        longitude = data["longitude"]
        longitude, latitude = np.meshgrid(longitude, latitude)
        landmask = is_land(latitude, longitude)

    # mask the points in data
    np_data = data.to_numpy()
    np_data[landmask] = np.nan

    data = xr.DataArray(name=data.name, data=np_data, dims=data.dims, coords=data.coords, attrs=data.attrs)
    return data


__all__ = ['solar_altitude_zenith', 'atmosphere_visibility', 'seafog_top_height', 'solar_altitude_zenith_ephem', 'CurlDownloader',
           'solar_altitude_zenith_formula', 'decompress_file', 'download_url', 'download_ftp', 'logger', 'PROGRESS', 'mask_land']
