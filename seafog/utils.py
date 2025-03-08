"""
This Python file contains some utility functions for the project.
"""

import gzip
import logging
import warnings
from datetime import datetime
from os import makedirs, remove, rename
from os.path import dirname, abspath, exists, basename, getsize
from time import sleep
from typing import Union, Tuple, Callable

import ephem
import numpy as np
import pycurl
import xarray as xr
from global_land_mask import is_land
from requests import get
from requests.exceptions import ConnectTimeout, ProxyError, SSLError, RequestException
from rich.logging import RichHandler
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

# close numpy RuntimeWarning
warnings.filterwarnings("ignore")

# init a logger
logger = logging.getLogger("seafog")
formatter = logging.Formatter("%(name)s :: %(message)s", datefmt="%m-%d %H:%M:%S")
# use rich handler
# handler = logging.StreamHandler()
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
PROGRESS = Progress(TextColumn("{task.description}"),
                    BarColumn(bar_width=None),
                    "[progress.percentage]{task.percentage:>3.1f}%",
                    "•", DownloadColumn(),
                    "•", TransferSpeedColumn(),
                    "•", TimeRemainingColumn())

sun = ephem.Sun()   # type: ignore
observer = ephem.Observer()

# disguise requests.get as wget
HEADERS = {
    "User-Agent": "Wget/1.12 (linux-gnu)"
}
# curl client to download ftp file
CURL = pycurl.Curl()


# call back class for ftp download function
class FTPCallback:
    def __init__(self, curl_handle: pycurl.Curl, progress: Progress = None, filename: str = "", callback: Callable = None):
        """
        A callback class that will be used to trace the progress of FTP download.

        :param curl_handle: The curl object to download file. We need to access this to stop downloading.
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


def solar_altitude_zenith_formula(date: str,
                                  longitude: Union[float, np.ndarray],
                                  latitude: Union[float, np.ndarray]) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
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


def solar_altitude_zenith_ephem(date: str,
                                longitude: Union[float, np.ndarray],
                                latitude: Union[float, np.ndarray]) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
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


def solar_altitude_zenith(date: str,
                          longitude: Union[float, np.ndarray],
                          latitude: Union[float, np.ndarray],
                          method='formula') -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
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


def seafog_thickness(temperature: Union[float, np.ndarray], solar_zenith: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    calculate the seafog thickness based on the temperature and solar zenith angle.

    :param temperature: temperature, units: K
    :param solar_zenith: solar zenith angle, units: degree
    :return: seafog thickness, units: meter
    """
    # define constant variables
    beta = 0.0685

    return 45 * np.power(
        temperature / 100 * np.cos(np.deg2rad(solar_zenith)) / ((1 - temperature / 100) * beta),
        2 / 3
    )


def decompress_file(file_path: str, save_path: str = None, remove_raw: bool = True):
    """
    Decompress a file.

    :param file_path: Compressed file path.
    :param save_path: Decompressed file save path, pass None to save the file in the same directory as the compressed file. Default is None
    :param remove_raw: Whether to remove the raw ".gz" file.
    :return:
    """
    # get the save path
    if save_path is not None and not exists(save_path):
        makedirs(save_path)
    elif save_path is None:
        save_path = abspath(dirname(file_path))

    if save_path[-1] != '/':
        save_path += '/'

    # get filename
    filename = basename(file_path).replace(".gz", "")

    # decompress file
    g_file = gzip.GzipFile(file_path)
    with open(save_path + filename, 'wb') as f:
        f.write(g_file.read())
    g_file.close()

    # remove .gz file
    if remove_raw:
        remove(file_path)


def download_url(url: str, save_path: str, filename: str, proxy_host: str = None, proxy_port: int = None, size: int = None, headers: dict = None,
                 show_progress=True, progress: Progress = None, callback: Callable = None) -> int:
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
            return _download_url(url, save_path, filename, proxy_host=proxy_host, proxy_port=proxy_port, size=size, headers=headers,
                                 progress=progress, callback=callback)
        else:
            with PROGRESS:
                # remove finished task
                for task in PROGRESS.task_ids:
                    PROGRESS.remove_task(task)
                return _download_url(url, save_path, filename, proxy_host=proxy_host, proxy_port=proxy_port, size=size, headers=headers,
                                     progress=PROGRESS, callback=callback)
    else:
        return _download_url(url, save_path, filename, proxy_host=proxy_host, proxy_port=proxy_port, size=size, headers=headers, callback=callback)


def _download_url(url: str, save_path: str, filename: str, proxy_host: str = None, proxy_port: int = None, size: int = None,
                  progress: Progress = None,
                  headers: dict = None, callback: Callable = None) -> int:
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


def download_ftp(ftp_url: str, save_path: str, filename: str, user: str = None, passwd: str = None, proxy_host: str = None, proxy_port: int = None,
                 show_progress=True, progress: Progress = None, callback: Callable = None, timeout=720) -> bool:
    """
    download file from ftp server.

    :param ftp_url: file ftp url
    :param save_path: data save path
    :param filename: filename
    :param user: ftp server username
    :param passwd: ftp server password
    :param proxy_host: proxy setting.
    :param proxy_port: proxy setting.
    :param show_progress: If True, display download progress in console
    :param progress: Progress object to display download progress
    :param callback: Callable object that will be called every iteration of `res.iter_content`. `callback` should accept two params: `total_size` and `step_size`.
                     `callback` can return a non-zero and non-None value to stop download.
    :param timeout: The max time which could be used to download file. Download will be terminated once the max time reached. Set it to -1 for no limitation. Unit: seconds.
    :return: bool value, True if download successfully
    """
    if show_progress:
        if isinstance(progress, Progress):
            return _download_ftp(ftp_url, save_path, filename, user=user, passwd=passwd, proxy_host=proxy_host, proxy_port=proxy_port,
                                 progress=progress, callback=callback, timeout=timeout)
        else:
            with PROGRESS:
                # remove finished task
                for task in PROGRESS.task_ids:
                    PROGRESS.remove_task(task)
                return _download_ftp(ftp_url, save_path, filename, user=user, passwd=passwd, proxy_host=proxy_host, proxy_port=proxy_port,
                                     progress=PROGRESS, callback=callback, timeout=timeout)
    else:
        return _download_ftp(ftp_url, save_path, filename, user=user, passwd=passwd, proxy_host=proxy_host, proxy_port=proxy_port, callback=callback, timeout=timeout)


def _download_ftp(ftp_url: str, save_path: str, filename: str, user: str = None, passwd: str = None, proxy_host: str = None, proxy_port: int = None,
                  progress: Progress = None, callback: Callable = None, timeout: int = 720) -> bool:
    """
    Download file from ftp server.

    :param ftp_url: File url
    :param save_path: Data save path
    :param filename: Filename
    :param user: FTP server username.
    :param passwd: FTP server password.
    :param proxy_host: Proxy host. Only support socks5.
    :param proxy_port: Proxy port.
    :param progress: Progress object to display download progress.
    :param callback: Callable object that will be called every iteration of `res.iter_content`. `callback` should accept two params: `total_size` and `step_size`.
                     `callback` can return a non-zero and non-None value to stop download.
    :param timeout: The max time which could be used to download file. Download will be terminated once the max time reached. Set it to -1 for no limitation. Unit: seconds.
    :return: bool value, True if download successfully
    """
    # Reset CURL
    CURL.reset()
    # Set connection timeout
    CURL.setopt(pycurl.CONNECTTIMEOUT, 30)
    # Sometimes it may take a very long time to download a file, so we set max time
    if timeout >= 0:
        CURL.setopt(pycurl.TIMEOUT, timeout)
    # Set Debug options
    # CURL.setopt(pycurl.VERBOSE, True)
    # If username and password are given, we need to generate new url
    if user is not None and passwd is not None:
        url = ftp_url.split("ftp://")[1]
        ftp_url = f"ftp://{user}:{passwd}@{url}"

    # If a proxy is set, use it
    if proxy_host is not None and proxy_port is not None:
        # check protocol
        # if not (proxy_host.startswith("https") or proxy_host.startswith("http") or proxy_host.startswith("socks")):
        if not proxy_host.startswith("socks5"):
            # logger.error(f"Unsupported proxy protocol: {proxy_host.split('://')[0]}")
            logger.error(f"Only support \"socks5\" protocol")
            return False
        else:
            CURL.setopt(pycurl.PROXY, f"{proxy_host}")
            CURL.setopt(pycurl.PROXYPORT, proxy_port)
            CURL.setopt(pycurl.PROXYTYPE, pycurl.PROXYTYPE_SOCKS5)

    logger.debug(f"Downloading file from {ftp_url}")

    # set callback object
    ftp_callback = FTPCallback(curl_handle=CURL, progress=progress, filename=filename, callback=callback)
    # save data to a temp file in case download is terminated
    temp_filename = f"{filename}.part"

    # start download
    CURL.setopt(pycurl.URL, ftp_url)
    # display progress
    CURL.setopt(pycurl.NOPROGRESS, False)
    # set callback
    CURL.setopt(pycurl.XFERINFOFUNCTION, ftp_callback)

    # check if temp file exists, and we can continue to download
    if exists(f"{save_path}/{temp_filename}"):
        # set write type
        open_type = "ab"
        # get file size
        file_size = getsize(f"{save_path}/{temp_filename}")
        # set curl
        CURL.setopt(pycurl.RESUME_FROM, file_size)
        # set progress
        ftp_callback.set_start_size(file_size)
    else:
        open_type = "wb"

    with open(f"{save_path}/{temp_filename}", open_type) as f:
        CURL.setopt(pycurl.WRITEDATA, f)

        try:
            CURL.perform()
        except pycurl.error as e:
            logger.error(f"Failed to download data: {ftp_url}")
            logger.error(f"Error occurred: {e}")
            return False

    if exists(f"{save_path}/{filename}"):
        remove(f"{save_path}/{filename}")
    # Check if download successfully
    if not exists(f"{save_path}/{temp_filename}"):
        return False

    rename(f"{save_path}/{temp_filename}", f"{save_path}/{filename}")

    return True


def mask_land(data: xr.DataArray, landmask: Union[str, xr.DataArray, None] = None) -> xr.DataArray:
    """
    Read landmask file and mask the corresponding point in input data.

    :param data: DataArray data
    :param landmask: Netcdf landmask data file or a DataArray object.
           If None, use Python package `global-land-mask` to generate land mask.
    :return: Data with land masked
    """
    # check data, we require data contains dimensions called "latitude" and "longitude"
    if "latitude" not in data.dims or "longitude" not in data.dims:
        logger.error(f"Can't found dimension `latitude` and `longitude` in data.")
        logger.error(f"If your data contains coordinates, considering rename it to `latitude` and `longitude`")
        raise KeyError

    # check file
    if isinstance(landmask, str):
        assert exists(landmask), f"landmask file {landmask} does not exist"

        # read landmask file
        landmask = xr.open_dataset(landmask)
        landmask = landmask['landmask']

    if isinstance(landmask, xr.DataArray):
        # check the length of each dimension in both file
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
    np_data[landmask == 1] = np.nan

    data = xr.DataArray(name=data.name, data=np_data, dims=data.dims, coords=data.coords, attrs=data.attrs)
    return data


__all__ = ['solar_altitude_zenith', 'atmosphere_visibility', 'seafog_thickness', 'solar_altitude_zenith_ephem',
           'solar_altitude_zenith_formula', 'decompress_file', 'download_url', 'download_ftp', 'logger', 'PROGRESS', 'mask_land']
