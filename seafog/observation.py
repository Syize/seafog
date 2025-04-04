"""
seafog.observation provides methods to download observation datas from various sources.
"""

from datetime import datetime
from os import makedirs, listdir
from os.path import exists
from shutil import rmtree
from typing import Callable, Tuple

from rich.progress import Progress

from .utils import decompress_file, download_url, logger

# prepbufr observation data root url, the {} for data's year
PREP_BUFR_ROOT_URL = "https://data.rda.ucar.edu/d337000/tarfiles/{}"
# the {} for data's date, in format "%Y%m%d"
PREP_BUFR_NAME_TEMPLATE = "prepbufr.{}.nr.tar.gz"


def prepbufr_find_data(date: str, save_path: str,  proxy_host: str = None, proxy_port: int = None,
                       progress: Progress = None, headers: dict = None, show_progress=True, callback: Callable = None) -> Tuple[str, str, str, str]:
    """
    Download observation data ``NCEP ADP Global Upper Air and Surface Weather Observations`` from `Research Data Archive <https://rda.ucar.edu/datasets/d337000/#>`_.

    :param date:
    :type date:
    :param save_path:
    :type save_path:
    :param proxy_host:
    :type proxy_host:
    :param proxy_port:
    :type proxy_port:
    :param progress:
    :type progress:
    :param headers:
    :type headers:
    :param show_progress:
    :type show_progress:
    :param callback:
    :type callback:
    :return:
    :rtype:
    """
    if not exists(save_path):
        makedirs(save_path)

    # we store the data in save_path/prepbufr.{} separately.
    data_date = datetime.strptime(date, "%Y-%m-%d %H:%M")
    subdir_name = f"prepbufr.{data_date.strftime('%Y%m%d')}"

    if exists(f"{save_path}/{subdir_name}"):
        file_list = [x for x in listdir(f"{save_path}/{subdir_name}") if x.endswith(".nr")]
        if len(file_list) == 4:
            return tuple(file_list)     # type: ignore
        else:
            rmtree(f"{save_path}/{subdir_name}")

    data_name = PREP_BUFR_NAME_TEMPLATE.format(data_date.strftime('%Y%m%d'))
    url = f"{PREP_BUFR_ROOT_URL.format(data_date.year)}/{data_name}"

    code = download_url(url, save_path, data_name, proxy_host=proxy_host, proxy_port=proxy_port,
                        headers=headers, show_progress=show_progress, progress=progress, callback=callback)

    if code == 404:
        logger.error(f"The file {data_name} doesn't exist in the server (status 404)")
        raise FileNotFoundError(f"The file {data_name} doesn't exist in the server (status 404)")

    elif code != 200:
        logger.error(f"Failed to download file {data_name}, status code is {code}. May be you can try again later, or check if this url is right: {url}")
        raise ConnectionError

    decompress_file(f"{save_path}/{data_name}", f"{save_path}/{subdir_name}", file_format="gztar")

    file_list = listdir(f"{save_path}/{subdir_name}")

    if len(file_list) != 4:
        logger.error(f"Expected 4 files in {save_path}/{subdir_name}, but got {len(file_list)}")
        raise Exception

    return tuple(file_list)     # type: ignore


__all__ = ["prepbufr_find_data"]
