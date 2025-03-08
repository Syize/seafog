"""
This module contains all paths of resource files and global variables.
"""

from os.path import abspath, dirname, exists
from sys import platform

res_path = abspath(dirname(__file__))
VIS_table_file = f"{res_path}/vis_table.txt"
LANDMASK_file = f"{res_path}/landmask.nc"
HMW8_file = f"{res_path}/HMW8.txt"
MTS2_file = f"{res_path}/MTS2.txt"

# save file in /tmp on linux
if platform == "linux":
    AVHRR_CACHE_FOLDER = f"/tmp/seafog/AVHRR/"
    TIFF_CACHE = f"/tmp/seafog/cache.tiff"
else:
    AVHRR_CACHE_FOLDER = f"{res_path}/AVHRR/"
    TIFF_CACHE = f"{res_path}/cache.tiff"


def get_default_table_file(header: str) -> str:
    """
    get table file based on the header.

    :param header: valid header: "MTS2" or "HMW8".
    :return: built-in table file path.
    """
    filepath = f"{res_path}/{header}.txt"
    if exists(filepath):
        return filepath
    else:
        return ""


__all__ = ['res_path', 'VIS_table_file', 'LANDMASK_file', 'HMW8_file', 'MTS2_file', 'get_default_table_file', 'AVHRR_CACHE_FOLDER', 'TIFF_CACHE']
