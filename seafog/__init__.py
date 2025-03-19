"""
Derived from toolkit ``Plot4Verify_Seafog``, ``seafog`` is a Python package to help download essential data, detect seafog top height, seafog area, and plot images about seafog.
"""

from avhrr import *
from detect import *
from gfs import *
from goos_sst import *
from himawari import *
from pgm import *
from res import LANDMASK_file
from utils import *
