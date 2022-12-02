# Copyright (C) 2022 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
Notebook utility functions.
"""

# from os.path import join

# try:
#     import csiborgtools
# except ModuleNotFoundError:
#     import sys
#     sys.path.append("../")


Nsplits = 200
dumpdir = "/mnt/extraspace/rstiskalek/csiborg/"


# Some chosen clusters
_coma = {"RA": (12 + 59/60 + 48.7 / 60**2) * 15,
         "DEC": 27 + 58 / 60 + 50 / 60**2,
         "COMDIST": 102.975}

_virgo = {"RA": (12 + 27 / 60) * 15,
          "DEC": 12 + 43/60,
          "COMDIST": 16.5}

specific_clusters = {"Coma": _coma, "Virgo": _virgo}
