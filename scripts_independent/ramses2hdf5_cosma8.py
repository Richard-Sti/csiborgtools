# Copyright (C) 2023 Richard Stiskalek
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
Script to convert a RAMSES snapshot to a compressed HDF5 file. Be careful
because reading the HDF5 file requires `hdf5plugin` package to be installed.
"""
from argparse import ArgumentParser
from datetime import datetime
from os.path import join, isdir
from os import listdir
import numpy as np

import hdf5plugin
import numpy
import pynbody
from h5py import File
import os
from glob import glob

from os import system

if __name__ == "__main__":
    base_dir = "/cosma8/data/dp016/dc-stis1/csiborg1/"

    chains = [7444 + n * 24 for n in range(101)][:1]


    for chain in chains:
        fpath = join(base_dir, f"ramses_out_{chain}", "output_*")
        fs = glob(fpath)

        if len(fs) != 1:
            raise ValueError(f"Found too many output folders in `{fpath}`.")

        snapshot_path = fs[0]

        system(f"python3 ramses2hdf5.py --snapshot_path {snapshot_path}")

        # Remove all part_* particles
        # print(f"rm -rf {snapshot_path}/part_*")

