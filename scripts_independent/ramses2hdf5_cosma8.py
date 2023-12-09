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
Submission script for converting CSiBORG 1 RAMSES snapshots to HDF5 files on
COSMA8. Be careful as this also deletes the part_* files.
"""
from glob import glob
from os import system
from os.path import join

if __name__ == "__main__":
    # CSiBORG1 chains. Modify this is needed.
    base_dir = "/cosma8/data/dp016/dc-stis1/csiborg1/csiborg_new"
    # chains = [7444 + n * 24 for n in range(2, 101)]
    chains = [8548 + n * 24 for n in range(55)]

    for chain in chains:
        fpath = join(base_dir, f"ramses_out_{chain}_new", "output_*")
        fs = glob(fpath)

        if len(fs) != 1:
            raise ValueError(f"Found too incorrect number of output folders in `{fpath}`.")

        snapshot_path = fs[0]
        
        print(f"Doing {snapshot_path}")
        system(f"python3 ramses2hdf5.py --snapshot_path {snapshot_path}")

        # Remove all part_* particles. Recommended to do this only once you
        # have ensured that the compression was OK.
        # system(f"rm -rf {snapshot_path}/part_*")

