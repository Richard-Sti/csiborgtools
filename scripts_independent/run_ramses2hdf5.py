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
Submission script for converting CSiBORG 1 RAMSES snapshots to HDF5 files.
"""
from glob import glob
from os import system, makedirs
from os.path import join, basename, exists


def main_cosma8():
    # CSiBORG1 chains. Modify this is needed.
    base_dir = "/cosma8/data/dp016/dc-stis1/csiborg1/csiborg_new"
    # chains = [7444 + n * 24 for n in range(2, 101)]
    chains = [8548 + n * 24 for n in range(55)]

    for chain in chains:
        fpath = join(base_dir, f"ramses_out_{chain}_new", "output_*")
        fs = glob(fpath)

        if len(fs) != 1:
            raise ValueError(f"Found a wrong number of folders in `{fpath}`.")

        snapshot_path = fs[0]

        print(f"Doing {snapshot_path}")
        system(f"python3 ramses2hdf5.py --snapshot_path {snapshot_path}")

        # Remove all part_* particles. Recommended to do this only once you
        # have ensured that the compression was OK.
        # system(f"rm -rf {snapshot_path}/part_*")


def convert_last_snapshot():
    """
    Read in RAMSES binary files and output them as HDF5 files. Works on
    glamdring.
    """
    chains = [7444 + n * 24 for n in range(101)]
    base_dir = "/mnt/extraspace/hdesmond/"

    for chain in chains[:1]:
        sourcedir = join(base_dir, f"ramses_out_{chain}")
        snap = max([int(basename(f).replace("output_", ""))
                    for f in glob(join(sourcedir, "output_*"))])

        snapshot_path = join(sourcedir, f"output_{str(snap).zfill(5)}")

        output_dir = f"/mnt/extraspace/rstiskalek/csiborg1/chain_{chain}"

        if not exists(output_dir):
            makedirs(output_dir)

        output_path = join(output_dir, f"snapshot_{str(snap).zfill(5)}.hdf5")

        cmd = f"addqueue -q berg -n 1x1 -m 64 /mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python ramses2hdf5.py --snapshot_path {snapshot_path} --output_path {output_path}"  # noqa
        print(cmd)
        system(cmd)


def convert_initial_snapshot():

    chains = [7444 + n * 24 for n in range(101)]
    chains = [7444, 7468]
    base_dir = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/output"

    for chain in chains:
        snapshot_path = join(base_dir,
                             f"ramses_out_{chain}_new", "output_00001")
        cmd = f"addqueue -q berg -n 1x1 -m 64 /mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python ramses2hdf5.py --snapshot_path {snapshot_path}"  # noqa
        system(cmd)


if __name__ == "__main__":
    convert_initial_snapshot()
