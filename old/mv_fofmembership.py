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
Short script to move and change format of the CSiBORG FoF membership files
calculated by Julien. Additionally, also orders the particles in the same way
as the PHEW halo finder output.
"""
from argparse import ArgumentParser
from datetime import datetime
from gc import collect
from os.path import join
from shutil import copy

import numpy
from mpi4py import MPI
from taskmaster import work_delegation
from tqdm import trange

from utils import get_nsims

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


def copy_membership(nsim, verbose=True):
    """
    Copy the FoF particle halo membership to the CSiBORG directory and write it
    as a NumPy array instead of a text file.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    fpath = join("/mnt/extraspace/jeg/greenwhale/Constrained_Sims",
                 f"sim_{nsim}/particle_membership_{nsim}_FOF.txt")
    if verbose:
        print(f"Loading from ... `{fpath}`.")
    data = numpy.genfromtxt(fpath, dtype=int)

    fout = paths.fof_membership(nsim, "csiborg")
    if verbose:
        print(f"Saving to ... `{fout}`.")
    numpy.save(fout, data)


def copy_catalogue(nsim, verbose=True):
    """
    Move the FoF catalogue to the CSiBORG directory.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    verbose : bool, optional
        Verbosity flag.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    source = join("/mnt/extraspace/jeg/greenwhale/Constrained_Sims",
                  f"sim_{nsim}/halo_catalog_{nsim}_FOF.txt")
    dest = paths.fof_cat(nsim, "csiborg")
    if verbose:
        print("Copying`{}` to `{}`.".format(source, dest))
    copy(source, dest)


def sort_fofid(nsim, verbose=True):
    """
    Read the FoF particle halo membership and sort the halo IDs to the ordering
    of particles in the PHEW clump IDs.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    verbose : bool, optional
        Verbosity flag.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, "csiborg"))
    fpath = paths.fof_membership(nsim, "csiborg")
    if verbose:
        print(f"{datetime.now()}: loading from ... `{fpath}`.")
    # Columns are halo ID, particle ID.
    fof = numpy.load(fpath)

    reader = csiborgtools.read.CSiBORGReader(paths)
    pars_extract = ["x"]  # Dummy variable
    __, pids = reader.read_snapshot(nsnap, nsim, pars_extract,
                                    return_structured=False, verbose=verbose)
    del __
    collect()

    # Map the particle IDs in pids to their corresponding PHEW array index
    if verbose:
        print(f"{datetime.now()}: mapping particle IDs to their indices.")
    pids_idx = {pid: i for i, pid in enumerate(pids)}

    if verbose:
        print(f"{datetime.now()}: mapping FoF HIDs to their array indices.")
    # Unassigned particle IDs are assigned a halo ID of 0. Same as PHEW.
    fof_hids = numpy.zeros(pids.size, dtype=numpy.int32)
    for i in trange(fof.shape[0]) if verbose else range(fof.shape[0]):
        hid, pid = fof[i]
        fof_hids[pids_idx[pid]] = hid

    fout = paths.fof_membership(nsim, "csiborg", sorted=True)
    if verbose:
        print(f"Saving the sorted data to ... `{fout}`")
    numpy.save(fout, fof_hids)


def main(nsim, verbose=True):
    copy_membership(nsim, verbose=verbose)
    copy_catalogue(nsim, verbose=verbose)
    sort_fofid(nsim, verbose=verbose)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, default="csiborg",
                        choices=["csiborg", "quijote"],
                        help="Simulation name")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="Indices of simulations to cross. If `-1` processes all simulations.")  # noqa
    args = parser.parse_args()

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(args, paths)
    work_delegation(main, nsims, MPI.COMM_WORLD)
