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
Script to sort the HaloMaker's `particle_membership` file to match the ordering
of particles in the simulation snapshot.
"""
from argparse import ArgumentParser
from datetime import datetime
from glob import iglob

import h5py
import numpy
import pynbody
from mpi4py import MPI
from taskmaster import work_delegation
from tqdm import trange

import csiborgtools


def sort_particle_membership(nsim, nsnap, method):
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
    print(f"{datetime.now()}:   starting simulation {nsim}, snapshot {nsnap} and method {method}.")  # noqa
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    fpath = next(iglob(f"/mnt/extraspace/rstiskalek/CSiBORG/halo_maker/ramses_{nsim}/output_{str(nsnap).zfill(5)}/**/*particle_membership*", recursive=True), None)  # noqa
    print(f"{datetime.now()}:   loading particle membership `{fpath}`.")
    # Columns are halo ID, particle ID
    membership = numpy.genfromtxt(fpath, dtype=int)

    print(f"{datetime.now()}:   loading particle IDs from the snapshot.")
    sim = pynbody.load(paths.snapshot(nsnap, nsim, "csiborg"))
    pids = numpy.asanyarray(sim["iord"])

    print(f"{datetime.now()}:   mapping particle IDs to their indices.")
    pids_idx = {pid: i for i, pid in enumerate(pids)}

    print(f"{datetime.now()}:   mapping HIDs to their array indices.")
    # Unassigned particle IDs are assigned a halo ID of 0.
    hids = numpy.zeros(pids.size, dtype=numpy.int32)
    for i in trange(membership.shape[0]):
        hid, pid = membership[i]
        hids[pids_idx[pid]] = hid

    fout = fpath.replace("_membership", "_membership_sorted.h5")
    print(f"{datetime.now()}:   saving the sorted data to ... `{fout}`")

    header = """
    This dataset represents halo indices for each particle.
        - The particles are ordered as they appear in the simulation snapshot.
        - Unassigned particles are given a halo index of 0.
        """
    with h5py.File(fout, 'w') as hdf:
        dset = hdf.create_dataset('hids_dataset', data=hids)
        dset.attrs['header'] = header


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", type=str, required=True,
                        help="HaloMaker method")
    parser.add_argument("--nsim", type=int, required=False, default=None,
                        help="IC index. If not set process all.")
    args = parser.parse_args()
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    if args.nsim is None:
        ics = paths.get_ics("csiborg")
    else:
        ics = [args.nsim]

    snaps = numpy.array([max(paths.get_snapshots(nsim, "csiborg"))
                         for nsim in ics])

    def main(n):
        sort_particle_membership(ics[n], snaps[n], args.method)

    work_delegation(main, list(range(len(ics))), MPI.COMM_WORLD)
