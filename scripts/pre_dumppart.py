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
Script to load in the simulation particles, sort them by their FoF halo ID and
dump into a HDF5 file. Stores the first and last index of each halo in the
particle array. This can be used for fast slicing of the array to acces
particles of a single clump.
"""
from argparse import ArgumentParser
from datetime import datetime
from gc import collect

import h5py
import numba
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


@numba.jit(nopython=True)
def minmax_halo(hid, halo_ids, start_loop=0):
    """
    Find the start and end index of a halo in a sorted array of halo IDs.
    This is much faster than using `numpy.where` and then `numpy.min` and
    `numpy.max`.
    """
    start = None
    end = None

    for i in range(start_loop, halo_ids.size):
        n = halo_ids[i]
        if n == hid:
            if start is None:
                start = i
            end = i
        elif n > hid:
            break
    return start, end


def main(nsim, simname, verbose):
    """
    Read in the snapshot particles, sort them by their FoF halo ID and dump
    into a HDF5 file. Stores the first and last index of each halo in the
    particle array for fast slicing of the array to acces particles of a single
    halo.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    simname : str
        Simulation name.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    None
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    partreader = csiborgtools.read.ParticleReader(paths)

    if simname == "quijote":
        raise NotImplementedError("Not implemented for Quijote yet.")

    # Keep "ID" as the last column!
    pars_extract = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'M', "ID"]
    nsnap = max(paths.get_snapshots(nsim))
    fname = paths.particles(nsim)
    # We first read in the halo IDs of the particles and infer the sorting.
    # Right away we dump the halo IDs to a HDF5 file and clear up memory.
    if verbose:
        print(f"{datetime.now()}: loading particles {nsim}.", flush=True)
    part_hids = partreader.read_fof_hids(nsim)
    sort_indxs = numpy.argsort(part_hids).astype(numpy.int32)
    part_hids = part_hids[sort_indxs]
    with h5py.File(fname, "w") as f:
        f.create_dataset("halo_ids", data=part_hids)
        f.close()
    del part_hids
    collect()

    # Next we read in the particles and sort them by their halo ID.
    # We cannot directly read this as an unstructured array because the float32
    # precision is insufficient to capture the halo IDs.
    parts, pids = partreader.read_particle(
        nsnap, nsim, pars_extract, return_structured=False, verbose=verbose)
    # Now we in two steps save the particles and particle IDs.
    if verbose:
        print(f"{datetime.now()}: dumping particles from {nsim}.", flush=True)
    parts = parts[sort_indxs]
    pids = pids[sort_indxs]
    del sort_indxs
    collect()

    with h5py.File(fname, "r+") as f:
        f.create_dataset("particle_ids", data=pids)
        f.close()
    del pids
    collect()

    with h5py.File(fname, "r+") as f:
        f.create_dataset("particles", data=parts)
        f.close()
    del parts
    collect()

    if verbose:
        print(f"{datetime.now()}: creating halo map for {nsim}.", flush=True)
    # Load clump IDs back to memory
    with h5py.File(fname, "r") as f:
        part_hids = f["halo_ids"][:]
    # We loop over the unique clump IDs.
    unique_halo_ids = numpy.unique(part_hids)
    halo_map = numpy.full((unique_halo_ids.size, 3), numpy.nan,
                          dtype=numpy.int32)
    start_loop = 0
    niters = unique_halo_ids.size
    for i in trange(niters) if verbose else range(niters):
        hid = unique_halo_ids[i]
        k0, kf = minmax_halo(hid, part_hids, start_loop=start_loop)
        halo_map[i, 0] = hid
        halo_map[i, 1] = k0
        halo_map[i, 2] = kf
        start_loop = kf

    # We save the mapping to a HDF5 file
    with h5py.File(paths.particles(nsim), "r+") as f:
        f.create_dataset("halomap", data=halo_map)
        f.close()

    del part_hids
    collect()


if __name__ == "__main__":
    # And next parse all the arguments and set up CSiBORG objects
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, default="csiborg",
                        choices=["csiborg", "quijote"],
                        help="Simulation name")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. If `-1` processes all .")
    args = parser.parse_args()

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(args, paths)

    def _main(nsim):
        main(nsim, args.simname, verbose=MPI.COMM_WORLD.comm.Get_size() == 1)

    work_delegation(_main, nsims, MPI.COMM_WORLD)
