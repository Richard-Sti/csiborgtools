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
Script to find the nearest neighbour of each halo in a given halo catalogue
from the remaining catalogues in the suite (CSIBORG or Quijote). The script is
MPI parallelized over the reference simulations.
"""
from argparse import ArgumentParser
from datetime import datetime
from distutils.util import strtobool
from os import remove

import numpy
import yaml
from mpi4py import MPI
from taskmaster import work_delegation
from tqdm import trange

from utils import open_catalogues

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools


def find_neighbour(args, nsim, cats, paths, comm, save_kind):
    """
    Find the nearest neighbour of each halo in the given catalogue.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    nsim : int
        Simulation index.
    cats : dict
        Dictionary of halo catalogues. Keys are simulation indices, values are
        the catalogues.
    paths : csiborgtools.paths.Paths
        Paths object.
    comm : mpi4py.MPI.Comm
        MPI communicator.
    save_kind : str
        Kind of data to save. Must be either `dist` or `bin_dist`.

    Returns
    -------
    None
    """
    assert save_kind in ["dist", "bin_dist"]
    ndist, cross_hindxs = csiborgtools.match.find_neighbour(nsim, cats)
    mass_key = "totpartmass" if args.simname == "csiborg" else "group_mass"
    cat0 = cats[nsim]
    rdist = cat0.radial_distance(in_initial=False)

    # Distance is saved optionally, whereas binned distance is always saved.
    if save_kind == "dist":
        out = {"ndist": ndist,
               "cross_hindxs": cross_hindxs,
               "mass": cat0[mass_key],
               "ref_hindxs": cat0["index"],
               "rdist": rdist}
        fout = paths.cross_nearest(args.simname, args.run, "dist", nsim)
        if args.verbose:
            print(f"Rank {comm.Get_rank()} writing to `{fout}`.", flush=True)
        numpy.savez(fout, **out)

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    reader = csiborgtools.read.NearestNeighbourReader(
        paths=paths, **csiborgtools.neighbour_kwargs)
    counts = numpy.zeros((reader.nbins_radial, reader.nbins_neighbour),
                         dtype=numpy.float32)
    counts = reader.count_neighbour(counts, ndist, rdist)
    out = {"counts": counts}
    fout = paths.cross_nearest(args.simname, args.run, "bin_dist", nsim)
    if args.verbose:
        print(f"Rank {comm.Get_rank()} writing to `{fout}`.", flush=True)
    numpy.savez(fout, **out)


def collect_dist(args, paths):
    """
    Collect the binned nearest neighbour distances into a single file.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    paths : csiborgtools.paths.Paths
        Paths object.

    Returns
    -------
    """
    fnames = paths.cross_nearest(args.simname, args.run, "bin_dist")

    if args.verbose:
        print("Collecting counts into a single file.", flush=True)

    for i in trange(len(fnames)) if args.verbose else range(len(fnames)):
        fname = fnames[i]
        data = numpy.load(fname)
        if i == 0:
            out = data["counts"]
        else:
            out += data["counts"]

        remove(fname)

    fout = paths.cross_nearest(args.simname, args.run, "tot_counts",
                               nsim=0, nobs=0)
    if args.verbose:
        print(f"Writing the summed counts to `{fout}`.", flush=True)
    numpy.savez(fout, tot_counts=out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=str, help="Run name")
    parser.add_argument("--simname", type=str, choices=["csiborg", "quijote"],
                        help="Simulation name")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="Indices of simulations to cross. If `-1` processes all simulations.")  # noqa
    parser.add_argument("--Rmax", type=float, default=155/0.705,
                        help="High-resolution region radius")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        default=False)
    args = parser.parse_args()
    with open("./match_finsnap.yml", "r") as file:
        config = yaml.safe_load(file)

    if args.simname == "csiborg":
        save_kind = "dist"
    else:
        save_kind = "bin_dist"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    cats = open_catalogues(args, config, paths, comm)

    def do_work(nsim):
        return find_neighbour(args, nsim, cats, paths, comm, save_kind)

    work_delegation(do_work, list(cats.keys()), comm,
                    master_verbose=args.verbose)

    comm.Barrier()
    if rank == 0:
        collect_dist(args, paths)
        print(f"{datetime.now()}: all finished. Quitting.")
