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

import numpy
import yaml
from mpi4py import MPI

from taskmaster import work_delegation
from utils import open_catalogues

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools


def find_neighbour(args, nsim, cats, paths, comm):
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

    Returns
    -------
    None
    """
    ndist, cross_hindxs = csiborgtools.match.find_neighbour(nsim, cats)

    mass_key = "totpartmass" if args.simname == "csiborg" else "group_mass"
    cat0 = cats[nsim]
    mass = cat0[mass_key]
    rdist = cat0.radial_distance(in_initial=False)

    fout = paths.cross_nearest(args.simname, args.run, nsim)
    if args.verbose:
        print(f"Rank {comm.Get_rank()} writing to `{fout}`.", flush=True)
    numpy.savez(fout, ndist=ndist, cross_hindxs=cross_hindxs, mass=mass,
                rdist=rdist)


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

    comm = MPI.COMM_WORLD
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    cats = open_catalogues(args, config, paths, comm)

    def do_work(nsim):
        return find_neighbour(args, nsim, cats, paths, comm)

    work_delegation(do_work, list(cats.keys()), comm,
                    master_verbose=args.verbose)

    comm.Barrier()
    if comm.Get_rank() == 0:
        print(f"{datetime.now()}: all finished. Quitting.")
