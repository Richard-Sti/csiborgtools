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
Script to calculate the HMF for CSIBORG and Quijote haloes.
"""
from argparse import ArgumentParser
from datetime import datetime
from distutils.util import strtobool

import numpy
from mpi4py import MPI

from taskmaster import work_delegation
from utils import get_nsims

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools


def get_counts(nsim, bins, paths, parser_args):
    """
    Calculate and save the number of haloes in each mass bin.

    Parameters
    ----------
    nsim : int
        Simulation index.
    bins : 1-dimensional array
        Array of bin edges (in log10 mass).
    paths : csiborgtools.read.Paths
        Paths object.
    parser_args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    None
    """
    simname = parser_args.simname
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    bounds = {"dist": (0, parser_args.Rmax)}

    if simname == "csiborg":
        cat = csiborgtools.read.HaloCatalogue(nsim, paths, bounds=bounds)
        logmass = numpy.log10(cat["totpartmass"])
        counts = csiborgtools.fits.number_counts(logmass, bins)
    elif simname == "quijote":
        cat0 = csiborgtools.read.QuijoteHaloCatalogue(nsim, paths, nsnap=4)
        nmax = int(cat0.box.boxsize // (2 * parser_args.Rmax))**3
        counts = numpy.full((nmax, len(bins) - 1), numpy.nan,
                            dtype=numpy.float32)

        for nobs in range(nmax):
            cat = cat0.pick_fiducial_observer(nobs, rmax=parser_args.Rmax)
            logmass = numpy.log10(cat["group_mass"])
            counts[nobs, :] = csiborgtools.fits.number_counts(logmass, bins)

    fout = paths.halo_counts(simname, nsim)
    if parser_args.verbose:
        print(f"{datetime.now()}: saving halo counts to `{fout}`.")
    numpy.savez(fout, counts=counts, bins=bins, rmax=parser_args.Rmax)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, choices=["csiborg", "quijote"],
                        help="Simulation name")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="Indices of simulations to cross. If `-1` processes all simulations.")  # noqa
    parser.add_argument("--Rmax", type=float, default=155/0.705,
                        help="High-resolution region radius")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        default=False)

    parser_args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()
    verbose = nproc == 1
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(parser_args, paths)
    bins = numpy.arange(11., 16., 0.2, dtype=numpy.float32)

    def do_work(nsim):
        get_counts(nsim, bins, paths, parser_args)

    work_delegation(do_work, nsims, comm, master_verbose=parser_args.verbose)