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
"""A script to match all IC pairs of a simulation."""
from argparse import ArgumentParser
from distutils.util import strtobool
from itertools import combinations
from random import Random

from mpi4py import MPI

import csiborgtools
from match_singlematch import pair_match, pair_match_max
from taskmaster import work_delegation


def get_combs(simname):
    """
    Get the list of all pairs of IC indices and permute them with a fixed
    seed.

    Parameters
    ----------
    simname : str
        Simulation name.

    Returns
    -------
    combs : list
        List of pairs of simulations.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    combs = list(combinations(paths.get_ics(simname), 2))

    Random(42).shuffle(combs)
    return combs


def main(comb, kind, simname, min_logmass, sigma, mult, verbose):
    """
    Match a pair of simulations.

    Parameters
    ----------
    comb : tuple
        Pair of simulation IC indices.
    kind : str
        Kind of matching.
    simname : str
        Simulation name.
    min_logmass : float
        Minimum log halo mass.
    sigma : float
        Smoothing scale in number of grid cells.
    mult : float
        Multiplicative factor for search radius.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    None
    """
    nsim0, nsimx = comb
    if kind == "overlap":
        pair_match(nsim0, nsimx, simname, min_logmass, sigma, verbose)
    elif args.kind == "max":
        pair_match_max(nsim0, nsimx, simname, min_logmass, mult, verbose)
    else:
        raise ValueError(f"Unknown matching kind: `{kind}`.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--kind", type=str, required=True,
                        choices=["overlap", "max"], help="Kind of matching.")
    parser.add_argument("--simname", type=str, required=True,
                        help="Simulation name.",
                        choices=["csiborg", "quijote"])
    parser.add_argument("--min_logmass", type=float, required=True,
                        help="Minimum log halo mass.")
    parser.add_argument("--sigma", type=float, default=0,
                        help="Smoothing scale in number of grid cells.")
    parser.add_argument("--mult", type=float, default=5,
                        help="Search radius multiplier for Max's method.")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        default=False, help="Verbosity flag.")
    args = parser.parse_args()

    combs = get_combs(args.simname)

    def _main(comb):
        main(comb, args.kind, args.simname, args.min_logmass, args.sigma,
             args.mult, args.verbose)

    work_delegation(_main, combs, MPI.COMM_WORLD)
