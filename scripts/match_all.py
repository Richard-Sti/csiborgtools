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
Script to match all pairs of CSiBORG simulations. Mathches main haloes whose
mass is above 1e12 solar masses.
"""
from argparse import ArgumentParser
from distutils.util import strtobool
from itertools import combinations
from random import Random

from mpi4py import MPI
from taskmaster import work_delegation

from match_singlematch import pair_match

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools


def get_combs():
    """
    Get the list of all pairs of simulations, then permute them with a known
    seed to minimise loading the same files simultaneously.

    Returns
    -------
    combs : list
        List of pairs of simulations.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    ics = paths.get_ics("csiborg")
    combs = list(combinations(ics, 2))
    Random(42).shuffle(combs)
    return combs


def do_work(comb):
    """
    Match a pair of simulations.

    Parameters
    ----------
    comb : tuple
        Pair of simulations.

    Returns
    -------
    None
    """
    nsim0, nsimx = comb
    pair_match(nsim0, nsimx, args.sigma, args.smoothen, args.verbose)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--smoothen", type=lambda x: bool(strtobool(x)),
                        default=None)
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        default=False)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    combs = get_combs()
    work_delegation(do_work, combs, comm, master_verbose=True)
