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
from datetime import datetime
from distutils.util import strtobool
from itertools import combinations
from random import Random

from mpi4py import MPI

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools

from taskmaster import master_process, worker_process

from match_singlematch import pair_match

# Argument parser
parser = ArgumentParser()
parser.add_argument("--sigma", type=float, default=None)
parser.add_argument("--smoothen", type=lambda x: bool(strtobool(x)),
                    default=None)
parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                    default=False)
args = parser.parse_args()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


def get_combs():
    """
    Get the list of all pairs of simulations, then permute them with a known
    seed and assign each process a chunk of the list.
    """
    paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
    ics = paths.get_ics()
    combs = list(combinations(ics, 2))
    Random(42).shuffle(combs)
    combs = combs[:10]  # NOTE remove later
    return combs


def do_work(comb):
    nsim0, nsimx = comb
    pair_match(nsim0, nsimx, args.sigma, args.smoothen, args.verbose)


if nproc > 1:
    if rank == 0:
        combs = get_combs()
        master_process(combs, comm, verbose=True)
    else:
        worker_process(do_work, comm, verbose=False)
else:
    combs = get_combs()
    for comb in combs:
        print(f"{datetime.now()}: completing task `{comb}`.", flush=True)
        do_work(comb)
