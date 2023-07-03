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
Short script to move and change format of the FoF membership files calculated
by Julien.
"""
from argparse import ArgumentParser
from os.path import join

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


def move_membership(nsim):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    fpath = join("/mnt/extraspace/jeg/greenwhale/Constrained_Sims",
                 f"sim_{nsim}/particle_membership_{nsim}_FOF.txt")
    print(f"Loading from ... `{fpath}`.")
    data = numpy.genfromtxt(fpath, dtype=int)

    fout = paths.fof_membership(nsim)
    print(f"Saving to ... `{fout}`.")
    numpy.save(fout, data)


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
    comm = MPI.COMM_WORLD

    work_delegation(move_membership, nsims, comm)
