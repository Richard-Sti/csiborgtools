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
Script to calculate the peculiar velocity of an observer in the centre of the
CSiBORG box.
"""
from argparse import ArgumentParser
from distutils.util import strtobool

import numpy
from mpi4py import MPI

from taskmaster import work_delegation
from tqdm import tqdm
from utils import get_nsims

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


def observer_peculiar_velocity(nsim, parser_args):
    """
    Calculate the peculiar velocity of an observer in the centre of the box
    for several smoothing scales.
    """
    pos = numpy.array([0.5, 0.5, 0.5]).reshape(-1, 3)
    boxsize = 677.7
    smooth_scales = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    observer_vp = numpy.full((len(smooth_scales), 3), numpy.nan,
                             dtype=numpy.float32)

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    field_path = paths.field("velocity", parser_args.MAS, parser_args.grid,
                             nsim, in_rsp=False)
    field0 = numpy.load(field_path)

    for j, smooth_scale in enumerate(tqdm(smooth_scales,
                                          desc="Smoothing the fields",
                                          disable=not parser_args.verbose)):
        if smooth_scale > 0:
            field = [None, None, None]
            for k in range(3):
                field[k] = csiborgtools.field.smoothen_field(
                    field0[k], smooth_scale, boxsize)
        else:
            field = field0

        v = csiborgtools.field.evaluate_cartesian(
            field[0], field[1], field[2], pos=pos)
        observer_vp[j, 0] = v[0][0]
        observer_vp[j, 1] = v[1][0]
        observer_vp[j, 2] = v[2][0]

    fout = paths.observer_peculiar_velocity(parser_args.MAS, parser_args.grid,
                                            nsim)
    if parser_args.verbose:
        print(f"Saving to ... `{fout}`")
    numpy.savez(fout, smooth_scales=smooth_scales, observer_vp=observer_vp)
    return observer_vp


###############################################################################
#                          Command line interface                             #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. `-1` for all simulations.")
    parser.add_argument("--kind", type=str,
                        choices=["density", "rspdensity", "velocity", "radvel",
                                 "potential", "environment"],
                        help="What derived field to calculate?")
    parser.add_argument("--MAS", type=str,
                        choices=["NGP", "CIC", "TSC", "PCS"])
    parser.add_argument("--grid", type=int, help="Grid resolution.")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        help="Verbosity flag for reading in particles.")
    parser.add_argument("--simname", type=str, default="csiborg",
                        help="Verbosity flag for reading in particles.")
    parser_args = parser.parse_args()

    comm = MPI.COMM_WORLD
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(parser_args, paths)

    def main(nsim):
        return observer_peculiar_velocity(nsim, parser_args)

    work_delegation(main, nsims, comm, master_verbose=True)
