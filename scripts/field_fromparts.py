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
MPI script to calculate the density fields on CSiBORG simulations in the final
snapshot.
"""
from argparse import ArgumentParser
from datetime import datetime
from distutils.util import strtobool

import numpy
from mpi4py import MPI

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
verbose = nproc == 1

parser = ArgumentParser()
parser.add_argument("--ics", type=int, nargs="+", default=None,
                    help="IC realisations. If `-1` processes all simulations.")
parser.add_argument("--kind", type=int, nargs="+", default=None,
                    help="IC realisations. If `-1` processes all simulations.")

parser.add_argument("--grid", type=int, help="Grid resolution.")
parser.add_argument("--in_rsp", type=lambda x: bool(strtobool(x)),
                    help="Calculate the density field in redshift space?")
parser.add_argument("--MAS", type=str, choices=["NGP", "CIC", "TSC", "PCS"])
args = parser.parse_args()
paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)

if args.ics is None or args.ics[0] == -1:
    ics = paths.get_ics()
else:
    ics = args.ics


for i in csiborgtools.fits.split_jobs(len(ics), nproc)[rank]:
    nsim = ics[i]
    print(f"{datetime.now()}: rank {rank} working on simulation {nsim}.",
          flush=True)

    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.BoxUnits(nsnap, nsim, paths)
    parts = csiborgtools.read.read_h5(paths.particles_path(nsim))["particles"]
    density_generator = csiborgtools.field.DensityField(box, args.MAS)

    rho = density_generator(parts, args.grid, in_rsp=args.in_rsp,
                            verbose=verbose)

    fout = paths.density_field_path(args.MAS, nsim, args.in_rsp)
    print(f"{datetime.now()}: rank {rank} saving output to `{fout}`.")
    numpy.save(fout, rho)
