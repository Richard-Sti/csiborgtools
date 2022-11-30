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
"""A script to dump or remove files for POWMES."""

from argparse import ArgumentParser
import numpy
from datetime import datetime
from os.path import join, exists
from os import remove
from mpi4py import MPI
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools
import utils

parser = ArgumentParser()
parser.add_argument("--mode", type=str, choices=["dump", "remove"])
args = parser.parse_args()

F64 = numpy.float64
I64 = numpy.int64

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


dumpdir = join(utils.dumpdir, "temp_powmes")
fout = join(dumpdir, "out_{}_{}.ascii")
paths = csiborgtools.read.CSiBORGPaths()


n_sims = paths.ic_ids[:1]
for i in csiborgtools.fits.split_jobs(len(n_sims), nproc)[rank]:
    print("{}: calculating {}th simulation.".format(datetime.now(), i))
    n_sim = n_sims[i]
    n_snap = paths.get_maximum_snapshot(n_sim)
    paths.set_info(n_sim, n_snap)

    f = fout.format(n_sim, n_snap)
    if args.mode == "dump":
        # Read the particles
        reader = csiborgtools.read.ParticleReader(paths)
        particles = reader.read_particle(["x", "y", "z", "M"])
        csiborgtools.read.make_ascii_powmes(particles, f, verbose=True)
    else:
        if exists(f):
            remove(f)

comm.Barrier()
if rank == 0:
    print("All finished! See you!")
