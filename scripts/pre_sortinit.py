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
Script to sort the initial snapshot particles according to their final
snapshot ordering, which is sorted by the clump IDs.
"""
from argparse import ArgumentParser
from datetime import datetime

import h5py
from gc import collect
import numpy
from mpi4py import MPI

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools


# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
verbose = nproc == 1

# Argument parser
parser = ArgumentParser()
parser.add_argument("--ics", type=int, nargs="+", default=None,
                    help="IC realisations. If `-1` processes all simulations.")
args = parser.parse_args()
paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
partreader = csiborgtools.read.ParticleReader(paths)
# NOTE: ID has to be the last column.
pars_extract = ["x", "y", "z", "M", "ID"]

if args.ics is None or args.ics[0] == -1:
    ics = paths.get_ics()
else:
    ics = args.ics

# We loop over simulations. Each simulation is then procesed with MPI, rank 0
# loads the data and broadcasts it to other ranks.
jobs = csiborgtools.fits.split_jobs(len(ics), nproc)[rank]
for i in jobs:
    nsim = ics[i]
    nsnap = max(paths.get_snapshots(nsim))

    print(f"{datetime.now()}: reading and processing simulation {nsim}.",
          flush=True)
    # We first load the particle IDs in the final snapshot.
    pidf = csiborgtools.read.read_h5(paths.particles_path(nsim))
    pidf = pidf["particle_ids"]
    # Then we load the particles in the initil snapshot and make sure that
    # their particle IDs are sorted as in the final snapshot.
    # Again, because of precision this must be read as structured.
    part0, pid0 = partreader.read_particle(
        1, nsim, pars_extract, return_structured=False, verbose=verbose)
    # First enforce them to already be sorted and then apply reverse
    # sorting from the final snapshot.
    part0 = part0[numpy.argsort(pid0)]
    del pid0
    collect()
    part0 = part0[numpy.argsort(numpy.argsort(pidf))]
    print(f"{datetime.now()}: dumping particles for {nsim}.", flush=True)
    with h5py.File(paths.initmatch_path(nsim, "particles"), "w") as f:
        f.create_dataset("particles", data=part0)
