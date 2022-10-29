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
A script to fit halos (concentration, ...). The particle array of each CSiBORG
realisation must have been split in advance by `run_split_halos`.
"""

import numpy
from os.path import join
from tqdm import trange
from mpi4py import MPI
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools
import utils

F64 = numpy.float64
I64 = numpy.int64

# Simulations and their snapshot to analyze
Nsims = [9844]
Nsnap = 1016

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

print("I am rank {} out of nproc {}".format(rank, nproc))

# Here will have to split the jobs.... Wait until we get the array
jobs = csiborgtools.fits.split_jobs(utils.Nsplits, nproc)[rank]


loaddir = join(utils.dumpdir, "temp")
dumpdir = utils.dumpdir

Nsim = Nsims[0]

for Nsplit in jobs:
    print(Nsplit)
    parts, part_clumps, clumps = csiborgtools.fits.load_split_particles(
        Nsplit, loaddir, Nsim, Nsnap)

    N = clumps.size
    cols = [("index", I64), ("npart", I64), ("totpartmass", F64),
            ("logRs", F64)]
    out = csiborgtools.utils.cols_to_structured(N, cols)
    out["index"] = clumps["index"]

    for n in trange(N):
        # Pick clump and its particles
        xs = csiborgtools.fits.pick_single_clump(n, parts, part_clumps, clumps)
        clump = csiborgtools.fits.Clump.from_arrays(*xs)
        out["npart"][n] = clump.Npart
        out["totpartmass"][n] = clump.total_particle_mass

        # NFW profile fit
        nfwpost = csiborgtools.fits.NFWPosterior(clump)
        logRs = nfwpost.maxpost_logRs()
        if logRs.success:
            out["logRs"][n] = logRs.x

    csiborgtools.io.dump_split(out, Nsplit, Nsim, Nsnap, dumpdir)

# Force all ranks to wait
comm.Barrier()
# Use the rank 0 to combine outputs for this CSiBORG realisation
if rank == 0:
    print("All finished! See ya!")
