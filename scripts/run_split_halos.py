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
Script to split particles into smaller files according to their clump
membership for faster manipulation. Currently does this for the maximum
snapshot of each simulation. Running this will require a lot of memory.
"""

from os.path import join
from mpi4py import MPI
from datetime import datetime
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools
import utils

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

Nsims = csiborgtools.read.get_csiborg_ids("/mnt/extraspace/hdesmond")
partcols = ["x", "y", "z", "vx", "vy", "vz", "M", "level"]
dumpdir = join(utils.dumpdir, "temp")

jobs = csiborgtools.fits.split_jobs(len(Nsims), nproc)[rank]
for icount, sim_index in enumerate(jobs):
    print("{}: rank {} working {} / {} jobs.".format(datetime.now(), rank,
                                                     icount + 1, len(jobs)))
    Nsim = Nsims[sim_index]
    simpath = csiborgtools.read.get_sim_path(Nsim)
    Nsnap = csiborgtools.read.get_maximum_snapshot(simpath)
    # Load the clumps, particles' clump IDs and particles.
    clumps = csiborgtools.read.read_clumps(Nsnap, simpath)
    particle_clumps = csiborgtools.read.read_clumpid(Nsnap, simpath,
                                                     verbose=False)
    particles = csiborgtools.read.read_particle(partcols, Nsnap, simpath,
                                                verbose=False)
    # Drop all particles whose clump index is 0 (not assigned to any halo)
    particle_clumps, particles = csiborgtools.read.drop_zero_indx(
        particle_clumps, particles)
    # Dump it!
    csiborgtools.fits.dump_split_particles(particles, particle_clumps, clumps,
                                           utils.Nsplits, dumpdir, Nsim, Nsnap,
                                           verbose=False)

print("All finished!")
