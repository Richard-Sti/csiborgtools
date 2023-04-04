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
snapshot of each simulation. Running this requires a lot of memory.
"""
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

paths = csiborgtools.read.CSiBORGPaths()
nsims = len(paths.ic_ids(False))
partcols = ["x", "y", "z", "vx", "vy", "vz", "M", "level"]

jobs = csiborgtools.fits.split_jobs(len(nsims), nproc)[rank]
for icount, sim_index in enumerate(jobs):
    print("{}: rank {} working {} / {} jobs."
          .format(datetime.now(), rank, icount + 1, len(jobs)), flush=True)
    nsim = nsims[sim_index]
    nsnap = max(paths.get_snapshots(nsim))
    partreader = csiborgtools.read.ParticleReader(paths)
    # Load the clumps, particles' clump IDs and particles.
    clumps = partreader.read_clumps(nsnap, nsim)
    particle_clumps = partreader.read_clumpid(nsnap, nsim, verbose=False)
    particles = partreader.read_particle(nsnap, nsim, partcols, verbose=False)
    # Drop all particles whose clump index is 0 (not assigned to any halo)
    particle_clumps, particles = partreader.drop_zero_indx(
        particle_clumps, particles)
    # Dump it!
    csiborgtools.fits.dump_split_particles(particles, particle_clumps, clumps,
                                           utils.Nsplits, nsnap, nsim, paths,
                                           verbose=False)

print("All finished!", flush=True)
