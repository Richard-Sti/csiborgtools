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
n_sims = paths.ic_ds[:1]
partcols = ["x", "y", "z", "vx", "vy", "vz", "M", "level"]

jobs = csiborgtools.fits.split_jobs(len(n_sims), nproc)[rank]
for icount, sim_index in enumerate(jobs):
    print("{}: rank {} working {} / {} jobs.".format(datetime.now(), rank,
                                                     icount + 1, len(jobs)))
    n_sim = n_sims[sim_index]
    n_snap = paths.get_maximum_snapshot(n_sim)
    # Set paths and inifitalise a particle reader
    paths.set_info(n_sim, n_snap)
    partreader = csiborgtools.read.ParticleReader(paths)
    # Load the clumps, particles' clump IDs and particles.
    clumps = partreader.read_clumps()
    particle_clumps = partreader.read_clumpid(verbose=False)
    particles = partreader.read_particle(partcols, verbose=False)
    # Drop all particles whose clump index is 0 (not assigned to any halo)
    particle_clumps, particles = partreader.drop_zero_indx(
        particle_clumps, particles)
    # Dump it!
    # NOTE this function still needs to be edited
    csiborgtools.fits.dump_split_particles(particles, particle_clumps, clumps,
                                           utils.Nsplits, paths, verbose=False)

print("All finished!")
