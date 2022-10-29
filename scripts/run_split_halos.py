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
membership for faster manipulation. Running this will require a lot of memory.
"""
from tqdm import tqdm
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools
import utils

Nsims = [9844]
Nsnap = 1016
partcols = ["x", "y", "z", "M", "level"]

for Nsim in tqdm(Nsims):
    simpath = csiborgtools.io.get_sim_path(Nsim)
    # Load the clumps, particles' clump IDs and particles.
    clumps = csiborgtools.io.read_clumps(Nsnap, simpath)
    particle_clumps = csiborgtools.io.read_clumpid(Nsnap, simpath,
                                                   verbose=False)
    particles = csiborgtools.io.read_particle(partcols, Nsnap, simpath,
                                              verbose=False)
    # Drop all particles whose clump index is 0 (not assigned to any halo)
    particle_clumps, particles = csiborgtools.io.drop_zero_indx(
        particle_clumps, particles)
    # Get only clumps that do have any particles
    with_particles = csiborgtools.fits.clump_with_particles(particle_clumps,
                                                            clumps)
    clumps = clumps[with_particles]
    # Dump it!
    csiborgtools.fits.dump_split_particles(particles, particle_clumps, clumps,
                                           utils.Nsplits, utils.dumppath, Nsim,
                                           Nsnap, verbose=False)

print("All finished!")
