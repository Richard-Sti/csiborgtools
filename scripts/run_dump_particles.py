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
A script to dump particle positions at the final and initial snapshot in a
format that is readable by Julia. Don't forget to delete once used in Julia as
these will take a lot of space (for all 100 simulations about 2TB), hence
should be processed in smaller batches.
"""
import numpy
from datetime import datetime
from os.path import join
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


init_paths = csiborgtools.read.CSiBORGPaths(to_new=True)
fin_paths = csiborgtools.read.CSiBORGPaths(to_new=False)
nsims = init_paths.ic_ids

dumpdir = "/mnt/extraspace/rstiskalek/csiborg/temp_initmatch_dump"
fpart = join(dumpdir, "out_{}_{}_{}.npy")
fclump = join(dumpdir, "out_{}_clumps.npy")
pars = ["x", "y", "z", "M", "ID"]

for nsim in nsims:
    print("{}: saving simulation {}.".format(datetime.now(), nsim))
    # Set the snapshot numbers
    init_paths.set_info(nsim, init_paths.get_minimum_snapshot(nsim))
    fin_paths.set_info(nsim, fin_paths.get_maximum_snapshot(nsim))
    # Set the readers
    init_reader = csiborgtools.read.ParticleReader(init_paths)
    fin_reader = csiborgtools.read.ParticleReader(fin_paths)

    # Read and dump the init particles
    particles = init_reader.read_particle(pars, verbose=False)
    # Put to a nicer array
    out = numpy.full((particles.size, 4), numpy.nan, dtype=numpy.float32)
    for i, par in enumerate(pars[:-1]):
        out[:, i] = particles[par]
    # Dump the particles and particle IDs
    with open(fpart.format(nsim, "init", "part"), 'wb') as f:
        numpy.save(f, out)
    # Rpeat for the particle IDs
    with open(fpart.format(nsim, "init", "ID"), 'wb') as f:
        numpy.save(f, particles["ID"])

    # Now repeat for the final snapshot but also get clump IDs
    particles = fin_reader.read_particle(pars, verbose=False)
    # Put to a nicer array
    out = numpy.full((particles.size, 4), numpy.nan, dtype=numpy.float32)
    for i, par in enumerate(pars[:-1]):
        out[:, i] = particles[par]
    # Dump the particles and particle IDs
    with open(fpart.format(nsim, "fin", "part"), 'wb') as f:
        numpy.save(f, out)
    # Rpeat for the particle IDs
    with open(fpart.format(nsim, "fin", "ID"), 'wb') as f:
        numpy.save(f, particles["ID"])
    # Clump IDs
    with open(fclump.format(nsim), 'wb') as f:
        numpy.save(f, fin_reader.read_clumpid(verbose=False))
