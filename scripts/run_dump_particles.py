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
format that is readable by Julia. Sorts the particle array by their particle
ID so that the index positions between the initial and final snapshots match.

Don't forget to delete once used in Julia as these will take a lot of space
(for all 100 simulations about 2TB), hence should be processed in smaller
batches.
"""
import numpy
from datetime import datetime
from mpi4py import MPI
from os.path import join, isdir
from os import mkdir
from sys import stdout
from gc import collect
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

init_paths = csiborgtools.read.CSiBORGPaths(to_new=True)
fin_paths = csiborgtools.read.CSiBORGPaths(to_new=False)
nsims = init_paths.ic_ids

# Output files
tempdumpdir = "/mnt/extraspace/rstiskalek/csiborg/temp_initmatch_dump"
fpart = join(tempdumpdir, "out_{}_{}_{}.npy")
fclump = join(tempdumpdir, "out_{}_clumps.npy")

permdumpdir = "/mnt/extraspace/rstiskalek/csiborg/temp_initmatch"

for nsim in nsims[:1]:
    if rank == 0:
        print("{}: reading simulation {}.".format(datetime.now(), nsim))
        stdout.flush()
    # Check that the output folder for this sim exists
    simdumpdir = join(permdumpdir, "out_{}".format(nsim))
    if not isdir(simdumpdir):
        mkdir(simdumpdir)
    # Set the snapshot numbers
    init_paths.set_info(nsim, init_paths.get_minimum_snapshot(nsim))
    fin_paths.set_info(nsim, fin_paths.get_maximum_snapshot(nsim))
    # Set the readers
    init_reader = csiborgtools.read.ParticleReader(init_paths)
    fin_reader = csiborgtools.read.ParticleReader(fin_paths)

    # Read and sort the initial particle files by their particle IDs
    part0 = init_reader.read_particle(["x", "y", "z", "M", "ID"],
                                      verbose=False)
    part0 = part0[numpy.argsort(part0["ID"])]

    # Order the final snapshot clump IDs by the particle IDs
    pid = fin_reader.read_particle(["ID"], verbose=False)["ID"]
    clump_ids = fin_reader.read_clumpid(verbose=False)
    clump_ids = clump_ids[numpy.argsort(pid)]

    del pid
    collect()
    # Get rid of the clumps whose index is 0 -- those are unassigned
    mask = clump_ids > 0
    clump_ids = clump_ids[mask]
    part0 = part0[mask]
    del mask
    collect()

    if rank == 0:
        print("{}: dumping clumps for simulation.".format(datetime.now()))
        stdout.flush()

    # Grab unique clump IDs and loop over them
    unique_clumpids = numpy.unique(clump_ids)

    njobs = unique_clumpids.size
    jobs = csiborgtools.fits.split_jobs(njobs, nproc)[rank]
    for i in jobs:
        n = unique_clumpids[i]
        fout = join(simdumpdir, "clump_{}.npy".format(n))
        print("Dumping clump {} to .. `{}`".format(n, fout))
        stdout.flush()
        with open(fout, "wb") as f:
            numpy.save(f, part0[clump_ids == n])
