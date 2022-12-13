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
A script to calculate the centre of mass of particles at redshift 70 that
are grouped in a clump at present redshift.

Optionally also dumps the clumps information, however watch out as this will
eat up a lot of memory.
"""
from argparse import ArgumentParser
import numpy
from datetime import datetime
from mpi4py import MPI
from distutils.util import strtobool
from os.path import join, isdir
from os import mkdir
from os import remove
from sys import stdout
from gc import collect
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools

parser = ArgumentParser()
parser.add_argument("--dump_clumps", default=False,
                    type=lambda x: bool(strtobool(x)))
args = parser.parse_args()

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

init_paths = csiborgtools.read.CSiBORGPaths(to_new=True)
fin_paths = csiborgtools.read.CSiBORGPaths(to_new=False)
nsims = init_paths.ic_ids

# Output files
dumpdir = "/mnt/extraspace/rstiskalek/csiborg/initmatch"
ftemp = join(dumpdir, "temp", "temp_{}_{}.npy")
fperm = join(dumpdir, "clump_cm_{}.npy")

for nsim in nsims:
    if rank == 0:
        print("{}: reading simulation {}.".format(datetime.now(), nsim))
        stdout.flush()
    # Check that the output folder for this sim exists
    clumpdumpdir = join(dumpdir, "out_{}".format(nsim))
    if args.dump_clumps and rank == 0 and not isdir(clumpdumpdir):
        mkdir(clumpdumpdir)

    # Barrier to make sure we created the directory with the rank 0
    comm.Barrier()

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
        x0 = part0[clump_ids == n]

        # Center of mass
        cm = numpy.asanyarray(
            [numpy.average(x0[p], weights=x0["M"]) for p in ('x', 'y', 'z')])
        # Dump the center of mass
        with open(ftemp.format(nsim, n), 'wb') as f:
            numpy.save(f, cm)
        # Optionally dump the entire clump
        if args.dump_clumps:
            fout = join(clumpdumpdir, "clump_{}.npy".format(n))
            stdout.flush()
            with open(fout, "wb") as f:
                numpy.save(f, x0)

    comm.Barrier()
    if rank == 0:
        print("Collecting CM files...")
        stdout.flush()
        # Collect the centre of masses and dump them
        dtype = {"names": ['x', 'y', 'z', "ID"],
                 "formats": [numpy.float32] * 3 + [numpy.int32]}
        out = numpy.full(njobs, numpy.nan, dtype=dtype)

        for i, n in enumerate(unique_clumpids):
            with open(ftemp.format(nsim, n), 'rb') as f:
                fin = numpy.load(f)
            out['x'][i] = fin[0]
            out['y'][i] = fin[1]
            out['z'][i] = fin[2]
            out["ID"][i] = n
            remove(ftemp.format(nsim, n))

        print("Dumping CM files to .. `{}`.".format(fperm.format(nsim)))
        with open(fperm.format(nsim), 'wb') as f:
            numpy.save(f, out)
