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
TODO: these docs
A script to calculate the centre of mass of particles at redshift 70 that
are grouped in a clump at present redshift.

Optionally also dumps the clumps information, however watch out as this will
eat up a lot of memory.
"""
from argparse import ArgumentParser
from datetime import datetime
from distutils.util import strtobool
from gc import collect
from os import remove
from os.path import join

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

# Argument parser
parser = ArgumentParser()
parser.add_argument("--dump", type=lambda x: bool(strtobool(x)))
args = parser.parse_args()
paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
# Temporary output file
ftemp = join(paths.dumpdir, "temp", "initmatch_{}_{}_{}.npy")

# We loop over all particles and then use MPI when matching halos to the
# initial snapshot and dumping them.
for nsim in paths.get_ics(tonew=True):
    if rank == 0:
        print("{}: reading simulation {}.".format(datetime.now(), nsim), flush=True)
    nsnap = max(paths.get_snapshots(nsim))
    reader = csiborgtools.read.ParticleReader(paths)

    # We first read all particles in the initial snap shot and sort them
    # according to their particle ID.
    part0 = reader.read_particle(1, nsim, ["x", "y", "z", "M", "ID"], verbose=False)
    part0 = part0[numpy.argsort(part0["ID"])]

    # Order the final snapshot clump IDs by the particle IDs
    pid = reader.read_particle(nsnap, nsim, ["ID"], verbose=False)["ID"]
    clump_ids = reader.read_clumpid(nsnap, nsim, verbose=False)
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
        print("{}: dumping intermediate files.".format(datetime.now()), flush=True)

    # Grab unique clump IDs and loop over them
    unique_clumpids = numpy.unique(clump_ids)

    njobs = unique_clumpids.size
    jobs = csiborgtools.utils.split_jobs(njobs, nproc)[rank]
    for i in jobs:
        n = unique_clumpids[i]
        x0 = part0[clump_ids == n]

        # Center of mass and Lagrangian patch size
        dist, cm = csiborgtools.match.dist_centmass(x0)
        patch = csiborgtools.match.dist_percentile(dist, [99], distmax=0.075)

        # Dump the center of mass
        with open(ftemp.format(nsim, n, "cm"), "wb") as f:
            numpy.save(f, cm)
        # Dump the Lagrangian patch size
        with open(ftemp.format(nsim, n, "lagpatch"), "wb") as f:
            numpy.save(f, patch)
        # Dump the entire clump
        if args.dump_clumps:
            with open(ftemp.format(nsim, n, "clump"), "wb") as f:
                numpy.save(f, x0)

    del part0, clump_ids
    collect()

    comm.Barrier()
    if rank == 0:
        print("{}: collecting summary files...".format(datetime.now()), flush=True)
        # Collect the centre of masses, patch size, etc. and dump them
        dtype = {
            "names": ["x", "y", "z", "lagpatch", "ID"],
            "formats": [numpy.float32] * 4 + [numpy.int32],
        }
        out = numpy.full(njobs, numpy.nan, dtype=dtype)

        for i, n in enumerate(unique_clumpids):
            # Load in CM vector
            fpath = ftemp.format(nsim, n, "cm")
            with open(fpath, "rb") as f:
                fin = numpy.load(f)
                out["x"][i] = fin[0]
                out["y"][i] = fin[1]
                out["z"][i] = fin[2]
            remove(fpath)

            # Load in the patch size
            fpath = ftemp.format(nsim, n, "lagpatch")
            with open(fpath, "rb") as f:
                out["lagpatch"][i] = numpy.load(f)
            remove(fpath)

            # Store the halo ID
            out["ID"][i] = n

        print(
            "{}: dumping to .. `{}`.".format(
                datetime.now(), paths.initmatch_path(nsim, "cm")
            ),
            flush=True,
        )
        with open(paths.initmatch_path(nsim, "cm"), "wb") as f:
            numpy.save(f, out)

        if args.dump_clumps:
            print("{}: collecting particle files...".format(datetime.now()), flush=True)
            out = [None] * unique_clumpids.size
            dtype = {"names": ["clump", "ID"], "formats": [object, numpy.int32]}
            out = numpy.full(unique_clumpids.size, numpy.nan, dtype=dtype)
            for i, n in enumerate(unique_clumpids):
                fpath = ftemp.format(nsim, n, "clump")
                with open(fpath, "rb") as f:
                    fin = numpy.load(f)
                out["clump"][i] = fin
                out["ID"][i] = n
                remove(fpath)

            fout = paths.initmatch_path(nsim, "particles")
            print("{}: dumping to .. `{}`.".format(datetime.now(), fout), flush=True)
            with open(fout, "wb") as f:
                numpy.save(f, out)

            del out
            collect()
