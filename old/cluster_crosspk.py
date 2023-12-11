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
MPI script to calculate the matter cross power spectrum between CSiBORG
IC realisations. Units are Mpc/h.
"""
raise NotImplementedError("This script is currently not working.")
from argparse import ArgumentParser
from datetime import datetime
from gc import collect
from itertools import combinations
from os import remove
from os.path import join

import joblib
import numpy
import Pk_library as PKL
from mpi4py import MPI

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


dumpdir = "/mnt/extraspace/rstiskalek/csiborg/"
parser = ArgumentParser()
parser.add_argument("--grid", type=int)
parser.add_argument("--halfwidth", type=float, default=0.5)
args = parser.parse_args()

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
MAS = "CIC"  # mass asignment scheme

paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
box = csiborgtools.read.CSiBORG1Box(paths)
reader = csiborgtools.read.CSiBORG1Reader(paths)
ics = paths.get_ics("csiborg")
nsims = len(ics)

# File paths
ftemp = join(dumpdir, "temp_crosspk",
             "out_{}_{}" + "_{}".format(args.halfwidth))
fout = join(dumpdir, "crosspk",
            "out_{}_{}" + "_{}.p".format(args.halfwidth))


jobs = csiborgtools.utils.split_jobs(nsims, nproc)[rank]
for n in jobs:
    print(f"Rank {rank} at {datetime.now()}: saving {n}th delta.", flush=True)
    nsim = ics[n]
    particles = reader.read_snapshot(max(paths.get_snapshots(nsim, "csiborg")),
                                     nsim, ["x", "y", "z", "M"], verbose=False)
    # Halfwidth -- particle selection
    if args.halfwidth < 0.5:
        particles = csiborgtools.read.halfwidth_select(
            args.halfwidth, particles)
        length = box.box2mpc(2 * args.halfwidth) * box.h  # Mpc/h
    else:
        length = box.box2mpc(1) * box.h  # Mpc/h
    # Calculate the overdensity field
    field = csiborgtools.field.DensityField(particles, length, box, MAS)
    delta = field.overdensity_field(args.grid, verbose=False)
    aexp = box._aexp

    # Try to clean up memory
    del field, particles, box, reader
    collect()

    # Dump the results
    with open(ftemp.format(nsim, "delta") + ".npy", "wb") as f:
        numpy.save(f, delta)
    joblib.dump([aexp, length], ftemp.format(nsim, "lengths") + ".p")

    # Try to clean up memory
    del delta
    collect()


comm.Barrier()

# Get off-diagonal elements and append the diagoal
combs = [c for c in combinations(range(nsims), 2)]
for i in range(nsims):
    combs.append((i, i))
prev_delta = [-1, None, None, None]  # i, delta, aexp, length

jobs = csiborgtools.utils.split_jobs(len(combs), nproc)[rank]
for n in jobs:
    i, j = combs[n]
    print("Rank {}@{}: combination {}.".format(rank, datetime.now(), (i, j)))

    # If i same as last time then don't have to load it
    if prev_delta[0] == i:
        delta_i = prev_delta[1]
        aexp_i = prev_delta[2]
        length_i = prev_delta[3]
    else:
        with open(ftemp.format(ics[i], "delta") + ".npy", "rb") as f:
            delta_i = numpy.load(f)
        aexp_i, length_i = joblib.load(ftemp.format(ics[i], "lengths") + ".p")
        # Store in prev_delta
        prev_delta[0] = i
        prev_delta[1] = delta_i
        prev_delta[2] = aexp_i
        prev_delta[3] = length_i

    # Get jth delta
    with open(ftemp.format(ics[j], "delta") + ".npy", "rb") as f:
        delta_j = numpy.load(f)
    aexp_j, length_j = joblib.load(ftemp.format(ics[j], "lengths") + ".p")

    # Verify the difference between the scale factors! Say more than 1%
    daexp = abs((aexp_i - aexp_j) / aexp_i)
    if daexp > 0.01:
        raise ValueError(
            "Boxes {} and {} final snapshot scale factors disagree by "
            "`{}` percent!".format(ics[i], ics[j], daexp * 100))
    # Check how well the boxsizes agree
    dlength = abs((length_i - length_j) / length_i)
    if dlength > 0.001:
        raise ValueError("Boxes {} and {} box sizes disagree by `{}` percent!"
                         .format(ics[i], ics[j], dlength * 100))

    # Calculate the cross power spectrum
    Pk = PKL.XPk([delta_i, delta_j], length_i, axis=1, MAS=[MAS, MAS],
                 threads=1)
    joblib.dump(Pk, fout.format(ics[i], ics[j]))

    del delta_i, delta_j, Pk
    collect()


# Clean up the temp files
comm.Barrier()
if rank == 0:
    print("Cleaning up the temporary files...")
    for ic in ics:
        remove(ftemp.format(ic, "delta") + ".npy")
        remove(ftemp.format(ic, "lengths") + ".p")

    print("All finished!")
