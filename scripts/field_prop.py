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
MPI script to evaluate field properties at the galaxy positions.
"""
from argparse import ArgumentParser
from datetime import datetime
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

import utils

dumpdir = "/mnt/extraspace/rstiskalek/csiborg/"
parser = ArgumentParser()
parser.add_argument("--survey", type=str, choices=["SDSS"])
parser.add_argument("--grid", type=int)
parser.add_argument("--MAS", type=str, choices=["NGP", "CIC", "TSC", "PCS"])
parser.add_argument("--halfwidth", type=float)
parser.add_argument("--smooth_scale", type=float, default=None)
args = parser.parse_args()
# Smooth scale of 0 means no smoothing. Note that this is in Mpc/h
args.smooth_scale = None if args.smooth_scale == 0 else args.smooth_scale

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

# Galaxy positions
survey = utils.surveys[args.survey]()()
pos = numpy.vstack([survey[p] for p in ("DIST", "RA", "DEC")]).T
pos = pos.astype(numpy.float32)

# File paths
fname = "out_{}_{}_{}_{}_{}".format(
    survey.name, args.grid, args.MAS, args.halfwidth, args.smooth_scale)
ftemp = join(dumpdir, "temp_fields", fname + "_{}.npy")
fperm = join(dumpdir, "fields", fname + ".npy")

# Edit depending on what is calculated
dtype = {"names": ["delta", "phi"], "formats": [numpy.float32] * 2}

# CSiBORG simulation paths
paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
ics = paths.get_ics(tonew=False)
nsims = len(ics)

for n in csiborgtools.utils.split_jobs(nsims, nproc)[rank]:
    print("Rank {}@{}: working on {}th IC.".format(rank, datetime.now(), n),
          flush=True)
    nsim = ics[n]
    nsnap = max(paths.get_snapshots(nsim))
    reader = csiborgtools.read.ParticleReader(paths)
    box = csiborgtools.read.BoxUnits(nsnap, nsim, paths)

    # Read particles and select a subset of them
    particles = reader.read_particle(nsnap, nsim, ["x", "y", "z", "M"],
                                     verbose=False)
    if args.halfwidth < 0.5:
        particles = csiborgtools.read.halfwidth_select(
            args.halfwidth, particles)
        length = box.box2mpc(2 * args.halfwidth) * box.h  # Mpc/h
    else:
        length = box.box2mpc(1) * box.h  # Mpc/h

    # Initialise the field object and output array
    field = csiborgtools.field.DensityField(particles, length, box, args.MAS)
    out = numpy.full(pos.shape[0], numpy.nan, dtype=dtype)

    # Calculate the overdensity field and interpolate at galaxy positions
    feval = field.overdensity_field(args.grid, args.smooth_scale,
                                    verbose=False)
    out["delta"] = field.evaluate_sky(feval, pos=pos, isdeg=True)[0]

    # Potential
    feval = field.potential_field(args.grid, args.smooth_scale, verbose=False)
    out["phi"] = field.evaluate_sky(feval, pos=pos, isdeg=True)[0]

    # Calculate the remaining fields
    # ...
    # ...

    # Dump the results
    with open(ftemp.format(nsim), "wb") as f:
        numpy.save(f, out)

# Wait for all ranks to finish
comm.Barrier()
if rank == 0:
    print("Collecting files...", flush=True)

    out = numpy.full((nsims, pos.shape[0]), numpy.nan, dtype=dtype)

    for n in range(nsims):
        nsim = ics[n]
        with open(ftemp.format(nsim), "rb") as f:
            fin = numpy.load(f, allow_pickle=True)
            for name in dtype["names"]:
                out[name][n, ...] = fin[name]
        # Remove the temporary file
        remove(ftemp.format(nsim))

    print("Saving results to `{}`.".format(fperm), flush=True)
    with open(fperm, "wb") as f:
        numpy.save(f, out)
