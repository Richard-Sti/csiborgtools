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

NOTE:
- Calculate for the entire box or just for a smaller region?
- Add argparser for different options.
- In the argparser add options to smoothen the field.
"""
import numpy
from datetime import datetime
from mpi4py import MPI
from os.path import join
from os import remove
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools
import utils

halfwidth = 0.5
MAS = "CIC"
grid = 256

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

# Galaxy positions
survey = "SDSS"
survey = utils.surveys[survey]()()
pos = numpy.vstack([survey[p] for p in ("DIST", "RA", "DEC")]).T
pos = pos.astype(numpy.float32)

# File paths
ftemp = join(utils.dumpdir, "temp_fields", "out_" + survey.name + "_{}.npy")
fperm = join(utils.dumpdir, "fields", "out_{}.npy".format(survey.name))

# Edit depending on what is calculated
dtype = {"names": ["delta", "phi"], "formats": [numpy.float32] * 2}

# CSiBORG simulation paths
paths = csiborgtools.read.CSiBORGPaths()
ics = paths.ic_ids[:10]
n_sims = len(ics)

for n in csiborgtools.fits.split_jobs(n_sims, nproc)[rank]:
    print("Rank {}@{}: working on {}th IC.".format(rank, datetime.now(), n),
          flush=True)
    # Set the paths
    n_sim = ics[n]
    paths.set_info(n_sim, paths.get_maximum_snapshot(n_sim))

    # Set reader and the box
    reader = csiborgtools.read.ParticleReader(paths)
    box = csiborgtools.units.BoxUnits(paths)

    # Read particles and select a subset of them
    particles = reader.read_particle(["x", "y", "z", "M"], verbose=False)
    if halfwidth < 0.5:
        particles = csiborgtools.read.halfwidth_select(halfwidth, particles)
        length = box.box2mpc(2 * halfwidth) * box.h  # Mpc/h
    else:
        length = box.box2mpc(1) * box.h  # Mpc/h

    # Initialise the field object and output array
    field = csiborgtools.field.DensityField(particles, length, box, MAS)
    out = numpy.full(pos.shape[0], numpy.nan, dtype=dtype)

    # Calculate the overdensity field and interpolate at galaxy positions
    feval = field.overdensity_field(grid, verbose=False)
    out["delta"] = field.evaluate_sky(feval, pos=pos, isdeg=True)[0]

    # Potential
    feval = field.potential_field(grid, verbose=False)
    out["phi"] = field.evaluate_sky(feval, pos=pos, isdeg=True)[0]

    # Calculate the remaining fields
    # ...
    # ...

    # Dump the results
    with open(ftemp.format(n_sim), "wb") as f:
        numpy.save(f, out)

# Wait for all ranks to finish
comm.Barrier()
if rank == 0:
    print("Collecting files...", flush=True)

    out = numpy.full((n_sims, pos.shape[0]), numpy.nan, dtype=dtype)

    for n in range(n_sims):
        n_sim = ics[n]
        with open(ftemp.format(n_sim), "rb") as f:
            fin = numpy.load(f, allow_pickle=True)
            for name in dtype["names"]:
                out[name][n, ...] = fin[name]
        # Remove the temporary file
        remove(ftemp.format(n_sim))

    print("Saving results to `{}`.".format(fperm), flush=True)
    with open(fperm, "wb") as f:
        numpy.save(f, out)
