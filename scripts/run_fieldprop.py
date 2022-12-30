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
MPI script to run the CSiBORG realisations matcher.

NOTE:
- The script has *not* been tested.
- Calculate for the entire box or just for a smaller region?
"""
import numpy
from datetime import datetime
from mpi4py import MPI
from gc import collect
from os.path import join
from os import remove
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools
import utils

halfwidth = 0.2
MAS = "CIC"
grid = 256

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

# Galaxy positions
survey = "SDSS"
survey = utils.surveys[survey]()()
pos = [survey[p] for p in ("DIST", "RA", "DEC")]

# File paths
ftemp = join(utils.dumpdir, "temp_fields", "out_" + survey.name + "{}.npy")
fperm = join(utils.dumpdir, "fields", "out_{}.npy".format(survey.name))

# Edit depending on what is calculated
dtype = {"names": ["delta"], "formats": [numpy.float32]}

# CSiBORG simulation paths
paths = csiborgtools.read.CSiBORGPaths()
ics = paths.ic_ids
n_sims = len(ics)

for n in csiborgtools.fits.split_jobs(n_sims, nproc)[rank]:
    print("Rank {}@{}: saving {}th delta.".format(rank, datetime.now(), n),
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
    delta = field.overdensity_field(grid, verbose=False)
    delta = field.evaluate_sky(delta, pos=pos, isdeg=True)
    out["delta"] = delta
    del delta
    collect()

    # Calculate the remaining fields
    # ...
    # ...

    # Dump the results
    with open(ftemp.format(n_sim), "wb") as f:
        numpy.save(f, out)

    del out
    collect()

# Wait for all ranks to finish
comm.Barrier()
if rank == 0:
    print("Collecting files...", flush=True)

    out = numpy.full((len(n_sims), pos.shape[0]), numpy.nan, dtype=dtype)

    for i, n_sim in enumerate(n_sims):
        with open(ftemp.format(n_sim), "rb") as f:
            fin = numpy.load(f, allow_pickle=True)
            for name in dtype["names"]:
                out["delta"][i, ...] = fin["delta"]

        # Remove the temporary file
        remove(ftemp.format(n_sim))

    print("Saving results to `{}`.".format(fperm))
    with open(fperm, "wb") as f:
        numpy.save(f, out)
