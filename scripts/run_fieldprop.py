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
This script is currently utterly broken.
- [ ] Calculate for the entire box -- 2048**3 yields resolution of up to 0.3 Mpc
But considering Nyquist we get about 0.6, which is still well above the scale of
the constraints which is 2.65 Mpc/h.
- [ ] Begin by calculating the overdensity field (1 number), gravitaitonal field (vector) and tidal field (tensor)
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

halfwidth = 0.2
MAS = "PCS"
grid = 1024

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

# File paths
ftemp = join(utils.dumpdir, "temp_match", "match_{}.npy")
fperm = join(utils.dumpdir, "match", "cross_matches.npy")


def steps(cls):
    return [(lambda x: cls[x], ("IN_DR7_LSS",)),
            (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
            (lambda x: cls[x] < 155, ("DIST", ))
            ]


surv = csiborgtools.read.SDSS(h=1, sel_steps=steps)
# Galaxy positions
pos = [surv[p] for p in ("DIST", "RA", "DEC")]

paths = csiborgtools.read.CSiBORGPaths()
ics = paths.ic_ids
n_sims = len(ics)

for n in csiborgtools.fits.split_jobs(n_sims, nproc)[rank]:
    print("Rank {}@{}: saving {}th delta.".format(rank, datetime.now(), n))
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


    # Calculate the overdensity field
    field = csiborgtools.field.DensityField(particles, length, box, MAS)

    delta = field.overdensity_field(grid, verbose=False)

    # Evaluate the potential
    potential = field.potential_field(grid, verbose=False)


    # Try to clean up memory
    del field, particles, box, reader, mask
    collect()

    # Dump the results
    with open(ftemp.format(n_sim, "delta") + ".npy", "wb") as f:
        numpy.save(f, delta)
    joblib.dump([aexp, length], ftemp.format(n_sim, "lengths") + ".p")

    # Try to clean up memory
    del delta
    collect()






# Set up the catalogue
paths = csiborgtools.read.CSiBORGPaths(to_new=False)
print("{}: started reading in the combined catalogue.".format(datetime.now()),
      flush=True)

cat = csiborgtools.read.CombinedHaloCatalogue(
    paths, min_m500=None, max_dist=None, verbose=False)
print("{}: finished reading in the combined catalogue with `{}`."
      .format(datetime.now(), cat.n_sims), flush=True)
matcher = csiborgtools.match.RealisationsMatcher(cat)


for i in csiborgtools.fits.split_jobs(len(cat.n_sims), nproc)[rank]:
    n = cat.n_sims[i]
    print("{}: rank {} working on simulation `{}`."
          .format(datetime.now(), rank, n), flush=True)
    out = matcher.cross_knn_position_single(
        i, nmult=15, dlogmass=2, init_dist=True, overlap=True, verbose=False,
        overlapper_kwargs={"smooth_scale": 0.5})

    # Dump the result
    with open(ftemp.format(n), "wb") as f:
        numpy.save(f, out)


comm.Barrier()
if rank == 0:
    print("Collecting files...", flush=True)

    dtype = {"names": ["match", "nsim"], "formats": [object, numpy.int32]}
    matches = numpy.full(len(cat.n_sims), numpy.nan, dtype=dtype)
    for i, n in enumerate(cat.n_sims):
        with open(ftemp.format(n), "rb") as f:
            matches["match"][i] = numpy.load(f, allow_pickle=True)
        matches["nsim"][i] = n
        remove(ftemp.format(n))

    print("Saving results to `{}`.".format(fperm))
    with open(fperm, "wb") as f:
        numpy.save(f, matches)
