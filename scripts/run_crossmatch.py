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

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

# File paths
ftemp = join(utils.dumpdir, "temp_match", "match_{}.npy")
fperm = join(utils.dumpdir, "match", "cross_matches.npy")

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
    # NOTE turn on overlap later
    out = matcher.cross_knn_position_single(
        i, nmult=15, dlogmass=2, init_dist=True, overlap=True, verbose=False)

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
