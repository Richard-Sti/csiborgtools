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

TODO
----
    - [ ] Update this script
"""
import numpy
from datetime import datetime
from mpi4py import MPI
from os.path import join
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
fperm = join(utils.dumpdir, "overlap", "cross_{}.npy")
# fperm = join(utils.dumpdir, "match", "cross_matches.npy")

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
        i, nmult=15, dlogmass=2, init_dist=True, overlap=False, verbose=False,
        overlapper_kwargs={"smooth_scale": 1})

    # Dump the result
    fout = fperm.format(n)
    print("Saving results to `{}`.".format(fout))
    with open(fout, "wb") as f:
        numpy.save(fout, out)


comm.Barrier()
if rank == 0:
    print("All finished.")
