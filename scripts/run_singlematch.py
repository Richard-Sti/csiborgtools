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
Script to test running the CSiBORG realisations matcher.
"""
import numpy
from datetime import datetime
from os.path import join
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools
import utils

# File paths
fperm = join(utils.dumpdir, "overlap", "cross_{}.npy")
nmult = 1.0
overlap = True
select_initial = True
fast_neighbours = False


paths = csiborgtools.read.CSiBORGPaths(to_new=False)
ic = 7468
paths.set_info(ic, paths.get_maximum_snapshot(ic))

print("{}: loading catalogues.".format(datetime.now()), flush=True)
cat = csiborgtools.read.CombinedHaloCatalogue(paths)


matcher = csiborgtools.match.RealisationsMatcher(cat)
nsim0 = cat.n_sims[0]
nsimx = cat.n_sims[1]

print("{}: crossing the simulations.".format(datetime.now()), flush=True)

out = matcher.cross_knn_position_single(
    0, nmult=nmult, dlogmass=2., overlap=overlap,
    select_initial=select_initial, fast_neighbours=fast_neighbours)

# Dump the result
fout = fperm.format(nsim0)
print("Saving results to `{}`.".format(fout), flush=True)
with open(fout, "wb") as f:
    numpy.save(fout, out)

print("All finished.", flush=True)
