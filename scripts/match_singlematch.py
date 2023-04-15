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
"""A script to calculate overlap between two CSiBORG realisations."""
from os.path import join
from argparse import ArgumentParser
from datetime import datetime
import numpy
from scipy.ndimage import gaussian_filter
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools
import utils

# Argument parser
parser = ArgumentParser()
parser.add_argument("--nsim0", type=int)
parser.add_argument("--nsimx", type=int)
parser.add_argument("--nmult", type=float)
parser.add_argument("--sigma", type=float)
args = parser.parse_args()

# File paths
paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
fout = join(utils.dumpdir, "overlap",
            "cross_{}_{}.npz".format(args.nsim0, args.nsimx))
smooth_kwargs = {"sigma": args.sigma, "mode": "constant", "cval": 0.0}
overlapper = csiborgtools.match.ParticleOverlap()

# Load catalogues
print("{}: loading catalogues {} and {}."
      .format(datetime.now(), args.nsim0, args.nsimx), flush=True)
cat0 = csiborgtools.read.ClumpsCatalogue(args.nsim0, paths)
catx = csiborgtools.read.ClumpsCatalogue(args.nsimx, paths)


print("{}: loading simulation {} and converting positions to cell numbers."
      .format(datetime.now(), args.nsim0), flush=True)

with open(paths.initmatch_path(args.nsim0, "particles"), "rb") as f:
    clumps0 = numpy.load(f, allow_pickle=True)
    overlapper.clumps_pos2cell(clumps0)
print("{}: loading simulation {} and converting positions to cell numbers."
      .format(datetime.now(), args.nsimx), flush=True)
with open(paths.initmatch_path(args.nsimx, "particles"), 'rb') as f:
    clumpsx = numpy.load(f, allow_pickle=True)
    overlapper.clumps_pos2cell(clumpsx)


print("{}: generating the background density fields.".format(datetime.now()),
      flush=True)
delta_bckg = overlapper.make_bckg_delta(clumps0)
delta_bckg = overlapper.make_bckg_delta(clumpsx, delta=delta_bckg)


print("{}: crossing the simulations.".format(datetime.now()), flush=True)
matcher = csiborgtools.match.RealisationsMatcher()
ref_indxs, cross_indxs, match_indxs, ngp_overlap = matcher.cross(
    cat0, catx, clumps0, clumpsx, delta_bckg)


print("{}: smoothing the background field.".format(datetime.now()), flush=True)
gaussian_filter(delta_bckg, output=delta_bckg, **smooth_kwargs)


print("{}: calculating smoothed overlaps.".format(datetime.now()), flush=True)
smoothed_overlap = matcher.smoothed_cross(clumps0, clumpsx, delta_bckg,
                                          ref_indxs, cross_indxs, match_indxs,
                                          smooth_kwargs)

# Dump the result
print("Saving results to `{}`.".format(fout), flush=True)
with open(fout, "wb") as f:
    numpy.savez(fout, ref_indxs=ref_indxs, cross_indxs=cross_indxs,
                match_indxs=match_indxs, ngp_overlap=ngp_overlap,
                smoothed_overlap=smoothed_overlap, sigma=args.sigma)
print("All finished.", flush=True)
