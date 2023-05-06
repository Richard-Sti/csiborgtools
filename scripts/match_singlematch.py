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
from argparse import ArgumentParser
from datetime import datetime
from distutils.util import strtobool

import numpy
from scipy.ndimage import gaussian_filter

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools
    from csiborgtools.read import HaloCatalogue, read_h5

# Argument parser
parser = ArgumentParser()
parser.add_argument("--nsim0", type=int)
parser.add_argument("--nsimx", type=int)
parser.add_argument("--nmult", type=float)
parser.add_argument("--sigma", type=float, default=None)
parser.add_argument("--smoothen", type=lambda x: bool(strtobool(x)),
                    default=None)
parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                    default=False)
args = parser.parse_args()
paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
smooth_kwargs = {"sigma": args.sigma, "mode": "constant", "cval": 0.0}
overlapper = csiborgtools.match.ParticleOverlap()
matcher = csiborgtools.match.RealisationsMatcher()

# Load the raw catalogues (i.e. no selection) including the initial CM
# positions and the particle archives.
cat0 = HaloCatalogue(args.nsim0, paths, load_initial=True,
                     minmass=("totpartmass", 1e14), with_lagpatch=True)
catx = HaloCatalogue(args.nsimx, paths, load_initial=True,
                     minmass=("totpartmass", 1e14), with_lagpatch=True)

clumpmap0 = read_h5(paths.particles_path(args.nsim0))["clumpmap"]
parts0 = read_h5(paths.initmatch_path(args.nsim0, "particles"))["particles"]
clid2map0 = {clid: i for i, clid in enumerate(clumpmap0[:, 0])}

clumpmapx = read_h5(paths.particles_path(args.nsimx))["clumpmap"]
partsx = read_h5(paths.initmatch_path(args.nsimx, "particles"))["particles"]
clid2mapx = {clid: i for i, clid in enumerate(clumpmapx[:, 0])}


# We generate the background density fields. Loads halos's particles one by one
# from the archive, concatenates them and calculates the NGP density field.
if args.verbose:
    print(f"{datetime.now()}: generating the background density fields.",
          flush=True)
delta_bckg = overlapper.make_bckg_delta(parts0, clumpmap0[clid2map0[0], 2] + 1)
delta_bckg = overlapper.make_bckg_delta(partsx, clumpmapx[clid2mapx[0], 2] + 1,
                                        delta=delta_bckg)

# We calculate the overlap between the NGP fields.
if args.verbose:
    print(f"{datetime.now()}: crossing the simulations.", flush=True)
match_indxs, ngp_overlap = matcher.cross(cat0, catx, parts0, partsx, clumpmap0,
                                         clumpmapx, delta_bckg,
                                         verbose=args.verbose)
# We wish to store the halo IDs of the matches, not their array positions in
# the catalogues
match_hids = numpy.copy(match_indxs)
for i, matches in enumerate(match_indxs):
    for j, match in enumerate(matches):
        match_hids[i][j] = catx["index"][match]

fout = paths.overlap_path(args.nsim0, args.nsimx, smoothed=False)
numpy.savez(fout, ref_hids=cat0["index"], match_hids=match_hids,
            ngp_overlap=ngp_overlap)
if args.verbose:
    print(f"{datetime.now()}: calculated NGP overlap. Output saved to {fout}.",
          flush=True)

if not args.smoothen:
    quit()

# We now smoothen up the background density field for the smoothed overlap
# calculation.
if args.verbose:
    print(f"{datetime.now()}: smoothing the background field.", flush=True)
gaussian_filter(delta_bckg, output=delta_bckg, **smooth_kwargs)

# We calculate the smoothed overlap for the pairs whose NGP overlap is > 0.
if args.verbose:
    print(f"{datetime.now()}: calculating smoothed overlaps.", flush=True)
smoothed_overlap = matcher.smoothed_cross(cat0, catx, parts0, partsx,
                                          clumpmap0, clumpmapx, delta_bckg,
                                          match_indxs, smooth_kwargs)

fout = paths.overlap_path(args.nsim0, args.nsimx, smoothed=True)
numpy.savez(fout, smoothed_overlap=smoothed_overlap, sigma=args.sigma)
if args.verbose:
    print(f"{datetime.now()}: calculated NGP overlap. Output saved to {fout}.",
          flush=True)
