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

# Argument parser
parser = ArgumentParser()
parser.add_argument("--nsim0", type=int)
parser.add_argument("--nsimx", type=int)
parser.add_argument("--nmult", type=float)
parser.add_argument("--sigma", type=float)
parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                    default=False)
args = parser.parse_args()
paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
smooth_kwargs = {"sigma": args.sigma, "mode": "constant", "cval": 0.0}
overlapper = csiborgtools.match.ParticleOverlap()
matcher = csiborgtools.match.RealisationsMatcher()

# Load the raw catalogues (i.e. no selection) including the initial CM
# positions and the particle archives.
cat0 = csiborgtools.read.HaloCatalogue(args.nsim0, paths, load_initial=True,
                                       rawdata=True)
catx = csiborgtools.read.HaloCatalogue(args.nsimx, paths, load_initial=True,
                                       rawdata=True)
halos0_archive = paths.initmatch_path(args.nsim0, "particles")
halosx_archive = paths.initmatch_path(args.nsimx, "particles")

# We generate the background density fields. Loads halos's particles one by one
# from the archive, concatenates them and calculates the NGP density field.
if args.verbose:
    print(f"{datetime.now()}: generating the background density fields.",
          flush=True)
delta_bckg = overlapper.make_bckg_delta(halos0_archive, verbose=args.verbose)
delta_bckg = overlapper.make_bckg_delta(halosx_archive, delta=delta_bckg,
                                        verbose=args.verbose)

# We calculate the overlap between the NGP fields.
if args.verbose:
    print(f"{datetime.now()}: crossing the simulations.", flush=True)
match_indxs, ngp_overlap = matcher.cross(cat0, catx, halos0_archive,
                                         halosx_archive, delta_bckg)

# We now smoothen up the background density field for the smoothed overlap
# calculation.
if args.verbose:
    print(f"{datetime.now()}: smoothing the background field.", flush=True)
gaussian_filter(delta_bckg, output=delta_bckg, **smooth_kwargs)

# We calculate the smoothed overlap for the pairs whose NGP overlap is > 0.
if args.verbose:
    print(f"{datetime.now()}: calculating smoothed overlaps.", flush=True)
smoothed_overlap = matcher.smoothed_cross(cat0, catx, halos0_archive,
                                          halosx_archive, delta_bckg,
                                          match_indxs, smooth_kwargs)

# We save the results at long last.
fout = paths.overlap_path(args.nsim0, args.nsimx)
if args.verbose:
    print(f"{datetime.now()}: saving results to `{fout}`.", flush=True)
numpy.savez(fout, match_indxs=match_indxs, ngp_overlap=ngp_overlap,
            smoothed_overlap=smoothed_overlap, sigma=args.sigma)
print(f"{datetime.now()}: all finished.", flush=True)
