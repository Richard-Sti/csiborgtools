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
from copy import deepcopy
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


def pair_match(nsim0, nsimx, sigma, smoothen, verbose):
    from csiborgtools.read import CSiBORGHaloCatalogue, read_h5

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    smooth_kwargs = {"sigma": sigma, "mode": "constant", "cval": 0.0}
    overlapper = csiborgtools.match.ParticleOverlap()
    matcher = csiborgtools.match.RealisationsMatcher()

    # Load the raw catalogues (i.e. no selection) including the initial CM
    # positions and the particle archives.
    bounds = {"totpartmass": (1e12, None)}
    cat0 = CSiBORGHaloCatalogue(nsim0, paths, load_initial=True, bounds=bounds,
                                with_lagpatch=True, load_clumps_cat=True)
    catx = CSiBORGHaloCatalogue(nsimx, paths, load_initial=True, bounds=bounds,
                                with_lagpatch=True, load_clumps_cat=True)

    clumpmap0 = read_h5(paths.particles(nsim0))["clumpmap"]
    parts0 = read_h5(paths.initmatch(nsim0, "particles"))["particles"]
    clid2map0 = {clid: i for i, clid in enumerate(clumpmap0[:, 0])}

    clumpmapx = read_h5(paths.particles(nsimx))["clumpmap"]
    partsx = read_h5(paths.initmatch(nsimx, "particles"))["particles"]
    clid2mapx = {clid: i for i, clid in enumerate(clumpmapx[:, 0])}

    # We generate the background density fields. Loads halos's particles one by
    # one from the archive, concatenates them and calculates the NGP density
    # field.
    if verbose:
        print(f"{datetime.now()}: generating the background density fields.",
              flush=True)
    delta_bckg = overlapper.make_bckg_delta(parts0, clumpmap0, clid2map0, cat0,
                                            verbose=verbose)
    delta_bckg = overlapper.make_bckg_delta(partsx, clumpmapx, clid2mapx, catx,
                                            delta=delta_bckg, verbose=verbose)

    # We calculate the overlap between the NGP fields.
    if verbose:
        print(f"{datetime.now()}: crossing the simulations.", flush=True)
    match_indxs, ngp_overlap = matcher.cross(cat0, catx, parts0, partsx,
                                             clumpmap0, clumpmapx, delta_bckg,
                                             verbose=verbose)
    # We wish to store the halo IDs of the matches, not their array positions
    # in the catalogues
    match_hids = deepcopy(match_indxs)
    for i, matches in enumerate(match_indxs):
        for j, match in enumerate(matches):
            match_hids[i][j] = catx["index"][match]

    fout = paths.overlap(nsim0, nsimx, smoothed=False)
    numpy.savez(fout, ref_hids=cat0["index"], match_hids=match_hids,
                ngp_overlap=ngp_overlap)
    if verbose:
        print(f"{datetime.now()}: calculated NGP overlap, saved to {fout}.",
              flush=True)

    if not smoothen:
        quit()

    # We now smoothen up the background density field for the smoothed overlap
    # calculation.
    if verbose:
        print(f"{datetime.now()}: smoothing the background field.", flush=True)
    gaussian_filter(delta_bckg, output=delta_bckg, **smooth_kwargs)

    # We calculate the smoothed overlap for the pairs whose NGP overlap is > 0.
    smoothed_overlap = matcher.smoothed_cross(cat0, catx, parts0, partsx,
                                              clumpmap0, clumpmapx, delta_bckg,
                                              match_indxs, smooth_kwargs,
                                              verbose=verbose)

    fout = paths.overlap(nsim0, nsimx, smoothed=True)
    numpy.savez(fout, smoothed_overlap=smoothed_overlap, sigma=sigma)
    if verbose:
        print(f"{datetime.now()}: calculated smoothing, saved to {fout}.",
              flush=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nsim0", type=int)
    parser.add_argument("--nsimx", type=int)
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--smoothen", type=lambda x: bool(strtobool(x)),
                        default=None)
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        default=False)
    args = parser.parse_args()

    pair_match(args.nsim0, args.nsimx, args.sigma, args.smoothen, args.verbose)
