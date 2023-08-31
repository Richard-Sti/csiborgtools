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
A script to calculate overlap between two IC realisations of the same
simulation.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from distutils.util import strtobool

import numpy
from scipy.ndimage import gaussian_filter

import csiborgtools


def pair_match_max(nsim0, nsimx, simname, min_logmass, mult, verbose):
    """
    Match a pair of simulations using the Max method.

    Parameters
    ----------
    nsim0, nsimx : int
        The reference and cross simulation IC index.
    simname : str
        Simulation name.
    min_logmass : float
        Minimum log halo mass.
    mult : float
        Multiplicative factor for search radius.
    verbose : bool
        Verbosity flag.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    if simname == "csiborg":
        mass_kind = "fof_totpartmass"
        maxdist = 155
        periodic = False
        bounds = {"dist": (0, maxdist), mass_kind: (10**min_logmass, None)}
        cat0 = csiborgtools.read.CSiBORGHaloCatalogue(
            nsim0, paths, bounds=bounds, load_fitted=True, load_initial=False)
        catx = csiborgtools.read.CSiBORGHaloCatalogue(
            nsimx, paths, bounds=bounds, load_fitted=True, load_initial=False)
    elif simname == "quijote":
        mass_kind = "group_mass"
        maxdist = None
        periodic = True
        bounds = {mass_kind: (10**min_logmass, None)}
        cat0 = csiborgtools.read.QuijoteHaloCatalogue(
            nsim0, paths, 4, bounds=bounds, load_fitted=True,
            load_initial=False)
        catx = csiborgtools.read.QuijoteHaloCatalogue(
            nsimx, paths, 4, bounds=bounds, load_fitted=True,
            load_initial=False)
    else:
        raise ValueError(f"Unknown simulation `{simname}`.")

    reader = csiborgtools.summary.PairOverlap(cat0, catx, paths, min_logmass,
                                           maxdist=maxdist)
    out = csiborgtools.match.matching_max(
        cat0, catx, mass_kind, mult=mult, periodic=periodic,
        overlap=reader.overlap(from_smoothed=True),
        match_indxs=reader["match_indxs"], verbose=verbose)

    fout = paths.match_max(simname, nsim0, nsimx, min_logmass, mult)
    if verbose:
        print(f"{datetime.now()}: saving to ... `{fout}`.", flush=True)
    numpy.savez(fout, **{p: out[p] for p in out.dtype.names})


def pair_match(nsim0, nsimx, simname, min_logmass, sigma, verbose):
    """
    Calculate overlaps between two simulations.

    Parameters
    ----------
    nsim0 : int
        The reference simulation IC index.
    nsimx : int
        The cross simulation IC index.
    simname : str
        Simulation name.
    min_logmass : float
        Minimum log halo mass.
    sigma : float
        Smoothing scale in number of grid cells.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    None
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    smooth_kwargs = {"sigma": sigma, "mode": "constant", "cval": 0}

    if simname == "csiborg":
        overlapper_kwargs = {"box_size": 2048, "bckg_halfsize": 512}
        mass_kind = "fof_totpartmass"
        bounds = {"dist": (0, 155), mass_kind: (10**min_logmass, None)}

        cat0 = csiborgtools.read.CSiBORGHaloCatalogue(
            nsim0, paths, bounds=bounds, load_fitted=False,
            with_lagpatch=True)
        catx = csiborgtools.read.CSiBORGHaloCatalogue(
            nsimx, paths, bounds=bounds, load_fitted=False,
            with_lagpatch=True)
    elif simname == "quijote":
        overlapper_kwargs = {"box_size": 512, "bckg_halfsize": 256}
        mass_kind = "group_mass"
        bounds = {mass_kind: (10**min_logmass, None)}

        cat0 = csiborgtools.read.QuijoteHaloCatalogue(
            nsim0, paths, 4, bounds=bounds, load_fitted=False,
            with_lagpatch=True)
        catx = csiborgtools.read.QuijoteHaloCatalogue(
            nsimx, paths, 4, bounds=bounds, load_fitted=False,
            with_lagpatch=True)
    else:
        raise ValueError(f"Unknown simulation name: `{simname}`.")

    halomap0 = csiborgtools.read.read_h5(
        paths.particles(nsim0, simname))["halomap"]
    parts0 = csiborgtools.read.read_h5(
        paths.initmatch(nsim0, simname, "particles"))["particles"]
    hid2map0 = {hid: i for i, hid in enumerate(halomap0[:, 0])}

    halomapx = csiborgtools.read.read_h5(
        paths.particles(nsimx, simname))["halomap"]
    partsx = csiborgtools.read.read_h5(
        paths.initmatch(nsimx, simname, "particles"))["particles"]
    hid2mapx = {hid: i for i, hid in enumerate(halomapx[:, 0])}

    overlapper = csiborgtools.match.ParticleOverlap(**overlapper_kwargs)
    delta_bckg = overlapper.make_bckg_delta(parts0, halomap0, hid2map0, cat0,
                                            verbose=verbose)
    delta_bckg = overlapper.make_bckg_delta(partsx, halomapx, hid2mapx, catx,
                                            delta=delta_bckg, verbose=verbose)

    matcher = csiborgtools.match.RealisationsMatcher(
        mass_kind=mass_kind, **overlapper_kwargs)
    match_indxs, ngp_overlap = matcher.cross(cat0, catx, parts0, partsx,
                                             halomap0, halomapx, delta_bckg,
                                             verbose=verbose)

    # We want to store the halo IDs of the matches, not their array positions
    # in the catalogues.
    match_hids = deepcopy(match_indxs)
    for i, matches in enumerate(match_indxs):
        for j, match in enumerate(matches):
            match_hids[i][j] = catx["index"][match]

    fout = paths.overlap(simname, nsim0, nsimx, min_logmass, smoothed=False)
    if verbose:
        print(f"{datetime.now()}: saving to ... `{fout}`.", flush=True)
    numpy.savez(fout, ref_hids=cat0["index"], match_hids=match_hids,
                ngp_overlap=ngp_overlap)

    if not sigma > 0:
        return

    if verbose:
        print(f"{datetime.now()}: smoothing the background field.", flush=True)
    gaussian_filter(delta_bckg, output=delta_bckg, **smooth_kwargs)

    # We calculate the smoothed overlap for the pairs whose NGP overlap is > 0.
    smoothed_overlap = matcher.smoothed_cross(cat0, catx, parts0, partsx,
                                              halomap0, halomapx, delta_bckg,
                                              match_indxs, smooth_kwargs,
                                              verbose=verbose)

    fout = paths.overlap(simname, nsim0, nsimx, min_logmass, smoothed=True)
    if verbose:
        print(f"{datetime.now()}: saving to ... `{fout}`.", flush=True)
    numpy.savez(fout, smoothed_overlap=smoothed_overlap, sigma=sigma)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--kind", type=str, required=True,
                        choices=["overlap", "max"], help="Kind of matching.")
    parser.add_argument("--nsim0", type=int, required=True,
                        help="Reference simulation IC index.")
    parser.add_argument("--nsimx", type=int, required=True,
                        help="Cross simulation IC index.")
    parser.add_argument("--simname", type=str, required=True,
                        help="Simulation name.")
    parser.add_argument("--min_logmass", type=float, required=True,
                        help="Minimum log halo mass.")
    parser.add_argument("--mult", type=float, default=5,
                        help="Search radius multiplier for Max's method.")
    parser.add_argument("--sigma", type=float, default=0,
                        help="Smoothing scale in number of grid cells.")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        default=False, help="Verbosity flag.")
    args = parser.parse_args()

    if args.kind == "overlap":
        pair_match(args.nsim0, args.nsimx, args.simname, args.min_logmass,
                   args.sigma, args.verbose)
    elif args.kind == "max":
        pair_match_max(args.nsim0, args.nsimx, args.simname, args.min_logmass,
                       args.mult, args.verbose)
    else:
        raise ValueError(f"Unknown matching kind: `{args.kind}`.")
