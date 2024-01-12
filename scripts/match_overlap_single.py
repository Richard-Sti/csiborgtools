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
    if simname == "csiborg1":
        maxdist = 155
        periodic = False
        bounds = {"dist": (0, maxdist), "totmass": (10**min_logmass, None)}
        cat0 = csiborgtools.read.CSiBORG1Catalogue(nsim0, bounds=bounds)
        catx = csiborgtools.read.CSiBORG1Catalogue(nsimx, bounds=bounds)
    elif "csiborg2" in simname:
        raise RuntimeError("CSiBORG2 currently not implemented..")
    elif simname == "quijote":
        maxdist = None
        periodic = True
        bounds = {"totmass": (10**min_logmass, None)}
        cat0 = csiborgtools.read.QuijoteCatalogue(nsim0, bounds=bounds)
        catx = csiborgtools.read.QuijoteHaloCatalogue(nsimx, bounds=bounds)
    else:
        raise ValueError(f"Unknown simulation `{simname}`.")

    reader = csiborgtools.summary.PairOverlap(cat0, catx, min_logmass, maxdist)
    out = csiborgtools.match.matching_max(
        cat0, catx, "totmass", mult=mult, periodic=periodic,
        overlap=reader.overlap(from_smoothed=True),
        match_indxs=reader["match_indxs"], verbose=verbose)

    fout = cat0.paths.match_max(simname, nsim0, nsimx, min_logmass, mult)
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
    bounds = {"lagpatch_size": (0, None)}

    if simname == "csiborg1":
        overlapper_kwargs = {"box_size": 2048, "bckg_halfsize": 512}
        bounds |= {"dist": (0, 150), "totmass": (10**min_logmass, None)}

        snap0 = csiborgtools.read.CSiBORG1Snapshot(
            nsim0, 0, keep_snapshot_open=True)
        cat0 = csiborgtools.read.CSiBORG1Catalogue(nsim0, snapshot=snap0,
                                                   bounds=bounds)

        snapx = csiborgtools.read.CSiBORG1Snapshot(
            nsimx, 0, keep_snapshot_open=True)
        catx = csiborgtools.read.CSiBORGCatalogue(nsimx, snapshot=snapx,
                                                  bounds=bounds)
    elif "csiborg2" in simname:
        raise RuntimeError("CSiBORG2 currently not implemented..")
    elif simname == "quijote":
        overlapper_kwargs = {"box_size": 512, "bckg_halfsize": 256}
        bounds |= {"totmass": (10**min_logmass, None)}

        snap0 = csiborgtools.read.QuijoteSnapshot(
            nsim0, "ICs", keep_snapshot_open=True)
        cat0 = csiborgtools.read.QuijoteCatalogue(nsim0, snapshot=snap0,
                                                  bounds=bounds)

        snapx = csiborgtools.read.QuijoteSnapshot(nsimx, "ICs",
                                                  keep_snapshot_open=True)
        catx = csiborgtools.read.QuijoteCatalogue(nsimx, snapshot=snapx,
                                                  bounds=bounds)
    else:
        raise ValueError(f"Unknown simulation name: `{simname}`.")

    overlapper = csiborgtools.match.ParticleOverlap(**overlapper_kwargs)
    delta_bckg = overlapper.make_bckg_delta(cat0, verbose=verbose)
    delta_bckg = overlapper.make_bckg_delta(catx, delta=delta_bckg,
                                            verbose=verbose)

    matcher = csiborgtools.match.RealisationsMatcher(**overlapper_kwargs)
    match_indxs, ngp_overlap = matcher.cross(cat0, catx, delta_bckg,
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
    smoothed_overlap = matcher.smoothed_cross(cat0, catx, delta_bckg,
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
