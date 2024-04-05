# Copyright (C) 2024 Richard Stiskalek
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
"""Script to help with `mah.py`."""
from datetime import datetime

import csiborgtools
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from h5py import File
from tqdm import tqdm, trange


def t():
    return datetime.now()


def load_data(nsim0, simname, min_logmass):
    """
    Load the reference catalogue, the cross catalogues, the merger trees and
    the overlap reader (in this order).
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(simname)
    if "csiborg2_" in simname:
        kind = simname.split("_")[-1]
        print(f"{t()}: loading {len(nsims)} halo catalogues.")
        cat0 = csiborgtools.read.CSiBORG2Catalogue(nsim0, 99, kind)
        catxs = [csiborgtools.read.CSiBORG2Catalogue(n, 99, kind)
                 for n in nsims if n != nsim0]

        print(f"{t()}: loading {len(nsims)} merger trees.")
        merger_trees = {}
        for nsim in tqdm(nsims):
            merger_trees[nsim] = csiborgtools.read.CSiBORG2MergerTreeReader(
                nsim, kind)
    else:
        raise ValueError(f"Unknown simname: {simname}")

    overlaps = csiborgtools.summary.NPairsOverlap(cat0, catxs, min_logmass)

    return cat0, catxs, merger_trees, overlaps


def extract_main_progenitor_maxoverlap(group_nr, overlaps, merger_trees):
    """
    Follow the main progenitor of a reference group and its maximum overlap
    group in the cross catalogues.
    """
    min_overlap = 0

    # NOTE these can be all cached in the overlap object.
    max_overlaps = overlaps.max_overlap(0, True)[group_nr]
    if np.sum(max_overlaps > 0) == 0:
        raise ValueError(f"No overlaps for group {group_nr}.")

    max_overlap_indxs = overlaps.max_overlap_key(
        "index", min_overlap, True)[group_nr]

    out = {}
    for i in trange(len(overlaps), desc="Cross main progenitors"):
        nsimx = overlaps[i].catx().nsim
        group_nr_cross = max_overlap_indxs[i]

        if np.isnan(group_nr_cross):
            continue

        x = merger_trees[nsimx].main_progenitor(int(group_nr_cross))
        x["Overlap"] = max_overlaps[i]

        out[nsimx] = x

    nsim0 = overlaps.cat0().nsim
    print(f"Appending main progenitor for {nsim0}.")
    out[nsim0] = merger_trees[nsim0].main_progenitor(group_nr)

    return out


def summarize_extracted_mah(simname, data, nsim0, nsimxs, key,
                            include_nsim0=True):
    """
    Turn the dictionaries of extracted MAHs into a single array.
    """
    if "csiborg2_" in simname:
        nsnap = 100
    else:
        raise ValueError(f"Unknown simname: {simname}")

    X = []
    for nsimx in nsimxs + [nsim0] if include_nsim0 else nsimxs:
        try:
            d = data[nsimx]
        except KeyError:
            continue

        x = np.full(nsnap, np.nan, dtype=np.float32)
        x[d["SnapNum"]] = d[key]

        X.append(x)

    cosmo = FlatLambdaCDM(H0=67.76, Om0=csiborgtools.simname2Omega_m(simname))
    zs = [csiborgtools.snap2redshift(i, simname) for i in range(nsnap)]
    age = cosmo.age(zs).value

    return age, np.vstack(X)


def extract_random_mah(simname, logmass_bounds, key):
    """
    Extract the random MAHs for a given simulation and mass range and key.
    Keys are for example: "MainProgenitorMass" or "GroupMass"
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(simname)

    X = []
    for i, nsim in enumerate(nsims):
        with File(paths.random_mah(simname, nsim), 'r') as f:
            final_mass = f["FinalGroupMass"][:]

            # Select the mass range
            mask = final_mass >= 10**logmass_bounds[0]
            mask &= final_mass < 10**logmass_bounds[1]

            X.append(f[key][:][mask])

            if i == 0:
                redshift = f["Redshift"][:]

    X = np.vstack(X)

    cosmo = FlatLambdaCDM(H0=67.76, Om0=csiborgtools.simname2Omega_m(simname))
    age = cosmo.age(redshift).value

    return age, X
