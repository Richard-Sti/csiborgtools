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
from cache_to_disk import cache_to_disk
from os.path import join


RANDOM_MAH_Sorce_Virgo_UPPER = np.array(
    [[2.18554217, 0.16246594],
     [2.93253012, 0.17284951],
     [3.2939759, 0.34169001],
     [3.75180723, 0.42006683],
     [4.28192771, 0.44691426],
     [4.61927711, 0.53819753],
     [5.34216867, 0.58454257],
     [5.89638554, 0.68954882],
     [6.23373494, 0.73361948],
     [6.45060241, 0.81341823],
     [7.05301205, 0.92071572],
     [7.82409639, 0.92071572],
     [8.28192771, 0.95953933],
     [8.61927711, 0.97956078],
     [9.70361446, 1.],
     [11.17349398, 1.],
     [13.07710843, 1.],
     [13.82409639, 1.]]
    )

RANDOM_MAH_SORCE_Virgo_LOWER = np.array(
    [[3.36626506e+00, 1.00000000e-02],
     [3.75180723e+00, 1.10877404e-02],
     [3.99277108e+00, 1.04216677e-02],
     [4.30602410e+00, 1.15552746e-02],
     [4.61927711e+00, 1.67577322e-02],
     [4.98072289e+00, 2.14703224e-02],
     [5.39036145e+00, 3.82789169e-02],
     [5.89638554e+00, 5.00670000e-02],
     [6.30602410e+00, 5.11116827e-02],
     [7.29397590e+00, 5.32668971e-02],
     [7.77590361e+00, 5.55129899e-02],
     [8.11325301e+00, 6.68516464e-02],
     [8.57108434e+00, 8.56515893e-02],
     [9.60722892e+00, 1.32152759e-01],
     [1.04265060e+01, 1.46527548e-01],
     [1.07638554e+01, 1.49584947e-01],
     [1.11493976e+01, 1.72849513e-01],
     [1.18240964e+01, 2.16931625e-01],
     [1.21855422e+01, 2.45546942e-01],
     [1.25951807e+01, 3.48819614e-01],
     [1.30771084e+01, 5.27197199e-01],
     [1.36795181e+01, 8.83462949e-01],
     [1.38000000e+01, 1.00000000e+00]]
    )


def t():
    return datetime.now()


@cache_to_disk(90)
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
                            min_age=0, include_nsim0=True):
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

    mask = age > min_age
    return age[mask], np.vstack(X)[:, mask]


def extract_mah(simname, logmass_bounds, key, min_age=0):
    """
    Extract the random MAHs for a given simulation and mass range and key.
    Keys are for example: "MainProgenitorMass" or "GroupMass"
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(simname)

    X = []
    for i, nsim in enumerate(nsims):
        with File(paths.random_mah(simname, nsim), 'r') as f:
            mah = f[key][:]
            final_mass = mah[:, -1]

            # Select the mass range
            mask = final_mass >= 10**logmass_bounds[0]
            mask &= final_mass < 10**logmass_bounds[1]

            X.append(mah[mask])

            if i == 0:
                redshift = f["Redshift"][:]

    X = np.vstack(X)

    cosmo = FlatLambdaCDM(H0=67.76, Om0=csiborgtools.simname2Omega_m(simname))
    age = cosmo.age(redshift).value

    mask = age > min_age
    return age[mask], X[:, mask]


def extract_mah_mdpl2(logmass_bounds, min_age=1.5):
    """
    MAH extraction for the MDPL2 simulation. Data comes from
    `https://arxiv.org/abs/2105.05859`
    """
    fdir = "/mnt/extraspace/rstiskalek/catalogs/"

    age = np.genfromtxt(join(fdir, "mdpl2_cosmic_time.txt"))
    with File(join(fdir, "diffmah_mdpl2.h5"), 'r') as f:
        log_mp = f["logmp_sim"][:]
        log_mah_sim = f["log_mah_sim"][...]

    xmin, xmax = logmass_bounds
    ks = np.where((log_mp > xmin) & (log_mp < xmax))[0]
    X = 10**log_mah_sim[ks]

    mask = age > min_age
    return age[mask], X[:, mask]
