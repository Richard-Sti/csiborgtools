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
"""Script to help with `hmf.py`."""

import csiborgtools
import numpy as np
from tqdm import tqdm


def calculate_hmf(simname, bin_edges, halofinder="FOF", max_distance=135):
    """
    Calculate the halo mass function for a given simulation from catalogues.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(simname)
    bounds = {"dist": (0, max_distance)}

    hmf = np.full((len(nsims), len(bin_edges) - 1), np.nan)
    volume = 4 / 3 * np.pi * max_distance**3
    for i, nsim in enumerate(tqdm(nsims)):
        if "csiborg2_" in simname:
            kind = simname.split("_")[-1]
            if halofinder == "FOF":
                cat = csiborgtools.read.CSiBORG2Catalogue(
                    nsim, 99, kind, bounds=bounds)
            elif halofinder == "SUBFIND":
                cat = csiborgtools.read.CSiBORG2SUBFINDCatalogue(
                    nsim, 99, kind, kind, bounds=bounds)
            else:
                raise ValueError(f"Unknown halofinder: {halofinder}")
        else:
            raise ValueError(f"Unknown simname: {simname}")

        hmf[i] = cat.halo_mass_function(bin_edges, volume, "totmass")[1]

    return hmf
