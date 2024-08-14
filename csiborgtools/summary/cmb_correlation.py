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
"""
Useful functions for getting summary data for CMB x MATTER cross-correlation.
"""
import numpy as np
from h5py import File
from healpy.sphtfunc import smoothing
import healpy as hp
from tqdm import tqdm


def read_projected_matter(simname, paths, fwhm_deg=None, remove_monopole=False,
                          remove_dipole=False, normalize=False):
    """
    Read the projected matter density field for a given simulation.

    Parameters
    ----------
    simname : str
        The name of the simulation.
    paths : csiborgtools.read.Paths
        Paths object.
    fwhm_deg : float, optional
        The full-width at half-maximum of the smoothing kernel in degrees.
    remove_monopole : bool, optional
        Whether to remove the monopole from the field.
    remove_dipole : bool, optional
        Whether to remove the dipole from the field.
    normalize : bool, optional
        Whether to apply standard normalization to the field.

    Returns
    -------
    dist_ranges : 2-dimensional array of shape (n_dist_ranges, 2)
        The distance ranges for the field.
    data : 3-dimensional array of shape (n_sims, n_dist_ranges, npix)
        The projected matter density field.
    """
    kind = "density"
    nsims = paths.get_ics(simname)

    fname = paths.field_projected(simname, kind)
    with File(fname, "r") as f:
        dist_ranges = f["dist_ranges"][...]

        npix = len(f[f"nsim_{nsims[0]}/dist_range_0"])

        data = np.zeros((len(nsims), len(dist_ranges), npix))
        for i, nsim in enumerate(tqdm(nsims, desc="Simulations")):
            for j in range(len(dist_ranges)):
                skymap = f[f"nsim_{nsim}/dist_range_{j}"][...]

                if fwhm_deg is not None:
                    skymap = smoothing(skymap, fwhm=fwhm_deg * np.pi / 180.0)

                if remove_monopole:
                    hp.pixelfunc.remove_monopole(skymap, copy=False)
                if remove_dipole:
                    hp.pixelfunc.remove_dipole(skymap, copy=False)

                if normalize:
                    skymap -= np.mean(skymap)
                    skymap /= np.std(skymap)

                data[i, j] = skymap

    return dist_ranges, data
