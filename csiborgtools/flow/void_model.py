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
"""Function to work with the void data from Sergij & Indranil's files."""

from glob import glob
from os.path import join
from re import search

import numpy as np
from jax import numpy as jnp
from jax import vmap
from jax.scipy.ndimage import map_coordinates
from tqdm import tqdm

###############################################################################
#                            I/O of the void data                             #
###############################################################################


def load_void_data(kind):
    """
    Load the void velocities from Sergij & Indranil's files for a given kind
    of void profile per observer.

    Parameters
    ----------
    kind : str
        The kind of void profile to load. One of "exp", "gauss", "mb".

    Returns
    -------
    velocities : 3-dimensional array of shape (nLG, nrad, nphi)
    """
    if kind not in ["exp", "gauss", "mb"]:
        raise ValueError("kind must be one of 'exp', 'gauss', 'mb'")

    fdir = "/mnt/extraspace/rstiskalek/catalogs/IndranilVoid"

    kind = kind.upper()
    fdir = join(fdir, f"{kind}profile")

    files = glob(join(fdir, "*.dat"))
    rLG = [int(search(rf'v_pec_{kind}profile_rLG_(\d+)', f).group(1))
           for f in files]
    rLG = np.sort(rLG)

    for i, ri in enumerate(tqdm(rLG, desc="Loading void observer data")):
        f = join(fdir, f"v_pec_{kind}profile_rLG_{ri}.dat")
        data_i = np.genfromtxt(f).T

        if i == 0:
            data = np.full((len(rLG), *data_i.shape), np.nan, dtype=np.float32)

        data[i] = data_i

    if np.any(np.isnan(data)):
        raise ValueError("Found NaNs in loaded data.")

    return rLG, data

###############################################################################
#                      Interpolation of void velocities                       #
###############################################################################


def interpolate_void(rLG, r, phi, data, rgrid_min, rgrid_max, rLG_min, rLG_max,
                     order=1):
    """
    Interpolate the void velocities from Sergij & Indranil's files for a given
    observer over a set of radial distances and at angles specifying the
    galaxies.

    Parameters
    ----------
    rLG : float
        The observer's distance from the center of the void.
    r : 1-dimensional array
        The radial distances at which to interpolate the velocities.
    phi : 1-dimensional array
        The angles at which to interpolate the velocities, in degrees,
        defining the galaxy position.
    data : 3-dimensional array of shape (nLG, nrad, nphi)
        The void velocities for different observers, radial distances, and
        angles.
    rgrid_min, rgrid_max : float
        The minimum and maximum radial distances in the data.
    rLG_min, rLG_max : float
        The minimum and maximum observer distances in the data.
    order : int, optional
        The order of the interpolation. Default is 1, can be 0.

    Returns
    -------
    vel : 2-dimensional array of shape (len(phi), len(r))
    """
    nLG, nrad, nphi = data.shape

    # Normalize rLG to the grid scale
    rLG_normalized = (rLG - rLG_min) / (rLG_max - rLG_min) * (nLG - 1)
    rLG_normalized = jnp.repeat(rLG_normalized, r.size)
    r_normalized = (r - rgrid_min) / (rgrid_max - rgrid_min) * (nrad - 1)

    # Function to perform interpolation for a single phi
    def interpolate_single_phi(phi_val):
        # Normalize phi to match the grid
        phi_normalized = phi_val / 180 * (nphi - 1)

        # Create the grid for this specific phi
        X = jnp.vstack([rLG_normalized,
                        r_normalized,
                        jnp.repeat(phi_normalized, r.size)])

        # Interpolate over the data using map_coordinates. The mode is nearest
        # to avoid extrapolation. But values outside of the grid should never
        # occur.
        return map_coordinates(data, X, order=order, mode='nearest')

    return vmap(interpolate_single_phi)(phi)
