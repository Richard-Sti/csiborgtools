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
"""Functions to work with the void data from Sergij & Indranil's files."""

from glob import glob
from os.path import exists, join
from re import search

import numpy as np
from astropy.coordinates import SkyCoord, angular_separation
from h5py import File
from jax import numpy as jnp
from jax import vmap
from jax.scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates as map_coordinates_np
from tqdm import tqdm

from ..params import SPEED_OF_LIGHT
from ..utils import fprint, galactic_to_radec
from .cosmography import distmod2dist, distmod2redshift

###############################################################################
#                         Basic void computations                             #
###############################################################################


def angular_distance_from_void_axis(RA, dec):
    """
    Calculate the angular distance of a galaxy from the void axis, all in
    degrees.
    """
    # Calculate the separation angle between the galaxy and the model axis.
    model_axis = SkyCoord(l=117, b=4, frame='galactic', unit='deg').icrs
    coords = SkyCoord(ra=RA, dec=dec, unit='deg').icrs
    return angular_separation(
        coords.ra.rad, coords.dec.rad,
        model_axis.ra.rad, model_axis.dec.rad) * 180 / np.pi


def select_void_h(kind):
    """Select 'little h' for void profile `kind`."""
    hs = {"mb": 0.7615, "gauss": 0.7724, "exp": 0.7725}
    try:
        return hs[kind]
    except KeyError:
        raise ValueError(f"Unknown void kind: `{kind}`.")


def select_void__fiducial_size(kind):
    """Select the fiducial void size in cMpc / h for a void profile `kind`."""
    rvoid = {"mb": 228.2, "gauss": 1030, "exp": 1030}
    try:
        return rvoid[kind] * select_void_h(kind)
    except KeyError:
        raise ValueError(f"Unknown void kind: `{kind}`.")


def select_vvoid(kind):
    """Select the fidicual void velocity for a void profile `kind`."""
    vvoid = {"exp": 2307, "gauss": 2018, "mb": 1586}
    try:
        return vvoid[kind]
    except KeyError:
        raise ValueError(f"Unknown void kind: `{kind}`.")


###############################################################################
#                            I/O of the void data                             #
###############################################################################


def load_void_fiducial(profile, kind, try_load_from_hdf5=True,
                       dump_to_hdf5=True):
    """
    Load the void velocities from Sergij & Indranil's files for a given kind
    of void profile per observer.

    Parameters
    ----------
    profile : str
        Void profile to load. One of "exp", "gauss", "mb".
    kind : str
        Data kind. One of "density", "vrad", "vx", or "vy".
    try_load_from_hdf5 : bool, optional
        Attempt to load the data from a preprocessed HDF5 file.
    dump_to_hdf5 : bool, optional
        Dump the loaded data to an HDF5 file for faster loading next time.

    Returns
    -------
    rLG : 1-dimensional array of shape `(nLG,)`
        The observer's distance from the center of the void.
    velocities : 3-dimensional array of shape (nLG, nrad, nphi)
        Void velocities for different observers, radial distances, and angles.
    """
    if profile not in ["exp", "gauss", "mb"]:
        raise ValueError("profile must be one of 'exp', 'gauss', 'mb'")

    if kind not in ["density", "vrad", "vx", "vy"]:
        raise ValueError("kind must be one of 'density', 'vrad', 'vx', 'vy'.")

    fdir_base = "/mnt/extraspace/rstiskalek/catalogs/IndranilVoid/SizeVariation"  # noqa
    fdir = join(fdir_base, "sizenumber10")
    fname_scratch = join(fdir, f"processed_fiducial_{profile}_{kind}.hdf5")

    if try_load_from_hdf5 and exists(fname_scratch):
        fprint(f"loading pre-processed data from `{fname_scratch}`.")
        with File(fname_scratch, 'r') as f:
            rLG = f['rLG'][...]
            data = f['data'][...]

        return rLG, data

    if kind == "density":
        fdir = join(fdir, "rho_data")
        tag = "rho"
    elif kind == "vrad":
        fdir = join(fdir, "vr_data")
        tag = "v_pec"
    elif kind == "vx":
        fdir = join(fdir, "vx_data")
        tag = "v_x"
    elif kind == "vy":
        fdir = join(fdir, "vy_data")
        tag = "v_y"
    else:
        raise ValueError(f"Unknown kind: `{kind}`.")

    profile = profile.upper()
    fdir = join(fdir, f"{profile}profile")

    files = glob(join(fdir, "*.dat"))
    rLG = [int(search(rf'{tag}_{profile}profile_rLG_(\d+)', f).group(1))
           for f in files]
    rLG = np.sort(rLG)

    for i, ri in enumerate(tqdm(rLG, desc=f"Loading void `{kind}`observer data")):  # noqa
        f = join(fdir, f"{tag}_{profile}profile_rLG_{ri}.dat")
        data_i = np.genfromtxt(f).T

        if i == 0:
            data = np.full((len(rLG), *data_i.shape), np.nan, dtype=np.float32)

        data[i] = data_i

    if np.any(np.isnan(data)):
        raise ValueError("Found NaNs in loaded data.")

    if dump_to_hdf5:
        fprint(f"dumping data to `{fname_scratch}`.")
        with File(fname_scratch, 'w') as f:
            f.create_dataset('rLG', data=rLG)
            f.create_dataset('data', data=data)

    return rLG, data


def load_void_size_variation(profile, kind, try_load_from_hdf5=True,
                             dump_to_hdf5=True):
    """
    Load the void velocities from Sergij & Indranil's files for a given kind
    of void profile per observer with varying void sizes.

    The original files are slow to load, so depending on the flags attempts
    to load from a preprocessed HDF5 file

    Parameters
    ----------
    profile : str
        Void profile to load. One of "exp", "gauss", "mb".
    kind : str
        Data kind, either "density" or "vrad".
    try_load_from_hdf5 : bool, optional
        Attempt to load the data from a preprocessed HDF5 file.
    dump_to_hdf5 : bool, optional
        Dump the loaded data to an HDF5 file for faster loading next time.

    Returns
    -------
    sizes: 1-dimensional array of shape `(nsize,)`
        Relative void sizes.
    rLG : 1-dimensional array of shape `(nLG,)`
        The observer's distance from the center of the void.
    velocities : 3-dimensional array of shape (nLG, nrad, nphi)
        Void velocities for different void sizes, observers, radial distances,
        and angles.
    """
    if profile not in ["exp", "gauss", "mb"]:
        raise ValueError("profile must be one of 'exp', 'gauss', 'mb'")

    if kind not in ["density", "vrad"]:
        raise ValueError("kind must be one of 'density', 'vrad'")

    base_dir = "/mnt/extraspace/rstiskalek/catalogs/IndranilVoid/SizeVariation"
    fname_scratch = join(base_dir, f"processed_{profile}_{kind}.hdf5")

    if try_load_from_hdf5 and exists(fname_scratch):
        fprint(f"loading pre-processed data from `{fname_scratch}`.")
        with File(fname_scratch, 'r') as f:
            size = f['size'][...]
            rLG = f['rLG'][...]
            data = f['data'][...]

        return size, rLG, data

    size_indxs = sorted(int(search(r'sizenumber(\d+)', d).group(1))
                        for d in glob(join(base_dir, 'sizenumber*')))
    size = np.array(size_indxs) / 10

    # Loop over the void sizes
    for ki, k in enumerate(tqdm(size_indxs, desc=f"Loading {profile}, {kind} void size variation data")):  # noqa
        fdir = join(base_dir, f"sizenumber{str(k).zfill(2)}")
        if kind == "density":
            fdir = join(fdir, "rho_data")
            tag = "rho"
        else:
            fdir = join(fdir, "vr_data")
            tag = "v_pec"

        profile = profile.upper()
        fdir = join(fdir, f"{profile}profile")

        files = glob(join(fdir, "*.dat"))
        rLG = [int(search(rf'{tag}_{profile}profile_rLG_(\d+)', f).group(1))
               for f in files]
        rLG = np.sort(rLG)

        for i, ri in enumerate(rLG):
            f = join(fdir, f"{tag}_{profile}profile_rLG_{ri}.dat")
            data_i = np.genfromtxt(f).T

            if i == 0 and ki == 0:
                data = np.full((len(size_indxs), len(rLG), *data_i.shape),
                               np.nan, dtype=np.float32)

            data[ki, i] = data_i

    if np.any(np.isnan(data)):
        raise ValueError("Found NaNs in loaded data.")

    if dump_to_hdf5:
        fprint(f"dumping data to `{fname_scratch}`.")
        with File(fname_scratch, 'w') as f:
            f.create_dataset('size', data=size)
            f.create_dataset('rLG', data=rLG)
            f.create_dataset('data', data=data)

    return size, rLG, data

###############################################################################
#                      Interpolation of void velocities                       #
###############################################################################


def interpolate_fiducial_void(void_size, rLG, r, phi, data, void_size_min,
                              void_size_max, rgrid_min, rgrid_max, rLG_min,
                              rLG_max, order=1):
    """
    Interpolate the void velocities from Sergij & Indranil's files for a given
    observer over a set of radial distances and at angles specifying the
    galaxies.

    `void_size`, `void_size_min`, and `void_size_max` are not used, but are
    kept for consistency with the other interpolation functions.

    Parameters
    ----------
    void_size : float
        Not used. Pass arbitrary value.
    rLG : float
        The observer's distance from the center of the void.
    r : 1-dimensional array of shape `(nsteps,)
        The radial distances at which to interpolate the velocities.
    phi : 1-dimensional array of shape `(ngal,)`
        The angles at which to interpolate the velocities, in degrees,
        defining the galaxy position.
    data : 3-dimensional array of shape (nLG, nrad, nphi)
        The void velocities for different observers, radial distances, and
        angles.
    void_size_min, void_size_max : float
        Not used. Pass arbitrary values.
    rgrid_min, rgrid_max : float
        The minimum and maximum radial distances in the data.
    rLG_min, rLG_max : float
        The minimum and maximum observer distances in the data.
    order : int, optional
        The order of the interpolation. Default is 1, can be 0.

    Returns
    -------
    vel : 2-dimensional array of shape `(ngal, nsteps)`
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


def interpolate_size_var_void(void_size, rLG, r, phi, data, void_size_min,
                              void_size_max, rgrid_min, rgrid_max, rLG_min,
                              rLG_max, order=1):
    """
    Interpolate the void velocities from Sergij & Indranil's files for a given
    void size and observer over a set of radial distances and at angles
    specifying the galaxies.

    Parameters
    ----------
    void_size : float
        The relative void size.
    rLG : float
        The observer's distance from the center of the void.
    r : 1-dimensional array of shape `(nsteps,)
        The radial distances at which to interpolate the velocities.
    phi : 1-dimensional array of shape `(ngal,)`
        The angles at which to interpolate the velocities, in degrees,
        defining the galaxy position.
    data : 3-dimensional array of shape (nLG, nrad, nphi)
        The void velocities for different observers, radial distances, and
        angles.
    void_size_min, void_size_max : float
        The minimum and maximum relative void sizes in the data.
    rgrid_min, rgrid_max : float
        The minimum and maximum radial distances in the data.
    rLG_min, rLG_max : float
        The minimum and maximum observer distances in the data.
    order : int, optional
        The order of the interpolation. Default is 1, can be 0.

    Returns
    -------
    vel : 2-dimensional array of shape `(ngal, nsteps)`
    """
    nsize, nLG, nrad, nphi = data.shape

    # Normalize the void size and rLG to the grid scale
    void_size_normalized = ((void_size - void_size_min)
                            / (void_size_max - void_size_min) * (nsize - 1))
    void_size_normalized = jnp.repeat(void_size_normalized, r.size)

    rLG_normalized = (rLG - rLG_min) / (rLG_max - rLG_min) * (nLG - 1)
    rLG_normalized = jnp.repeat(rLG_normalized, r.size)

    r_normalized = (r - rgrid_min) / (rgrid_max - rgrid_min) * (nrad - 1)

    # Function to perform interpolation for a single phi
    def interpolate_single_phi(phi_val):
        # Normalize phi to match the grid
        phi_normalized = phi_val / 180 * (nphi - 1)

        # Create the grid for this specific phi
        X = jnp.vstack([void_size_normalized,
                        rLG_normalized,
                        r_normalized,
                        jnp.repeat(phi_normalized, r.size)])

        # Interpolate over the data using map_coordinates. The mode is nearest
        # to avoid extrapolation. But values outside of the grid should never
        # occur.
        return map_coordinates(data, X, order=order, mode='nearest')

    return vmap(interpolate_single_phi)(phi)


###############################################################################
#                          Mock void data                                     #
###############################################################################


def mock_void(vrad_data, rLG_index, profile,
              a_TF=-22.8, b_TF=-7.2, sigma_TF=0.1, sigma_v=100.,
              mean_eta=0.069, std_eta=0.078, mean_e_eta=0.012,
              mean_mag=10.31, std_mag=0.83, mean_e_mag=0.044,
              bmin=None, add_malmquist=False, nsamples=2000, seed=42,
              Om0=0.3175, verbose=False, **kwargs):
    """Mock 2MTF-like TFR data with void velocities."""
    truths = {"a": a_TF, "b": b_TF, "e_mu": sigma_TF, "sigma_v": sigma_v,
              "mean_eta": mean_eta, "std_eta": std_eta,
              "mean_mag": mean_mag, "std_mag": std_mag,
              }

    gen = np.random.default_rng(seed)

    # Sample the sky-distribution, either full-sky or mask out the Galactic
    # plane.
    l = gen.uniform(0, 360, size=nsamples)  # noqa
    if bmin is None:
        b = np.arcsin(gen.uniform(-1, 1, size=nsamples))
    else:
        b = np.arcsin(gen.uniform(np.sin(np.deg2rad(bmin)), 1,
                                  size=nsamples))
        b[gen.rand(nsamples) < 0.5] *= -1

    b = np.rad2deg(b)

    RA, DEC = galactic_to_radec(l, b)
    # Calculate the angular separation from the void axis, in degrees.
    phi = angular_distance_from_void_axis(RA, DEC)

    # Sample the linewidth of each galaxy from a Gaussian distribution to mimic
    # the MNR procedure.
    eta_true = gen.normal(mean_eta, std_eta, nsamples)
    eta_obs = gen.normal(eta_true, mean_e_eta)

    # Subtract the mean of the observed linewidths, so that they are
    # centered around zero. For consistency subtract from both observed
    # and true values.
    eta_mean_sampled = np.mean(eta_obs)
    eta_true -= eta_mean_sampled
    eta_obs -= eta_mean_sampled

    # Sample the magnitude from some Gaussian distribution to replicate MNR.
    mag_true = gen.normal(mean_mag, std_mag, nsamples)
    mag_obs = gen.normal(mag_true, mean_e_mag)

    # Calculate the 'true' distance modulus and redshift from the TFR distance.
    mu_TFR = mag_true - (a_TF + b_TF * eta_true)
    if add_malmquist:
        raise NotImplementedError("Malmquist bias not implemented yet.")
    else:
        mu_true = gen.normal(mu_TFR, sigma_TF)

    # Convert the true distance modulus to true distance and cosmological
    # redshift.
    r = distmod2dist(mu_true, Om0)
    zcosmo = distmod2redshift(mu_true, Om0)

    # Little h of this void profile
    h = select_void_h(profile)

    # Extract the velocities for the galaxies from the grid for this LG
    # index.
    vrad_data_rLG = vrad_data[rLG_index]

    r_grid = np.arange(0, 251) * h
    phi_grid = np.arange(0, 181)
    Vr = RegularGridInterpolator((r_grid, phi_grid), vrad_data_rLG,
                                 fill_value=np.nan, bounds_error=False,
                                 method="cubic")(np.vstack([r, phi]).T)

    # The true redshift of the source.
    zCMB_true = (1 + zcosmo) * (1 + Vr / SPEED_OF_LIGHT) - 1
    zCMB_obs = gen.normal(zCMB_true, sigma_v / SPEED_OF_LIGHT)

    sample = {"RA": RA,
              "DEC": DEC,
              "z_CMB": zCMB_obs,
              "eta": eta_obs,
              "mag": mag_obs,
              "e_eta": np.ones(nsamples) * mean_e_eta,
              "e_mag": np.ones(nsamples) * mean_e_mag,
              "r": r,
              "distmod_true": mu_true,
              "distmod_TFR": mu_TFR}

    # Apply a true distance cut to the mocks.
    mask = r < np.max(r_grid)
    for key in sample:
        sample[key] = sample[key][mask]

    if verbose and np.any(~mask):
        print(f"Removed {(~mask).sum()} out of {mask.size} samples "
              "due to the true distance cutoff.")

    return sample, truths


###############################################################################
#                        Void-predicted bulk flows                            #
###############################################################################


def void_velocity_vector(X_cartesian, vx_grid, vy_grid, r_grid, phi_grid,
                         vvoid_to_subtract=None):
    """
    Calculate the 3D velocity of each galaxy in ICRS.

    Parameters
    ----------
    X_cartesian : 2-dimensional array of shape `(npoints, 3)`
        Cartesian ICRS coordinates of the galaxies in Mpc.
    vx_grid, vy_grid : 2-dimensional array of shape `(nrad, nphi)`
        Grids of void velocities.
    r_grid, phi_grid : 1-dimensional array
        Radial and angular grid of the void model.
    vvoid_to_subtract : 1-dimensional array of shape `(3,)`, optional
        The void velocity to subtract from the bulk flow if some was added.

    Returns
    -------
    vel : 2-dimensional array of shape `(npoints, 3)`
        3D velocity of each galaxy in ICRS.
    """
    raise ValueError("This function does not yield consistent results. "
                     "Most likely there is something wrong with the data.. "
                     "investigating...")

    if not vx_grid.ndim == vy_grid.ndim == 2:
        raise ValueError("`vx_grid` and `vy_grid` must be 2-dimensional.")

    # Unit vector pointing towards (l, b) = (117, 4) in degrees.
    n_hat = np.asarray([0.4035093, -0.01363162, 0.91487399])

    # Unit vector pointing towards each galaxy.
    r = np.linalg.norm(X_cartesian, axis=1)
    r_hat = X_cartesian / r[:, np.newaxis]

    # Angular separation of each point from the void axis.
    cos_phi = np.sum(r_hat * n_hat[None, :], axis=1)

    # We use grid-like interpolation, it is faster.
    rgrid_min, rgrid_max = r_grid.min(), r_grid.max()
    phi_grid_min, phi_grid_max = phi_grid.min(), phi_grid.max()

    nrad, npi = vx_grid.shape
    r_normalized = (r - rgrid_min) / (rgrid_max - rgrid_min) * (nrad - 1)
    phi_normalized = np.arccos(cos_phi) * 180 / np.pi / (phi_grid_max - phi_grid_min) * (npi - 1)  # noqa

    vx = map_coordinates_np(vx_grid, np.vstack([r_normalized, phi_normalized]),
                            order=1, mode='constant', cval=np.nan)
    vy = map_coordinates_np(vy_grid, np.vstack([r_normalized, phi_normalized]),
                            order=1, mode='constant', cval=np.nan)
    # vrad = map_coordinates_np(vrad_grid, np.vstack([r_normalized, phi_normalized]),  # noqa
    #                           order=1, mode='constant', cval=np.nan)

    # Start calculating the 3D velocity, shape is `(npoints, 3)`
    vel = vx[:, None] * n_hat[None, :]
    vel += vy[:, None] * (r_hat - cos_phi[:, None] * n_hat[None, :]) / np.sqrt(1 - cos_phi[:, None]**2)  # noqa

    if vvoid_to_subtract is not None:
        vel -= vvoid_to_subtract[None, :]

    return vel, vx, vy
