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
from numba import jit
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates as map_coordinates_np
from tqdm import tqdm

from ..params import SPEED_OF_LIGHT
from ..utils import fprint, galactic_to_radec, radec_to_cartesian

###############################################################################
#                         Basic void computations                             #
###############################################################################


def angular_distance_from_void_axis_fiducial(RA, dec):
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


def angular_distance_from_void_axis(rhat, Vext):
    """
    Calculate the angular distance (in degrees) of unit vectors `rhat` of shape
    `(ngal, 3)` from the void axis which is opposite to `Vext` of shape `(3,)`.
    """
    # Fiducial Vext direction, pointing towards (l, b) = (297, -4) in degrees.
    # Vext = jnp.asarray([-0.4035093, 0.01363162, -0.91487399])  # but here we
    # want to infer it!
    cos_phi = -jnp.sum(rhat * Vext[None, :], axis=1) / jnp.linalg.norm(Vext)
    cos_phi = jnp.clip(cos_phi, -1, 1,)
    return jnp.arccos(cos_phi) * 180 / jnp.pi


def select_void_h(void_size_percent, profile, fname=None, return_all=False):
    if fname is None:
        fname = "/mnt/extraspace/rstiskalek/catalogs/IndranilVoid/SizeVariation_newDecember/H0_of_voids.dat"  # noqa

    profiles = ["mb", "gauss", "exp"]

    if profile not in profiles:
        raise ValueError(f"Profile `{profile}` not supported. "
                         f"Must be one of `{profiles}`.")
    d = np.genfromtxt(fname)

    sizes = d[:, 0].astype(int)
    H0s = d[:, 1 + profiles.index(profile)]

    if return_all:
        return sizes.astype(float), H0s / 100

    ks = np.where(sizes == void_size_percent)[0]

    if len(ks) == 0:
        raise ValueError(f"Void size {void_size_percent} not found.")

    return H0s[ks[0]] / 100


###############################################################################
#                            I/O of the void data                             #
###############################################################################


def load_void_fiducial(profile, kind, size_indx=None, try_load_from_hdf5=True,
                       dump_to_hdf5=True):
    """
    Load the void velocities from Sergij & Indranil's files for a given kind
    of void profile per observer. If `size_index` is `None`, load the fiducial
    void size, otherwise load the void size at the given index.

    Parameters
    ----------
    profile : str
        Void profile to load. One of "exp", "gauss", "mb".
    kind : str
        Data kind. One of "density", "vrad", "vx", or "vy".
    size_indx : int, optional
        Index of the void size to load. If `None`, load the fiducial void size.
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
        raise ValueError("`profile` must be one of 'exp', 'gauss', 'mb'")

    if kind not in ["density", "vrad", "vx", "vy"]:
        raise ValueError(
            "`kind` must be one of 'density', 'vrad', 'vx', 'vy'.")

    fdir_base = "/mnt/extraspace/rstiskalek/catalogs/IndranilVoid/SizeVariation_newDecember"  # noqa

    if size_indx is None:
        size_indx = 100

    fdir = join(fdir_base, f"sizenumber{size_indx}")
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

    if len(files) == 0:
        raise ValueError(f"No files found in `{fdir}`.")

    for i, ri in enumerate(tqdm(rLG, desc=f"Loading void `{kind}`observer data")):  # noqa
        f = join(fdir, f"{tag}_{profile}profile_rLG_{ri}.dat")
        data_i = np.genfromtxt(f)

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


def load_void_size_variation(profile, kind, which_run="all",
                             try_load_from_hdf5=True, dump_to_hdf5=True):
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
        Data kind, either "density", "vrad", "vx", or "vy".
    which_run : str
        Which run to load, either "coarse", "zoom" or "all".
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
        raise ValueError("`profile` must be one of 'exp', 'gauss', 'mb'")

    def make_mask(size):
        if which_run == "coarse":
            return size % 10 == 0
        elif which_run == "zoom":
            return size <= 20
        elif which_run == "all":
            return np.ones_like(size, dtype=bool)
        else:
            raise ValueError("`which_run` must be one of 'coarse' or 'zoom'.")

    if kind not in ["density", "vrad", "vx", "vy"]:
        raise ValueError("`kind` must be one of 'density', 'vrad'")


    base_dir = "/mnt/extraspace/rstiskalek/catalogs/IndranilVoid/SizeVariation_newDecember"  # noqa
    fname_scratch = join(base_dir, f"processed_{profile}_{kind}.hdf5")

    if try_load_from_hdf5 and exists(fname_scratch):
        fprint(f"loading pre-processed data from `{fname_scratch}`.")
        with File(fname_scratch, 'r') as f:
            size = f['size'][...]
            rLG = f['rLG'][...]
            data = f['data'][...]

        m = make_mask(size)
        return size[m].astype(np.float32) / 100, rLG, data[m]

    size_indxs = sorted(int(search(r'sizenumber(\d+)', d).group(1))
                        for d in glob(join(base_dir, 'sizenumber*')))
    size = np.asarray(size_indxs, dtype=int)

    # Loop over the void sizes
    for ki, k in enumerate(tqdm(size_indxs, desc=f"Loading {profile}, {kind} void size variation data")):  # noqa
        fdir = join(base_dir, f"sizenumber{str(k).zfill(3)}")
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

        if len(files) == 0:
            raise ValueError(f"No files found in `{fdir}`.")

        for i, ri in enumerate(rLG):
            f = join(fdir, f"{tag}_{profile}profile_rLG_{ri}.dat")
            data_i = np.genfromtxt(f)

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

    m = make_mask(size)
    return size[m].astype(np.float32) / 100, rLG, data[m]

###############################################################################
#                      Interpolation of void velocities                       #
###############################################################################


def interpolate_fiducial_void(void_size, rLG, h_void, Vext, r, rhat, data,
                              void_size_min, void_size_max, rgrid_min,
                              rgrid_max, rLG_min, rLG_max, order=1):
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
        The observer's distance from the center of the void in Mpc.
    h_void : float
        The void Hubble parameter to convert from Mpc / h to Mpc.
    Vext : 1-dimensional array of shape `(3,)`
        The void external velocity used to define the void axis.
    r : 1-dimensional array of shape `(nsteps,)
        The radial distances at which to interpolate the velocities in Mpc / h.
    rhat : 2-dimensional array of shape `(ngal, 3)`
        The unit vectors defining the galaxy positions on the sky.
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
        The order of interpolation. Default is 1, can be 0.

    Returns
    -------
    vel : 2-dimensional array of shape `(ngal, nsteps)`
    """
    nLG, nphi, nrad = data.shape

    rLG_sign = jnp.sign(rLG)
    rLG_is_negative = rLG < 0

    # Normalize rLG to the grid scale
    rLG_normalized = (jnp.abs(rLG) - rLG_min) / (rLG_max - rLG_min) * (nLG - 1)
    rLG_normalized = jnp.repeat(rLG_normalized, r.size)
    r_normalized = (r / h_void - rgrid_min) / (rgrid_max - rgrid_min) * (nrad - 1)  # noqa

    # Function to perform interpolation for a single phi
    def interpolate_single_phi(phi_val):
        # Normalize phi to match the grid
        phi_normalized = phi_val / 180 * (nphi - 1)

        # Create the grid for this specific phi
        X = jnp.vstack([rLG_normalized,
                        jnp.repeat(phi_normalized, r.size),
                        r_normalized])

        # Interpolate over the data using map_coordinates. The mode is nearest
        # to avoid extrapolation. But values outside of the grid should never
        # occur.
        return map_coordinates(data, X, order=order, mode='nearest')

    phi = angular_distance_from_void_axis(rhat, Vext)

    return vmap(interpolate_single_phi)(
        rLG_is_negative * 180 + rLG_sign * phi)


def interpolate_size_var_void(void_size, rLG, h_void, Vext, r, rhat, data,
                              void_size_min, void_size_max, rgrid_min,
                              rgrid_max, rLG_min, rLG_max, order=1):
    """
    Interpolate the void velocities from Sergij & Indranil's files for a given
    void size and observer over a set of radial distances and at angles
    specifying the galaxies.

    Parameters
    ----------
    void_size : float
        The relative void size.
    rLG : float
        The observer's distance from the center of the void in Mpc.
    h_void : float
        The void Hubble parameter to convert from Mpc / h to Mpc.
    Vext : 1-dimensional array of shape `(3,)`
        The void external velocity used to define the void axis.
    r : 1-dimensional array of shape `(nsteps,)
        The radial distances at which to interpolate the velocities in Mpc / h.
    rhat : 2-dimensional array of shape `(ngal, 3)`
        The unit vectors defining the galaxy positions on the sky.
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
        The order of interpolation. Default is 1, can be 0.

    Returns
    -------
    vel : 2-dimensional array of shape `(ngal, nsteps)`
    """
    nsize, nLG, nphi, nrad = data.shape

    rLG_sign = jnp.sign(rLG)
    rLG_is_negative = rLG < 0

    # Normalize the void size and rLG to the grid scale
    void_size_normalized = ((void_size - void_size_min)
                            / (void_size_max - void_size_min) * (nsize - 1))
    void_size_normalized = jnp.repeat(void_size_normalized, r.size)

    rLG_normalized = (jnp.abs(rLG) - rLG_min) / (rLG_max - rLG_min) * (nLG - 1)
    rLG_normalized = jnp.repeat(rLG_normalized, r.size)

    r_normalized = (r / h_void - rgrid_min) / (rgrid_max - rgrid_min) * (nrad - 1)  # noqa

    # Function to perform interpolation for a single phi
    def interpolate_single_phi(phi_val):
        # Normalize phi to match the grid
        phi_normalized = phi_val / 180 * (nphi - 1)

        # Create the grid for this specific phi
        X = jnp.vstack([void_size_normalized,
                        rLG_normalized,
                        jnp.repeat(phi_normalized, r.size),
                        r_normalized])

        # Interpolate over the data using map_coordinates. The mode is nearest
        # to avoid extrapolation. But values outside of the grid should never
        # occur.
        return map_coordinates(data, X, order=order, mode='nearest')

    phi = angular_distance_from_void_axis(rhat, Vext)

    return vmap(interpolate_single_phi)(
        rLG_is_negative * 180 + rLG_sign * phi)


###############################################################################
#                          Mock void data                                     #
###############################################################################


def mock_void(vrad_data, h_void, a_TF=-22.8, b_TF=-7.2, sigma_TF=0.1,
              sigma_v=100., Vext_mag=0., mean_eta=0.069, std_eta=0.078,
              mean_e_eta=0.012, mean_mag=10.31, std_mag=0.83, mean_e_mag=0.044,
              beta=1., bmin=None, add_malmquist=False, nsamples=2000, seed=42,
              negative_Roffset=False, Om0=0.3, verbose=False, **kwargs):
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
    phi = angular_distance_from_void_axis_fiducial(RA, DEC)
    if negative_Roffset:
        phi = 180 - phi

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
    zcosmo = interp_distmod2redshift(mu_true, Om0)
    r = interp_distmod2dist(mu_true, Om0)

    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(zcosmo)):
        raise ValueError("Some distance moduli are outside the interpolation "
                         "range.")

    # Extract the velocities for the galaxies from the grid for this LG
    # index.
    len_phi, len_r = vrad_data.shape
    r_grid = np.arange(0, len_r) * h_void
    phi_grid = np.arange(0, len_phi)
    print(f"Assuming grid of {len_phi} points in phi and "
          f"{len_r} points in r.")
    Vr = RegularGridInterpolator((phi_grid, r_grid), vrad_data,
                                 fill_value=np.nan, bounds_error=False,
                                 method="cubic")(np.vstack([phi, r]).T)
    if np.any(~np.isfinite(Vr)):
        raise ValueError("Some void velocities are outside the interpolation "
                         "range.")
    Vr *= beta

    if Vext_mag > 0:
        rhat = radec_to_cartesian(np.vstack([np.ones_like(RA), RA, DEC]).T)
        Vext = Vext_mag * jnp.asarray([-0.4035093, 0.01363162, -0.91487399])

        Vr += np.sum(rhat * Vext[None, :], axis=1)

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
                         Vext=None, is_negative_Roffset=False,
                         return_icrs=True):
    """
    Calculate the 3D velocity of each galaxy in ICRS.

    Parameters
    ----------
    X_cartesian : 2-dimensional array of shape `(npoints, 3)`
        Cartesian ICRS coordinates of the galaxies in Mpc.
    vx_grid, vy_grid : 2-dimensional array of shape `(nphi, nrad)`
        Grids of void velocities.
    r_grid, phi_grid : 1-dimensional array
        Radial and angular grid of the void model.
    Vext : 1-dimensional array of shape `(3,)`, optional
        External velocity of the void in ICRS coordinates, its opposite
        defines the void axis.
    is_negative_Roffset : bool, optional
        Whether the observer offset is negative, in which case flips the
        sign of `cos(phi)`.
    return_icrs : bool, optional
        Whether to return the velocity in ICRS coordinates, otherwise in
        the void frame.

    Returns
    -------
    vel : 2-dimensional array of shape `(npoints, 3)`
        3D velocity of each galaxy in ICRS.
    """
    if not vx_grid.ndim == vy_grid.ndim == 2:
        raise ValueError("`vx_grid` and `vy_grid` must be 2-dimensional.")

    if Vext is None:
        Vext = np.asarray([-0.4035093, 0.01363162, -0.91487399])

    # Note the negative sign, the void axis is opposite to Vext.
    n_hat = -Vext / np.linalg.norm(Vext)

    # Unit vector pointing towards each galaxy.
    r = np.linalg.norm(X_cartesian, axis=1)
    r_hat = X_cartesian / r[:, None]

    # Angular separation of each point from the void axis.
    cos_phi = np.sum(r_hat * n_hat[None, :], axis=1)
    if is_negative_Roffset:
        cos_phi *= -1
        n_hat *= -1
    # Clip in case of small numerical errors.
    cos_phi = np.clip(cos_phi, -1, 1)

    # We use grid-like interpolation, it is faster.
    rgrid_min, rgrid_max = r_grid.min(), r_grid.max()
    phi_grid_min, phi_grid_max = phi_grid.min(), phi_grid.max()

    nphi, nrad = vx_grid.shape
    r_normalized = (r - rgrid_min) / (rgrid_max - rgrid_min) * (nrad - 1)
    phi_normalized = np.arccos(cos_phi) * 180 / np.pi / (phi_grid_max - phi_grid_min) * (nphi - 1)  # noqa

    vx = map_coordinates_np(vx_grid, np.vstack([phi_normalized, r_normalized]),
                            order=1, mode='constant', cval=np.nan)
    vy = map_coordinates_np(vy_grid, np.vstack([phi_normalized, r_normalized]),
                            order=1, mode='constant', cval=np.nan)

    if not return_icrs:
        return np.vstack([vx, vy]).T

    # Start calculating the 3D velocity, shape is `(npoints, 3)`
    vel = vx[:, None] * n_hat[None, :]
    vel += vy[:, None] * (r_hat - cos_phi[:, None] * n_hat[None, :]) / np.sqrt(1 - cos_phi[:, None]**2)  # noqa

    return vel


@jit(nopython=True)
def _cell_rdist(i, j, k, Ncells, boxsize):
    """Radial distance of the center of a cell from the center of the box."""
    xi = boxsize / Ncells * (i + 0.5) - boxsize / 2
    yi = boxsize / Ncells * (j + 0.5) - boxsize / 2
    zi = boxsize / Ncells * (k + 0.5) - boxsize / 2

    return (xi**2 + yi**2 + zi**2)**0.5


@jit(nopython=True, boundscheck=False)
def _field_enclosed(field, rmax, boxsize):
    Ncells = field.shape[0]
    cell_volume = (boxsize / Ncells)**3

    weight = 0.
    volume = 0.
    for i in range(Ncells):
        for j in range(Ncells):
            for k in range(Ncells):
                if _cell_rdist(i, j, k, Ncells, boxsize) < rmax:
                    weight += field[i, j, k]
                    volume += 1.

    return weight * cell_volume, volume * cell_volume


def field_enclosed(field, distances, boxsize, verbose=True):
    """
    Calculate the approximate enclosed field within a given radius. Sums
    the field over all cells whose centers are within the radius.

    Parameters
    ----------
    field : 3-dimensional array
        Field to calculate the enclosed sum of.
    rmax : 1-dimensional array
        Radii to calculate the enclosed mass at.
    boxsize : float
        Box size in `Mpc / h` (or the same as `rmax`).
    verbose : bool
        Verbosity flag.

    Returns
    -------
    enclosed_field : 1-dimensional array
        Enclosed mass at each distance.
    enclosed_volume : 1-dimensional array
        Enclosed grid-like volume at each distance.
    """
    enclosed_field = np.zeros_like(distances)
    enclosed_volume = np.zeros_like(distances)

    for i, dist in enumerate(tqdm(distances, disable=not verbose)):
        enclosed_field[i], enclosed_volume[i] = _field_enclosed(
            field, dist, boxsize)

    return enclosed_field, enclosed_volume


def make_grid(ngrid, rmax, boxsize, reshape_to_3d=True):
    """
    Make a grid of `ngrid` cells in a subbox if size `2 rmax` in a box of
    size `boxsize`.
    """
    boxsize = 2 * rmax

    x = boxsize / ngrid * (np.arange(ngrid) + 0.5) - boxsize / 2
    X = np.vstack([x.reshape(-1,) for x in np.meshgrid(x, x, x)]).T

    if reshape_to_3d:
        X = X.reshape(ngrid, ngrid, ngrid)

    return X


def void_bulk_flow(r, vx, vy, ngrid, r_grid, phi_grid, Vext=None,
                   is_negative_Roffset=False, in_icrs=True, verbose=True):
    """
    Calculate the bulk flow of the void velocity field.

    Parameters
    ----------
    r : 1-dimensional array
        Radial distances at which to calculate the bulk flow.
    vx, vy : 2-dimensional array of shape `(nphi, nrad)`
        Velocity along the x- and y-axis of the void.
    ngrid : int
        Number of grid points in each dimension.
    r_grid, phi_grid : 1-dimensional array
        Void radial and angular grid.
    Vext : 1-dimensional array of shape `(3,)`, optional
        External velocity of the void in ICRS coordinates.
    is_negative_Roffset : bool, optional
        Whether the observer offset is negative, in which case flips the
        sign of `cos(phi)`.
    in_icrs : bool, optional
        Whether to return the bulk flow in ICRS coordinates or in the void
        coordinates.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    bulk_flow : 2-dimensional array of shape `(len(r), 3)`
        Bulk flow at each distance.
    """
    rmax = np.max(r)
    boxsize = 2 * rmax
    X = make_grid(ngrid, rmax, boxsize, reshape_to_3d=False)

    vel = void_velocity_vector(
        X, vx, vy, r_grid, phi_grid, Vext=Vext,
        is_negative_Roffset=is_negative_Roffset, return_icrs=in_icrs)

    ndim = 3 if in_icrs else 2
    bulk_flow = np.full((len(r), ndim), np.nan)

    for n in range(ndim):
        vi = vel[:, n].reshape(ngrid, ngrid, ngrid)
        enclosed_vel, enclosed_vol = field_enclosed(vi, r, boxsize, verbose)

        # Don't divide if nothing is enclosed.
        m = enclosed_vol > 0
        bulk_flow[m, n] = enclosed_vel[m] / enclosed_vol[m]

    return bulk_flow


def void_monopole(r, vr, ngrid, r_grid, phi_grid, Vext=None,
                  is_negative_Roffset=False, verbose=True):
    """
    Calculate the monopole of the void velocity field.

    Parameters
    ----------
    r : 1-dimensional array
        Radial distances at which to calculate the monopole.
    vr : 2-dimensional array of shape `(nphi, nrad)`
        Radial void velocity field.
    ngrid : int
        Number of grid points in each dimension.
    r_grid, phi_grid : 1-dimensional array
        Void radial and angular grid.
    Vext : 1-dimensional array of shape `(3,)`, optional
        External velocity of the void in ICRS coordinates.
    is_negative_Roffset : bool, optional
        Whether the observer offset is negative, in which case flips the
        sign of `cos(phi)`.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    enclosed_vel : 1-dimensional array of shape `(len(r), )`
        Enclosed monopole velocity at each distance.
    """
    rmax = np.max(r)
    boxsize = 2 * rmax
    X = make_grid(ngrid, rmax, boxsize, reshape_to_3d=False)

    vel = void_velocity_vector(
        X, vr, np.zeros_like(vr), r_grid, phi_grid,
        Vext=Vext, is_negative_Roffset=is_negative_Roffset,
        return_icrs=False)
    vel_rad = vel[:, 0]

    enclosed_vel, enclosed_vol = field_enclosed(
        vel_rad.reshape(ngrid, ngrid, ngrid), r, boxsize, verbose)

    m = enclosed_vol > 0
    enclosed_vel[m] /= enclosed_vol[m]
    enclosed_vel[~m] = np.nan

    return enclosed_vel
