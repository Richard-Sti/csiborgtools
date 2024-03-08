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
Validation of the CSiBORG velocity field against PV measurements. Based on [1].

References
----------
[1] https://arxiv.org/abs/1912.09383.
"""
from datetime import datetime
from warnings import warn

import numpy as np
import numpyro
import numpyro.distributions as dist
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from h5py import File
from jax import numpy as jnp
from jax import vmap
from tqdm import tqdm, trange

from ..params import simname2Omega_m
from ..read import CSiBORG1Catalogue

SPEED_OF_LIGHT = 299792.458  # km / s


def t():
    """Shortcut to get the current time."""
    return datetime.now().strftime("%H:%M:%S")


def radec_to_galactic(ra, dec):
    """
    Convert right ascension and declination to galactic coordinates (all in
    degrees.)

    Parameters
    ----------
    ra, dec : float or 1-dimensional array
        Right ascension and declination in degrees.

    Returns
    -------
    l, b : float or 1-dimensional array
    """
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    return c.galactic.l.degree, c.galactic.b.degree


###############################################################################
#                             Data loader                                     #
###############################################################################


class DataLoader:
    """
    Data loader for the line of sight (LOS) interpolated fields and the
    corresponding catalogues.

    Parameters
    ----------
    simname : str
        Simulation name.
    catalogue : str
        Name of the catalogue with LOS objects.
    catalogue_fpath : str
        Path to the LOS catalogue file.
    paths : csiborgtools.read.Paths
        Paths object.
    ksmooth : int, optional
        Smoothing index.
    store_full_velocity : bool, optional
        Whether to store the full 3D velocity field. Otherwise stores only
        the radial velocity.
    """
    def __init__(self, simname, catalogue, catalogue_fpath, paths,
                 ksmooth=None, store_full_velocity=False):
        print(f"{t()}: reading the catalogue.")
        self._cat = self._read_catalogue(catalogue, catalogue_fpath)
        self._catname = catalogue

        print(f"{t()}: reading the interpolated field.")
        self._field_rdist, self._los_density, self._los_velocity = self._read_field(  # noqa
            simname, catalogue, ksmooth, paths)

        if len(self._field_rdist) % 2 == 0:
            warn(f"The number of radial steps is even. Skipping the first "
                 f"step at {self._field_rdist[0]} because Simpson's rule "
                 "requires an odd number of steps.")
            self._field_rdist = self._field_rdist[1:]
            self._los_density = self._los_density[..., 1:]
            self._los_velocity = self._los_velocity[..., 1:]

        if len(self._cat) != len(self._los_density):
            raise ValueError("The number of objects in the catalogue does not "
                             "match the number of objects in the field.")

        print(f"{t()}: calculating the radial velocity.")
        nobject, nsim = self._los_density.shape[:2]

        # In case of Carrick 2015 the box is in galactic coordinates..
        if simname == "Carrick2015":
            d1, d2 = radec_to_galactic(self._cat["RA"], self._cat["DEC"])
        else:
            d1, d2 = self._cat["RA"], self._cat["DEC"]

        radvel = np.empty((nobject, nsim, len(self._field_rdist)),
                          self._los_velocity.dtype)
        for i in trange(nobject):
            for j in range(nsim):
                radvel[i, j, :] = radial_velocity_los(
                    self._los_velocity[:, i, j, ...], d1[i], d2[i])
        self._los_radial_velocity = radvel

        if not store_full_velocity:
            self._los_velocity = None

        Omega_m = simname2Omega_m(simname)

        # Normalize the CSiBORG density by the mean matter density
        if "csiborg" in simname:
            cosmo = FlatLambdaCDM(H0=100, Om0=Omega_m)
            mean_rho_matter = cosmo.critical_density0.to("Msun/kpc^3").value
            mean_rho_matter *= Omega_m
            self._los_density /= mean_rho_matter

        # Since Carrick+2015 provide `rho / <rho> - 1`
        if simname == "Carrick2015":
            self._los_density += 1

    @property
    def cat(self):
        """
        The distance indicators catalogue.

        Returns
        -------
        structured array
        """
        return self._cat

    @property
    def catname(self):
        """
        Name of the catalogue.

        Returns
        -------
        str
        """
        return self._catname

    @property
    def rdist(self):
        """
        Radial distances where the field was interpolated for each object.

        Returns
        -------
        1-dimensional array
        """
        return self._field_rdist

    @property
    def los_density(self):
        """
        Density field along the line of sight.

        Returns
        ----------
        3-dimensional array of shape (n_objects, n_simulations, n_steps)
        """
        return self._los_density

    @property
    def los_velocity(self):
        """
        Velocity field along the line of sight.

        Returns
        -------
        4-dimensional array of shape (n_objects, n_simulations, 3, n_steps)
        """
        if self._los_velocity is None:
            raise ValueError("The 3D velocities were not stored.")
        return self._los_velocity

    @property
    def los_radial_velocity(self):
        """
        Radial velocity along the line of sight.

        Returns
        -------
        3-dimensional array of shape (n_objects, n_simulations, n_steps)
        """
        return self._los_radial_velocity

    def _read_field(self, simname, catalogue, k, paths):
        """Read in the interpolated field."""
        out_density = None
        out_velocity = None
        has_smoothed = False

        nsims = paths.get_ics(simname)
        with File(paths.field_los(simname, catalogue), 'r') as f:
            has_smoothed = True if f[f"density_{nsims[0]}"].ndim > 2 else False
            if has_smoothed and (k is None or not isinstance(k, int)):
                raise ValueError("The output contains smoothed field but "
                                 "`ksmooth` is None. It must be provided.")

            for i, nsim in enumerate(tqdm(nsims)):
                if out_density is None:
                    nobject, nstep = f[f"density_{nsim}"].shape[:2]
                    out_density = np.empty(
                        (nobject, len(nsims), nstep), dtype=np.float32)
                    out_velocity = np.empty(
                        (3, nobject, len(nsims), nstep), dtype=np.float32)

                indx = (..., k) if has_smoothed else (...)
                out_density[:, i, :] = f[f"density_{nsim}"][indx]
                out_velocity[:, :, i, :] = f[f"velocity_{nsim}"][indx]

            rdist = f[f"rdist_{nsims[0]}"][:]

        return rdist, out_density, out_velocity

    def _read_catalogue(self, catalogue, catalogue_fpath):
        """
        Read in the distance indicator catalogue.
        """
        if catalogue == "A2":
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                arr = np.empty(len(f["RA"]), dtype=dtype)
                for key in f.keys():
                    arr[key] = f[key][:]
        elif catalogue == "LOSS" or catalogue == "Foundation":
            with File(catalogue_fpath, 'r') as f:
                grp = f[catalogue]

                dtype = [(key, np.float32) for key in grp.keys()]
                arr = np.empty(len(grp["RA"]), dtype=dtype)
                for key in grp.keys():
                    arr[key] = grp[key][:]
        elif "csiborg1" in catalogue:
            nsim = int(catalogue.split("_")[-1])
            cat = CSiBORG1Catalogue(nsim, bounds={"totmass": (1e13, None)})

            seed = 42
            gen = np.random.default_rng(seed)
            mask = gen.choice(len(cat), size=100, replace=False)

            keys = ["r_hMpc", "RA", "DEC"]
            dtype = [(key, np.float32) for key in keys]
            arr = np.empty(len(mask), dtype=dtype)

            sph_pos = cat["spherical_pos"]
            arr["r_hMpc"] = sph_pos[mask, 0]
            arr["RA"] = sph_pos[mask, 1]
            arr["DEC"] = sph_pos[mask, 2]
            # TODO: add peculiar velocit
        else:
            raise ValueError(f"Unknown catalogue: `{catalogue}`.")

        return arr


###############################################################################
#                       Supplementary flow functions                          #
###############################################################################


def radial_velocity_los(los_velocity, ra, dec):
    """
    Calculate the radial velocity along the line of sight.

    Parameters
    ----------
    los_velocity : 2-dimensional array of shape (3, n_steps)
        Line of sight velocity field.
    ra, dec : floats
        Right ascension and declination of the line of sight.
    is_degrees : bool, optional
        Whether the angles are in degrees.

    Returns
    -------
    1-dimensional array of shape (n_steps)
    """
    types = (float, np.float32, np.float64)
    if not isinstance(ra, types) and not isinstance(dec, types):
        raise ValueError("RA and dec must be floats.")

    if los_velocity.ndim != 2 and los_velocity.shape[0] != 3:
        raise ValueError("The shape of `los_velocity` must be (3, n_steps).")

    ra_rad, dec_rad = np.deg2rad(ra), np.deg2rad(dec)

    vx, vy, vz = los_velocity
    return (vx * np.cos(ra_rad) * np.cos(dec_rad)
            + vy * np.sin(ra_rad) * np.cos(dec_rad)
            + vz * np.sin(dec_rad))


###############################################################################
#                           JAX Flow model                                    #
###############################################################################


def lognorm_mean_std_to_loc_scale(mu, std):
    """
    Calculate the location and scale parameters for the log-normal distribution
    from the mean and standard deviation.

    Parameters
    ----------
    mu, std : float
        Mean and standard deviation.

    Returns
    -------
    loc, scale : float
    """
    loc = np.log(mu) - 0.5 * np.log(1 + (std / mu) ** 2)
    scale = np.sqrt(np.log(1 + (std / mu) ** 2))
    return loc, scale


def simps(y, dx):
    """
    Simpson's rule 1D integration, assuming that the number of steps is even
    and that the step size is constant.

    Parameters
    ----------
    y : 1-dimensional array
        Function values.
    dx : float
        Step size.

    Returns
    -------
    float
    """
    if len(y) % 2 == 0:
        raise ValueError("The number of steps must be odd.")

    return dx / 3 * jnp.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])


def dist2redshift(dist, Omega_m):
    """
    Convert comoving distance to cosmological redshift if the Universe is
    flat and z << 1.

    Parameters
    ----------
    dist : float or 1-dimensional array
        Comoving distance in `Mpc / h`.
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    float or 1-dimensional array
    """
    H0 = 100
    eta = 3 * Omega_m / 2
    return 1 / eta * (1 - (1 - 2 * H0 * dist / SPEED_OF_LIGHT * eta)**0.5)


def dist2distmodulus(dist, Omega_m):
    """
    Convert comoving distance to distance modulus, assuming z << 1.

    Parameters
    ----------
    dist : float or 1-dimensional array
        Comoving distance in `Mpc / h`.
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    float or 1-dimensional array
    """
    zcosmo = dist2redshift(dist, Omega_m)
    luminosity_distance = dist * (1 + zcosmo)
    return 5 * jnp.log10(luminosity_distance) + 25


def project_Vext(Vext_x, Vext_y, Vext_z, RA, dec):
    """
    Project the external velocity onto the line of sight along direction
    specified by RA/dec. Note that the angles must be in radians.

    Parameters
    ----------
    Vext_x, Vext_y, Vext_z : floats
        Components of the external velocity.
    RA, dec : floats
        Right ascension and declination in radians

    Returns
    -------
    float
    """
    cos_dec = jnp.cos(dec)
    return (Vext_x * jnp.cos(RA) * cos_dec
            + Vext_y * jnp.sin(RA) * cos_dec
            + Vext_z * jnp.sin(dec))


def predict_zobs(dist, beta, Vext_radial, vpec_radial, Omega_m):
    """
    Predict the observed redshift at a given comoving distance given some
    velocity field.

    Parameters
    ----------
    dist : float
        Comoving distance in `Mpc / h`.
    beta : float
        Velocity bias parameter.
    Vext_radial : float
        Radial component of the external velocity along the LOS.
    vpec_radial : float
        Radial component of the peculiar velocity along the LOS.
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    float
    """
    zcosmo = dist2redshift(dist, Omega_m)

    vrad = beta * vpec_radial + Vext_radial
    return (1 + zcosmo) * (1 + vrad / SPEED_OF_LIGHT) - 1


###############################################################################
#                          Flow validation models                             #
###############################################################################


def calculate_ptilde_wo_bias(xrange, mu, err, r_squared_xrange=None):
    """
    Calculate `ptilde(r)` without any bias.

    Parameters
    ----------
    xrange : 1-dimensional array
        Radial distances where the field was interpolated for each object.
    mu : float
        Comoving distance in `Mpc / h`.
    err : float
        Error on the comoving distance in `Mpc / h`.
    r_squared_xrange : 1-dimensional array, optional
        Radial distances squared where the field was interpolated for each
        object. If not provided, the `r^2` correction is not applied.

    Returns
    -------
    1-dimensional array
    """
    ptilde = jnp.exp(-0.5 * ((xrange - mu) / err)**2)

    if r_squared_xrange is not None:
        ptilde *= r_squared_xrange

    return ptilde


def calculate_ll_zobs(zobs, zobs_pred, sigma_v):
    """
    Calculate the likelihood of the observed redshift given the predicted
    redshift.

    Parameters
    ----------
    zobs : float
        Observed redshift.
    zobs_pred : float
        Predicted redshift.
    sigma_v : float
        Velocity uncertainty.

    Returns
    -------
    float
    """
    dcz = SPEED_OF_LIGHT * (zobs - zobs_pred)
    return jnp.exp(-0.5 * (dcz / sigma_v)**2) / jnp.sqrt(2 * np.pi) / sigma_v


class SD_PV_validation_model:
    """
    Simple distance peculiar velocity (PV) validation model, assuming that
    we already have a calibrated estimate of the comoving distance to the
    objects.

    Parameters
    ----------
    los_density : 2-dimensional array of shape (n_objects, n_steps)
        LOS density field.
    los_velocity : 3-dimensional array of shape (n_objects, n_steps)
        LOS radial velocity field.
    RA, dec : 1-dimensional arrays of shape (n_objects)
        Right ascension and declination in degrees.
    z_obs : 1-dimensional array of shape (n_objects)
        Observed redshifts.
    r_hMpc : 1-dimensional array of shape (n_objects)
        Estimated comoving distances in `h^-1 Mpc`.
    e_r_hMpc : 1-dimensional array of shape (n_objects)
        Errors on the estimated comoving distances in `h^-1 Mpc`.
    r_xrange : 1-dimensional array
        Radial distances where the field was interpolated for each object.
    Omega_m : float
        Matter density parameter.
    """

    def __init__(self, los_density, los_velocity, RA, dec, z_obs,
                 r_hMpc, e_r_hMpc, r_xrange, Omega_m):
        # Convert everything to JAX arrays.
        dt = jnp.float32
        self._los_density = jnp.asarray(los_density, dtype=dt)
        self._los_velocity = jnp.asarray(los_velocity, dtype=dt)
        self._RA = jnp.asarray(np.deg2rad(RA), dtype=dt)
        self._dec = jnp.asarray(np.deg2rad(dec), dtype=dt)
        self._z_obs = jnp.asarray(z_obs, dtype=dt)
        self._r_hMpc = jnp.asarray(r_hMpc, dtype=dt)
        self._e_rhMpc = jnp.asarray(e_r_hMpc, dtype=dt)

        # Get radius squared
        r2_xrange = r_xrange**2
        r2_xrange /= r2_xrange.mean()

        # Get the stepsize, we need it to be constant for Simpson's rule.
        dr = np.diff(r_xrange)
        if not np.all(np.isclose(dr, dr[0], atol=1e-5)):
            raise ValueError("The radial step size must be constant.")
        dr = dr[0]

        # Get the various vmapped functions
        self._vmap_ptilde_wo_bias = vmap(lambda mu, err: calculate_ptilde_wo_bias(r_xrange, mu, err, r2_xrange))                        # noqa
        self._vmap_simps = vmap(lambda y: simps(y, dr))
        self._vmap_zobs = vmap(lambda beta, Vr, vpec_rad: predict_zobs(r_xrange, beta, Vr, vpec_rad, Omega_m), in_axes=(None, 0, 0))    # noqa
        self._vmap_ll_zobs = vmap(lambda zobs, zobs_pred, sigma_v: calculate_ll_zobs(zobs, zobs_pred, sigma_v), in_axes=(0, 0, None))   # noqa

        # Vext_x, Vext_y, Vext_z: external velocity components
        self._dist_Vext = dist.Uniform(-1000, 1000)
        # We want sigma_v to be 150 +- 100 km / s (lognormal)
        self._dist_sigma_v = dist.LogNormal(
            *lognorm_mean_std_to_loc_scale(150, 100))
        # Density power-law bias
        self._dist_alpha = dist.LogNormal(
            *lognorm_mean_std_to_loc_scale(1.0, 0.5))
        # Velocity bias
        self._dist_beta = dist.Normal(1., 0.5)

    def __call__(self):
        """
        The simple distance NumPyro PV validation model. Samples the following
        parameters:
            - `Vext_x`, `Vext_y`, `Vext_z`: external velocity components
            - `alpha`: density bias parameter
            - `beta`: velocity bias parameter
            - `sigma_v`: velocity uncertainty
        """
        Vx = numpyro.sample("Vext_x", self._dist_Vext)
        Vy = numpyro.sample("Vext_y", self._dist_Vext)
        Vz = numpyro.sample("Vext_z", self._dist_Vext)
        alpha = numpyro.sample("alpha", self._dist_alpha)
        beta = numpyro.sample("beta", self._dist_beta)
        sigma_v = numpyro.sample("sigma_v", self._dist_sigma_v)

        Vext_rad = project_Vext(Vx, Vy, Vz, self._RA, self._dec)

        # Calculate p(r) and multiply it by the galaxy bias
        ptilde = self._vmap_ptilde_wo_bias(self._r_hMpc, self._e_rhMpc)
        ptilde *= self._los_density**alpha

        # Normalization of p(r)
        pnorm = self._vmap_simps(ptilde)

        # Calculate p(z_obs) and multiply it by p(r)
        zobs_pred = self._vmap_zobs(beta, Vext_rad, self._los_velocity)
        ptilde *= self._vmap_ll_zobs(self._z_obs, zobs_pred, sigma_v)

        ll = jnp.sum(jnp.log(self._vmap_simps(ptilde) / pnorm))
        numpyro.factor("ll", ll)


# def SN_PV_wcal_validation_model(los_overdensity=None, los_velocity=None,
#                                 RA=None, dec=None, z_CMB=None,
#                                 mB=None, x1=None, c=None,
#                                 e_mB=None, e_x1=None, e_c=None,
#                                 mu_xrange=None, r_xrange=None,
#                                 norm_r2_xrange=None, Omega_m=None, dr=None):
#     """
#     Pass
#     """
#     Vx = numpyro.sample("Vext_x", dist.Uniform(-1000, 1000))
#     Vy = numpyro.sample("Vext_y", dist.Uniform(-1000, 1000))
#     Vz = numpyro.sample("Vext_z", dist.Uniform(-1000, 1000))
#     beta = numpyro.sample("beta", dist.Uniform(-10, 10))
#
#     # TODO: Later sample these as well.
#     e_mu_intrinsic = 0.064
#     alpha_cal = 0.135
#     beta_cal = 2.9
#     mag_cal = -18.555
#     sigma_v = 112
#
#     # TODO: Check these for fiducial values.
#     mu = mB - mag_cal + alpha_cal * x1 - beta_cal * c
#     squared_e_mu = e_mB**2 + alpha_cal**2 * e_x1**2 + beta_cal**2 * e_c**2
#
#     squared_e_mu += e_mu_intrinsic**2
#     ll = 0.
#     for i in range(len(los_overdensity)):
#         # Project the external velocity for this galaxy.
#         Vext_rad = project_Vext(Vx, Vy, Vz, RA[i], dec[i])
#
#         dmu = mu_xrange - mu[i]
#         ptilde = norm_r2_xrange * jnp.exp(-0.5 * dmu**2 / squared_e_mu[i])
#         # TODO: Add some bias
#         ptilde *= (1 + los_overdensity[i])
#
#         zobs_pred = predict_zobs(r_xrange, beta, Vext_rad, los_velocity[i],
#                                  Omega_m)
#
#         dczobs = SPEED_OF_LIGHT * (z_CMB[i] - zobs_pred)
#
#         ll_zobs = jnp.exp(-0.5 * (dczobs / sigma_v)**2) / sigma_v
#
#         ll += jnp.log(simps(ptilde * ll_zobs, dr))
#         ll -= jnp.log(simps(ptilde, dr))
#
#     numpyro.factor("ll", ll)
