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
Validation of the CSiBORG velocity field against PV measurements. A lot here
is based on [1], though with many modifications. Throughout, comoving distances
are in `Mpc / h` and velocities in `km / s`.

References
----------
[1] https://arxiv.org/abs/1912.09383.
"""
from abc import ABC, abstractmethod

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from interpax import interp1d
from jax import jit
from jax import numpy as jnp
from jax import vmap
from jax.scipy.special import erf, logsumexp
from numpyro import factor, plate, sample
from numpyro.distributions import MultivariateNormal, Normal, Uniform
from quadax import simpson
from tqdm import trange

from ..params import SPEED_OF_LIGHT
from ..utils import fprint
from .cosmography import (dist2redshift, distmod2dist, distmod2dist_gradient,
                          distmod2redshift, gradient_redshift2dist)
from .selection import toy_log_magnitude_selection
from .void_model import (angular_distance_from_void_axis, interpolate_void,
                         load_void_data)

H0 = 100  # km / s / Mpc


###############################################################################
#                       Various flow utilities                                #
###############################################################################

def project_Vext(Vext_x, Vext_y, Vext_z, RA_radians, dec_radians):
    """Project the external velocity vector onto the line of sight."""
    cos_dec = jnp.cos(dec_radians)
    return (Vext_x * jnp.cos(RA_radians) * cos_dec
            + Vext_y * jnp.sin(RA_radians) * cos_dec
            + Vext_z * jnp.sin(dec_radians))


def predict_zobs(dist, beta, Vext_radial, vpec_radial, Omega_m):
    """
    Predict the observed redshift at a given comoving distance given some
    velocity field.
    """
    zcosmo = dist2redshift(dist, Omega_m)
    vrad = beta * vpec_radial + Vext_radial
    return (1 + zcosmo) * (1 + vrad / SPEED_OF_LIGHT) - 1


###############################################################################
#                          Flow validation models                             #
###############################################################################


def log_ptilde_wo_bias(xrange, mu, err_squared, log_r_squared_xrange):
    """Calculate `ptilde(r)` without imhomogeneous Malmquist bias."""
    return (-0.5 * (xrange - mu)**2 / err_squared
            - 0.5 * jnp.log(2 * np.pi * err_squared)
            + log_r_squared_xrange)


def likelihood_zobs(zobs, zobs_pred, e2_cz):
    """
    Calculate the likelihood of the observed redshift given the predicted
    redshift. Multiplies the redshifts by the speed of light.
    """
    dcz = SPEED_OF_LIGHT * (zobs - zobs_pred)
    return jnp.exp(-0.5 * dcz**2 / e2_cz) / jnp.sqrt(2 * np.pi * e2_cz)


def log_likelihood_zobs(zobs, zobs_pred, e2_cz):
    """
    Calculate the log-likelihood of the observed redshift given the predicted
    redshift. Multiplies the redshifts by the speed of light.
    """
    dcz = SPEED_OF_LIGHT * (zobs - zobs_pred)
    return -0.5 * dcz**2 / e2_cz - 0.5 * jnp.log(2 * np.pi * e2_cz)


def normal_logpdf(x, loc, scale):
    """Log of the normal probability density function."""
    return (-0.5 * ((x - loc) / scale)**2
            - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi))


def upper_truncated_normal_logpdf(x, loc, scale, xmax):
    """Log of the normal probability density function truncated at `xmax`."""
    # Need the absolute value just to avoid sometimes things going wrong,
    # but it should never occur that loc > xmax.
    norm = 0.5 * (1 + erf((jnp.abs(xmax - loc)) / (jnp.sqrt(2) * scale)))
    return normal_logpdf(x, loc, scale) - jnp.log(norm)


###############################################################################
#                            LOS interpolation                                #
###############################################################################


def interpolate_los(r, los, rgrid, method="cubic"):
    """
    Interpolate the LOS field at a given radial distance.

    Parameters
    ----------
    r : 1-dimensional array of shape `(n_gal, )`
        Radial distances at which to interpolate the LOS field.
    los : 3-dimensional array of shape `(n_sims, n_gal, n_steps)`
        LOS field.
    rmin, rmax : float
        Minimum and maximum radial distances in the data.
    order : int, optional
        The order of the interpolation. Default is 1, can be 0.

    Returns
    -------
    2-dimensional array of shape `(n_sims, n_gal)`
    """
    # Vectorize over the inner loop (ngal) first, then the outer loop (nsim)
    def f(rn, los_row):
        return interp1d(rn, rgrid, los_row, method=method)

    return vmap(vmap(f, in_axes=(0, 0)), in_axes=(None, 0))(r, los)


###############################################################################
#                          Base flow validation                               #
###############################################################################


class BaseFlowValidationModel(ABC):

    def _setattr_as_jax(self, names, values):
        for name, value in zip(names, values):
            setattr(self, f"{name}", jnp.asarray(value))

    def _set_calibration_params(self, calibration_params):
        names, values = [], []
        for key, value in calibration_params.items():
            names.append(key)
            values.append(value)

            # Store also the squared uncertainty
            if "e_" in key:
                key = key.replace("e_", "e2_")
                value = value**2
                names.append(key)
                values.append(value)

        self._setattr_as_jax(names, values)

    def _set_radial_spacing(self, r_xrange, Omega_m):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Omega_m)

        r_xrange = jnp.asarray(r_xrange)
        r2_xrange = r_xrange**2
        r2_xrange_mean = r2_xrange.mean()
        r2_xrange /= r2_xrange_mean

        self.r_xrange = r_xrange
        self.log_r2_xrange = jnp.log(r2_xrange)
        self.log_r2_xrange_mean = jnp.log(r2_xrange_mean)

        # Require `zmin` < 0 because the first radial step is likely at 0.
        z_xrange = z_at_value(
            cosmo.comoving_distance, r_xrange * u.Mpc, zmin=-0.01)
        mu_xrange = cosmo.distmod(z_xrange).value
        # In case the first distance is 0 and its distance modulus is infinite.
        if not np.isfinite(mu_xrange[0]):
            mu_xrange[0] = mu_xrange[1] - 1

        self.z_xrange = jnp.asarray(z_xrange)
        self.mu_xrange = jnp.asarray(mu_xrange)

    def _set_void_data(self, RA, dec, profile, kind, h, order):
        """Create the void interpolator."""
        # h is the MOND model value of local H0 to convert the radial grid
        # to Mpc / h
        rLG_grid, void_grid = load_void_data(profile, kind)
        void_grid = jnp.asarray(void_grid, dtype=jnp.float32)
        rLG_grid = jnp.asarray(rLG_grid, dtype=jnp.float32)

        rLG_grid *= h
        rLG_min, rLG_max = rLG_grid.min(), rLG_grid.max()
        rgrid_min, rgrid_max = 0, 250
        fprint(f"setting radial grid from {rLG_min} to {rLG_max} Mpc / h.")
        rgrid_max *= h

        # Get angular separation of each object from the model axis.
        phi = angular_distance_from_void_axis(RA, dec)
        phi = jnp.asarray(phi, dtype=jnp.float32)

        if kind == "density":
            void_grid = jnp.log(void_grid)
            self.void_log_rho_interpolator = lambda rLG: interpolate_void(
                rLG, self.r_xrange, phi, void_grid, rgrid_min, rgrid_max,
                rLG_min, rLG_max, order)
        elif kind == "vrad":
            self.void_vrad_interpolator = lambda rLG: interpolate_void(
                rLG, self.r_xrange, phi, void_grid, rgrid_min, rgrid_max,
                rLG_min, rLG_max, order)
        else:
            raise ValueError(f"Unknown kind: `{kind}`.")

    @property
    def ndata(self):
        """Number of PV objects in the catalogue."""
        return len(self.RA)

    @property
    def num_sims(self):
        """Number of simulations."""
        if self.is_void_data:
            return 1.

        return len(self.log_los_density())

    def los_density(self, **kwargs):
        if self.is_void_data:
            # Currently we have no densities for the void.
            # return jnp.ones((1, self.ndata, len(self.r_xrange)))
            raise NotImplementedError("Only log-density for the void.")

        return self._los_density

    def log_los_density(self, **kwargs):
        if self.is_void_data:
            # We want the shape to be `(1, n_objects, n_radial_steps)``.
            return self.void_log_rho_interpolator(kwargs["rLG"])[None, ...]

        return self._log_los_density

    def los_velocity(self, **kwargs):
        if self.is_void_data:
            # We want the shape to be `(1, n_objects, n_radial_steps)``.
            return self.void_vrad_interpolator(kwargs["rLG"])[None, ...]

        return self._los_velocity

    def log_los_density_at_r(self, r):
        return interpolate_los(r, self.log_los_density(), self.r_xrange, )

    def los_velocity_at_r(self, r):
        return interpolate_los(r, self.los_velocity(), self.r_xrange, )

    @abstractmethod
    def __call__(self, **kwargs):
        pass


###############################################################################
#                         Sampling shortcuts                                  #
###############################################################################

def sample_alpha_bias(name, xmin, xmax, to_sample):
    if to_sample:
        return sample(f"alpha_{name}", Uniform(xmin, xmax))
    else:
        return 1.0


###############################################################################
#                          SNIa parameters sampling                           #
###############################################################################


def distmod_SN(mag, x1, c, mag_cal, alpha_cal, beta_cal):
    """Distance modulus of a SALT2 SN Ia."""
    return mag - mag_cal + alpha_cal * x1 - beta_cal * c


def e2_distmod_SN(e2_mag, e2_x1, e2_c, alpha_cal, beta_cal, e_mu_intrinsic):
    """Squared error on the distance modulus of a SALT2 SN Ia."""
    return (e2_mag + alpha_cal**2 * e2_x1 + beta_cal**2 * e2_c
            + e_mu_intrinsic**2)


def sample_SN(e_mu_min, e_mu_max, mag_cal_mean, mag_cal_std, alpha_cal_mean,
              alpha_cal_std, beta_cal_mean, beta_cal_std, alpha_min, alpha_max,
              sample_alpha, name):
    """Sample SNIe Tripp parameters."""
    e_mu = sample(f"e_mu_{name}", Uniform(e_mu_min, e_mu_max))
    mag_cal = sample(f"mag_cal_{name}", Normal(mag_cal_mean, mag_cal_std))
    alpha_cal = sample(
        f"alpha_cal_{name}", Normal(alpha_cal_mean, alpha_cal_std))
    beta_cal = sample(f"beta_cal_{name}", Normal(beta_cal_mean, beta_cal_std))
    alpha = sample_alpha_bias(name, alpha_min, alpha_max, sample_alpha)

    return {"e_mu": e_mu,
            "mag_cal": mag_cal,
            "alpha_cal": alpha_cal,
            "beta_cal": beta_cal,
            "alpha": alpha
            }


###############################################################################
#                          Tully-Fisher parameters sampling                   #
###############################################################################

def distmod_TFR(mag, eta, a, b, c):
    """Distance modulus of a TFR calibration."""
    return mag - (a + b * eta + c * eta**2)


def e2_distmod_TFR(e2_mag, e2_eta, eta, b, c, e_mu_intrinsic):
    """
    Squared error on the TFR distance modulus with linearly propagated
    magnitude and linewidth uncertainties.
    """
    return e2_mag + (b + 2 * c * eta)**2 * e2_eta + e_mu_intrinsic**2


def sample_TFR(e_mu_min, e_mu_max, a_mean, a_std, b_mean, b_std,
               c_mean, c_std, alpha_min, alpha_max, sample_alpha,
               a_dipole_mean, a_dipole_std, sample_a_dipole,
               sample_curvature, name):
    """Sample Tully-Fisher calibration parameters."""
    e_mu = sample(f"e_mu_{name}", Uniform(e_mu_min, e_mu_max))
    a = sample(f"a_{name}", Normal(a_mean, a_std))

    if sample_a_dipole:
        ax, ay, az = sample(f"a_dipole_{name}", Normal(a_dipole_mean, a_dipole_std).expand([3]))  # noqa
    else:
        ax, ay, az = 0.0, 0.0, 0.0

    b = sample(f"b_{name}", Normal(b_mean, b_std))

    if sample_curvature:
        c = sample(f"c_{name}", Normal(c_mean, c_std))
    else:
        c = 0.

    alpha = sample_alpha_bias(name, alpha_min, alpha_max, sample_alpha)

    return {"e_mu": e_mu,
            "a": a,
            "ax": ax, "ay": ay, "az": az,
            "b": b,
            "c": c,
            "alpha": alpha,
            "sample_a_dipole": sample_a_dipole,
            }


###############################################################################
#                    Simple calibration parameters sampling                   #
###############################################################################

def sample_simple(e_mu_min, e_mu_max, dmu_min, dmu_max, alpha_min, alpha_max,
                  dmu_dipole_mean, dmu_dipole_std, sample_alpha,
                  sample_dmu_dipole, name):
    """Sample simple calibration parameters."""
    e_mu = sample(f"e_mu_{name}", Uniform(e_mu_min, e_mu_max))
    dmu = sample(f"dmu_{name}", Uniform(dmu_min, dmu_max))
    alpha = sample_alpha_bias(name, alpha_min, alpha_max, sample_alpha)

    if sample_dmu_dipole:
        dmux, dmuy, dmuz = sample(
            f"dmu_dipole_{name}",
            Normal(dmu_dipole_mean, dmu_dipole_std).expand([3]))
    else:
        dmux, dmuy, dmuz = 0.0, 0.0, 0.0

    return {"e_mu": e_mu,
            "dmu": dmu,
            "dmux": dmux, "dmuy": dmuy, "dmuz": dmuz,
            "alpha": alpha,
            "sample_dmu_dipole": sample_dmu_dipole,
            }

###############################################################################
#                    Calibration parameters sampling                          #
###############################################################################


def sample_calibration(Vext_min, Vext_max, Vmono_min, Vmono_max, beta_min,
                       beta_max, sigma_v_min, sigma_v_max, h_min, h_max,
                       rLG_min, rLG_max, no_Vext, sample_Vmono, sample_beta,
                       sample_h, sample_rLG, sample_Vmag_vax):
    """Sample the flow calibration."""
    sigma_v = sample("sigma_v", Uniform(sigma_v_min, sigma_v_max))

    if sample_beta:
        beta = sample("beta", Uniform(beta_min, beta_max))
    else:
        beta = 1.0

    if not no_Vext and sample_Vmag_vax:
        raise RuntimeError("Cannot sample Vext and Vext magnitude along the "
                           "void axis simultaneously.")

    if no_Vext:
        Vext = jnp.zeros(3)

        if sample_Vmag_vax:
            Vext_mag = sample("Vext_axis_mag", Uniform(Vext_min, Vext_max))
            # In the direction if (l, b) = (117, 4)
            Vext = Vext_mag * jnp.asarray([0.4035093, -0.01363162, 0.91487396])

    else:
        Vext = sample("Vext", Uniform(Vext_min, Vext_max).expand([3]))

    if sample_Vmono:
        Vmono = sample("Vmono", Uniform(Vmono_min, Vmono_max))
    else:
        Vmono = 0.0

    if sample_h:
        h = sample("h", Uniform(h_min, h_max))
    else:
        h = 1.0

    if sample_rLG:
        rLG = sample("rLG", Uniform(rLG_min, rLG_max))
    else:
        rLG = None

    return {"Vext": Vext,
            "Vmono": Vmono,
            "sigma_v": sigma_v,
            "beta": beta,
            "h": h,
            "sample_h": sample_h,
            "rLG": rLG,
            }


def sample_gaussian_hyperprior(param, name, xmin, xmax):
    """Sample MNR Gaussian hyperprior mean and standard deviation."""
    mean = sample(f"{param}_mean_{name}", Uniform(xmin, xmax))
    std = sample(f"{param}_std_{name}", Uniform(0.0, xmax - xmin))
    return mean, std


###############################################################################
#            PV calibration model without absolute calibration                #
###############################################################################


class PV_LogLikelihood(BaseFlowValidationModel):
    """
    Peculiar velocity validation model log-likelihood with numerical
    integration of the true distances.

    Parameters
    ----------
    los_density : 3-dimensional array of shape (n_sims, n_objects, n_steps)
        LOS density field. Set to `None` if the data is void data.
    los_velocity : 3-dimensional array of shape (n_sims, n_objects, n_steps)
        LOS radial velocity field. Set to `None` if the data is void data.
    RA, dec : 1-dimensional arrays of shape (n_objects)
        Right ascension and declination in degrees.
    z_obs : 1-dimensional array of shape (n_objects)
        Observed redshifts.
    e_zobs : 1-dimensional array of shape (n_objects)
        Errors on the observed redshifts.
    calibration_params : dict
        Calibration parameters of each object.
    mag_selection : dict
        Magnitude selection parameters, optional.
    r_xrange : 1-dimensional array
        Radial distances where the field was interpolated for each object.
    Omega_m : float
        Matter density parameter.
    kind : str
        Catalogue kind, either "TFR", "SN", or "simple".
    name : str
        Name of the catalogue.
    void_kwargs : dict, optional
        Void data parameters. If `None` the data is not void data.
    wo_num_dist_marginalisation : bool, optional
        Whether to directly sample the distance without numerical
        marginalisation. in which case the tracers can be coupled by a
        covariance matrix. By default `False`.
    with_homogeneous_malmquist : bool, optional
        Whether to include the homogeneous Malmquist bias. By default `True`.
    with_inhomogeneous_malmquist : bool, optional
        Whether to include the inhomogeneous Malmquist bias. By default `True`.
    """

    def __init__(self, los_density, los_velocity, RA, dec, z_obs, e_zobs,
                 calibration_params, mag_selection, r_xrange, Omega_m, kind,
                 name, void_kwargs=None, wo_num_dist_marginalisation=False,
                 with_homogeneous_malmquist=True,
                 with_inhomogeneous_malmquist=True):
        if e_zobs is not None:
            e2_cz_obs = jnp.asarray((SPEED_OF_LIGHT * e_zobs)**2)
        else:
            e2_cz_obs = jnp.zeros_like(z_obs)

        self.is_void_data = void_kwargs is not None

        # This must be done before we convert to radians.
        if void_kwargs is not None:
            self._set_void_data(RA=RA, dec=dec,  kind="density", **void_kwargs)
            self._set_void_data(RA=RA, dec=dec,  kind="vrad", **void_kwargs)

        # Convert RA/dec to radians.
        RA, dec = np.deg2rad(RA), np.deg2rad(dec)

        names = ["RA", "dec", "z_obs", "e2_cz_obs"]
        values = [RA, dec, z_obs, e2_cz_obs]

        # If ever start running out of memory, may be better not to store
        # both the density and log_density
        if not self.is_void_data:
            names += ["_log_los_density", "_los_velocity"]
            values += [jnp.log(los_density), los_velocity]

            # Density required only if not numerically marginalising.
            if not wo_num_dist_marginalisation:
                names += ["_los_density"]
                values += [los_density]

        self._setattr_as_jax(names, values)
        self._set_calibration_params(calibration_params)
        self._set_radial_spacing(r_xrange, Omega_m)

        self.kind = kind
        self.name = name
        self.Omega_m = Omega_m
        self.wo_num_dist_marginalisation = wo_num_dist_marginalisation
        self.with_homogeneous_malmquist = with_homogeneous_malmquist
        self.with_inhomogeneous_malmquist = with_inhomogeneous_malmquist
        self.norm = - self.ndata * jnp.log(self.num_sims)

        if mag_selection is not None:
            self.mag_selection_kind = mag_selection["kind"]

            if self.mag_selection_kind == "hard":
                self.mag_selection_max = mag_selection["coeffs"]
                fprint(f"catalogue {name} with selection mmax = {self.mag_selection_max}.")               # noqa
            elif self.mag_selection_kind == "soft":
                self.m1, self.m2, self.a = mag_selection["coeffs"]
                fprint(f"catalogue {name} with selection m1 = {self.m1}, m2 = {self.m2}, a = {self.a}.")  # noqa
                self.log_Fm = toy_log_magnitude_selection(
                    self.mag, self.m1, self.m2, self.a)
        else:
            self.mag_selection_kind = None

        if mag_selection is not None and kind != "TFR":
            raise ValueError("Magnitude selection is only implemented "
                             "for TFRs.")

        if kind == "TFR":
            self.mag_min, self.mag_max = jnp.min(self.mag), jnp.max(self.mag)
            eta_mu = jnp.mean(self.eta)
            fprint(f"setting the linewith mean to 0 instead of {eta_mu:.3f}.")
            self.eta -= eta_mu
            self.eta_min, self.eta_max = jnp.min(self.eta), jnp.max(self.eta)
        elif kind == "SN":
            self.mag_min, self.mag_max = jnp.min(self.mag), jnp.max(self.mag)
            self.x1_min, self.x1_max = jnp.min(self.x1), jnp.max(self.x1)
            self.c_min, self.c_max = jnp.min(self.c), jnp.max(self.c)
        elif kind == "simple":
            self.mu_min, self.mu_max = jnp.min(self.mu), jnp.max(self.mu)
        else:
            raise RuntimeError("Support most be added for other kinds.")

        if self.mag_selection_kind == "hard" and self.mag_selection_max > self.mag_max:  # noqa
            raise ValueError("The maximum magnitude cannot be larger than "
                             "the selection threshold.")

    def __call__(self, field_calibration_params, distmod_params,
                 inference_method):
        if inference_method not in ["mike", "bayes", "delta"]:
            raise ValueError(f"Unknown method: `{inference_method}`.")

        ll0 = 0.0
        sigma_v = field_calibration_params["sigma_v"]
        e2_cz = self.e2_cz_obs + sigma_v**2

        Vext = field_calibration_params["Vext"]
        Vmono = field_calibration_params["Vmono"]
        Vext_rad = project_Vext(Vext[0], Vext[1], Vext[2], self.RA, self.dec)

        e_mu = distmod_params["e_mu"]

        # Jeffrey's prior on sigma_v and the intrinsic scatter, they are above
        # "sampled" from uniform distributions.
        ll0 -= jnp.log(sigma_v) + jnp.log(e_mu)

        # ------------------------------------------------------------
        # 1. Sample true observables and obtain the distance estimate
        # ------------------------------------------------------------
        if self.kind == "SN":
            mag_cal = distmod_params["mag_cal"]
            alpha_cal = distmod_params["alpha_cal"]
            beta_cal = distmod_params["beta_cal"]

            if inference_method == "bayes":
                mag_mean, mag_std = sample_gaussian_hyperprior(
                    "mag", self.name, self.mag_min, self.mag_max)
                x1_mean, x1_std = sample_gaussian_hyperprior(
                    "x1", self.name, self.x1_min, self.x1_max)
                c_mean, c_std = sample_gaussian_hyperprior(
                    "c", self.name, self.c_min, self.c_max)

                # Jeffrey's prior on the the MNR hyperprior widths.
                ll0 -= jnp.log(mag_std) + jnp.log(x1_std) + jnp.log(c_std)

                # NOTE: that the true variables are currently uncorrelated.
                with plate(f"true_SN_{self.name}", self.ndata):
                    mag_true = sample(
                        f"mag_true_{self.name}", Normal(mag_mean, mag_std))
                    x1_true = sample(
                        f"x1_true_{self.name}", Normal(x1_mean, x1_std))
                    c_true = sample(
                        f"c_true_{self.name}", Normal(c_mean, c_std))

                # Log-likelihood of the observed magnitudes.
                if self.mag_selection_kind is None:
                    ll0 += jnp.sum(normal_logpdf(
                        mag_true, self.mag, self.e_mag))
                else:
                    raise NotImplementedError("Maxmag selection not implemented.")  # noqa

                # Log-likelihood of the observed x1 and c.
                ll0 += jnp.sum(normal_logpdf(x1_true, self.x1, self.e_x1))
                ll0 += jnp.sum(normal_logpdf(c_true, self.c, self.e_c))
                e2_mu = jnp.ones_like(mag_true) * e_mu**2
            else:
                mag_true = self.mag
                x1_true = self.x1
                c_true = self.c
                if inference_method == "mike":
                    e2_mu = e2_distmod_SN(
                        self.e2_mag, self.e2_x1, self.e2_c, alpha_cal,
                        beta_cal, e_mu)
                else:
                    e2_mu = jnp.ones_like(mag_true) * e_mu**2

            mu = distmod_SN(
                mag_true, x1_true, c_true, mag_cal, alpha_cal, beta_cal)
        elif self.kind == "TFR":
            a = distmod_params["a"]
            b = distmod_params["b"]
            c = distmod_params["c"]

            if distmod_params["sample_a_dipole"]:
                ax, ay, az = (distmod_params[k] for k in ["ax", "ay", "az"])
                a = a + project_Vext(ax, ay, az, self.RA, self.dec)

            if inference_method == "bayes":
                # Sample the true TFR parameters.
                mag_mean, mag_std = sample_gaussian_hyperprior(
                    "mag", self.name, self.mag_min, self.mag_max)
                eta_mean, eta_std = sample_gaussian_hyperprior(
                    "eta", self.name, self.eta_min, self.eta_max)
                corr_mag_eta = sample(
                    f"corr_mag_eta_{self.name}", Uniform(-1, 1))

                # Jeffrey's prior on the the MNR hyperprior widths.
                ll0 -= jnp.log(mag_std) + jnp.log(eta_std)

                loc = jnp.array([mag_mean, eta_mean])
                cov = jnp.array(
                    [[mag_std**2, corr_mag_eta * mag_std * eta_std],
                     [corr_mag_eta * mag_std * eta_std, eta_std**2]])

                with plate(f"true_TFR_{self.name}", self.ndata):
                    x_true = sample(
                        f"x_TFR_{self.name}", MultivariateNormal(loc, cov))

                mag_true, eta_true = x_true[..., 0], x_true[..., 1]
                # Log-likelihood of the observed magnitudes.
                if self.mag_selection_kind == "hard":
                    ll0 += jnp.sum(upper_truncated_normal_logpdf(
                        self.mag, mag_true, self.e_mag,
                        self.mag_selection_max))
                elif self.mag_selection_kind == "soft":
                    ll_mag = self.log_Fm
                    ll_mag += normal_logpdf(self.mag, mag_true, self.e_mag)

                    # Normalization per datapoint, initially (ndata, nxrange)
                    mu_start = mag_true - 5 * self.e_mag
                    mu_end = mag_true + 5 * self.e_mag
                    # 100 is a reasonable and sufficient choice.
                    mu_xrange = jnp.linspace(mu_start, mu_end, 100).T

                    norm = toy_log_magnitude_selection(
                        mu_xrange, self.m1, self.m2, self.a)
                    norm = norm + normal_logpdf(
                        mu_xrange, mag_true[:, None], self.e_mag[:, None])
                    # Now integrate over the magnitude range.
                    norm = simpson(jnp.exp(norm), x=mu_xrange, axis=-1)

                    ll0 += jnp.sum(ll_mag - jnp.log(norm))
                else:
                    ll0 += jnp.sum(normal_logpdf(
                        self.mag, mag_true, self.e_mag))

                # Log-likelihood of the observed linewidths.
                ll0 += jnp.sum(normal_logpdf(eta_true, self.eta, self.e_eta))

                e2_mu = jnp.ones_like(mag_true) * e_mu**2
            else:
                eta_true = self.eta
                mag_true = self.mag
                if inference_method == "mike":
                    e2_mu = e2_distmod_TFR(
                        self.e2_mag, self.e2_eta, eta_true, b, c, e_mu)
                else:
                    e2_mu = jnp.ones_like(mag_true) * e_mu**2

            mu = distmod_TFR(mag_true, eta_true, a, b, c)
        elif self.kind == "simple":
            dmu = distmod_params["dmu"]

            if distmod_params["sample_dmu_dipole"]:
                dmux, dmuy, dmuz = (
                    distmod_params[k] for k in ["dmux", "dmuy", "dmuz"])
                dmu = dmu + project_Vext(dmux, dmuy, dmuz, self.RA, self.dec)

            if inference_method == "bayes":
                raise NotImplementedError("Bayes for simple not implemented.")
            else:
                if inference_method == "mike":
                    e2_mu = e_mu**2 + self.e2_mu
                else:
                    e2_mu = jnp.ones_like(mag_true) * e_mu**2

            mu = self.mu + dmu
        else:
            raise ValueError(f"Unknown kind: `{self.kind}`.")

        # ----------------------------------------------------------------
        # 2. Log-likelihood of the true distance and observed redshifts.
        # The marginalisation of the true distance can be done numerically.
        # ----------------------------------------------------------------
        if not self.wo_num_dist_marginalisation:

            if field_calibration_params["sample_h"]:
                raise NotImplementedError(
                    "Sampling of 'h' is not supported if numerically "
                    "marginalising the true distance.")

            # Calculate p(r) (Malmquist bias). Shape is (ndata, nxrange)
            if self.with_homogeneous_malmquist:
                log_ptilde = log_ptilde_wo_bias(
                    self.mu_xrange[None, :], mu[:, None], e2_mu[:, None],
                    self.log_r2_xrange[None, :])
            else:
                log_ptilde = log_ptilde_wo_bias(
                    self.mu_xrange[None, :], mu[:, None], e2_mu[:, None],
                    0.)

            if self.is_void_data:
                rLG = field_calibration_params["rLG"]
                log_los_density = self.log_los_density(rLG=rLG)
                los_velocity = self.los_velocity(rLG=rLG)
            else:
                log_los_density = self.log_los_density()
                los_velocity = self.los_velocity()

            # Inhomogeneous Malmquist bias. Shape: (nsims, ndata, nxrange)
            alpha = distmod_params["alpha"]
            log_ptilde = log_ptilde[None, ...]
            if self.with_inhomogeneous_malmquist:
                log_ptilde += alpha * log_los_density

            ptilde = jnp.exp(log_ptilde)
            # Normalization of p(r). Shape: (nsims, ndata)
            pnorm = simpson(ptilde, x=self.r_xrange, axis=-1)

            # Calculate z_obs at each distance. Shape: (nsims, ndata, nxrange)
            vrad = field_calibration_params["beta"] * los_velocity
            vrad += (Vext_rad[None, :, None] + Vmono)
            zobs = 1 + self.z_xrange[None, None, :]
            zobs *= 1 + vrad / SPEED_OF_LIGHT
            zobs -= 1.

            # Shape remains (nsims, ndata, nxrange)
            ptilde *= likelihood_zobs(
                self.z_obs[None, :, None], zobs, e2_cz[None, :, None])

            # Integrate over the radial distance. Shape: (nsims, ndata)
            ll = jnp.log(simpson(ptilde, x=self.r_xrange, axis=-1))
            ll -= jnp.log(pnorm)

            return ll0 + jnp.sum(logsumexp(ll, axis=0)) + self.norm
        else:
            e_mu = jnp.sqrt(e2_mu)
            # True distance modulus, shape is `(n_data)`. If we have absolute
            # calibration, then this distance modulus assumes a particular h.
            with plate("plate_mu", self.ndata):
                mu_true = sample("mu", Normal(mu, e_mu))

            # Likelihood of the true distance modulii given the calibration.
            if field_calibration_params["sample_h"]:
                raise RuntimeError(
                    "Sampling of 'h' has not yet been thoroughly tested.")
                h = field_calibration_params["h"]

                # Now, the rest of the code except the calibration likelihood
                # uses the distance modulus in units of h
                mu_true_h = mu_true + 5 * jnp.log10(h)

                # Calculate the log-likelihood of the calibration, but the
                # shape is `(n_calibrators, n_data)`. Where there is no data
                # we set the likelihood to 0 (or the log-likelihood to -inf)
                ll_calibration = jnp.where(
                   self.is_finite_calibrator,
                   normal_logpdf(self.mu_calibration, mu_true[None, :],
                                 self.e_mu_calibration),
                   -jnp.inf)

                # Now average out over the calibrators, however only if the
                # there is at least one calibrator. If there isn't, then we
                # just assing a log-likelihood of 0.
                ll_calibration = jnp.where(
                    self.any_calibrator,
                    logsumexp(ll_calibration, axis=0) - jnp.log(self.counts_calibrators),  # noqa
                    0.)
            else:
                mu_true_h = mu_true

            # True distance and redshift, shape is `(n_data)`. The distance
            # here is in units of `Mpc / h``.
            r_true = distmod2dist(mu_true_h, self.Omega_m)
            z_true = distmod2redshift(mu_true_h, self.Omega_m)

            if self.is_void_data:
                raise NotImplementedError(
                    "Void data not implemented yet for distance sampling.")
            else:
                # Grid log(density), shape is `(n_sims, n_data, n_rad)`
                log_los_density_grid = self.log_los_density()
                # Densities and velocities at the true distances, shape is
                # `(n_sims, n_data)`
                log_density = self.log_los_density_at_r(r_true)
                los_velocity = self.los_velocity_at_r(r_true)

            alpha = distmod_params["alpha"]

            # Normalisation of p(mu), shape is `(n_sims, n_data, n_rad)`
            pnorm = normal_logpdf(
                self.mu_xrange[None, :], mu[:, None], e_mu[:, None])[None, ...]
            if self.with_homogeneous_malmquist:
                pnorm += self.log_r2_xrange[None, None, :]
            if self.with_inhomogeneous_malmquist:
                pnorm += alpha * log_los_density_grid

            pnorm = jnp.exp(pnorm)
            # Now integrate over the radial steps. Shape is `(nsims, ndata)`.
            # No Jacobian here because I integrate over distance, not the
            # distance modulus.
            pnorm = simpson(pnorm, x=self.r_xrange, axis=-1)

            # Jacobian |dr / dmu|_(mu_true_h), shape is `(n_data)`.
            jac = jnp.abs(distmod2dist_gradient(mu_true_h, self.Omega_m))

            # Calculate unnormalized log p(mu). Shape is (nsims, ndata)
            ll = 0.0
            if self.with_homogeneous_malmquist:
                ll += (+ jnp.log(jac)
                       + (2 * jnp.log(r_true) - self.log_r2_xrange_mean))
            if self.with_inhomogeneous_malmquist:
                ll += alpha * log_density

            # Subtract the normalization. Shape remains (nsims, ndata)
            ll -= jnp.log(pnorm)

            # Calculate z_obs at the true distance. Shape: (nsims, ndata)
            vrad = field_calibration_params["beta"] * los_velocity
            vrad += (Vext_rad[None, :] + Vmono)
            zobs = 1 + z_true[None, :]
            zobs *= 1 + vrad / SPEED_OF_LIGHT
            zobs -= 1.

            # Add the log-likelihood of observed redshifts. Shape remains
            # `(nsims, ndata)`
            ll += log_likelihood_zobs(
                self.z_obs[None, :], zobs, e2_cz[None, :])

            if field_calibration_params["sample_h"]:
                ll += ll_calibration[None, :]

            return ll0 + jnp.sum(logsumexp(ll, axis=0)) + self.norm


###############################################################################
#                      Combining several catalogues                           #
###############################################################################


def PV_validation_model(models, distmod_hyperparams_per_model,
                        field_calibration_hyperparams, inference_method):
    """
    Peculiar velocity validation NumPyro model.

    Parameters
    ----------
    models : list of `PV_LogLikelihood`
        List of PV validation log-likelihoods for each catalogue.
    distmod_hyperparams_per_model : list of dict
        Distance modulus hyperparameters for each model/catalogue.
    field_calibration_hyperparams : dict
        Field calibration hyperparameters.
    inference_method : str
        Either `mike` or `bayes`.
    """
    ll = 0.0

    field_calibration_params = sample_calibration(
        **field_calibration_hyperparams)

    # We sample the components of Vext with a uniform prior, which means
    # there is a |Vext|^2 prior, we correct for this so that the sampling
    # is effecitvely uniformly in magnitude of Vext and angles.
    if "Vext" in field_calibration_params and not field_calibration_hyperparams["no_Vext"]:  # noqa
        ll -= jnp.log(jnp.sum(field_calibration_params["Vext"]**2))

    for n in range(len(models)):
        model = models[n]
        name = model.name
        distmod_hyperparams = distmod_hyperparams_per_model[n]

        if model.kind == "TFR":
            distmod_params = sample_TFR(**distmod_hyperparams, name=name)
        elif model.kind == "SN":
            distmod_params = sample_SN(**distmod_hyperparams, name=name)
        elif model.kind == "simple":
            distmod_params = sample_simple(**distmod_hyperparams, name=name)
        else:
            raise ValueError(f"Unknown kind: `{model.kind}`.")

        ll += model(field_calibration_params, distmod_params, inference_method)

    factor("ll", ll)


###############################################################################
#                     Predicting z_cosmo from z_obs                           #
###############################################################################


def _posterior_element(r, beta, Vext_radial, los_velocity, Omega_m, zobs,
                       sigma_v, alpha, dVdOmega, los_density):
    """
    Helper function function to compute the unnormalized posterior in
    `Observed2CosmologicalRedshift`.
    """
    zobs_pred = predict_zobs(r, beta, Vext_radial, los_velocity, Omega_m)

    # Likelihood term
    dcz = SPEED_OF_LIGHT * (zobs - zobs_pred)
    posterior = jnp.exp(-0.5 * dcz**2 / sigma_v**2)
    posterior /= jnp.sqrt(2 * jnp.pi * sigma_v**2)

    # Prior term
    posterior *= dVdOmega * los_density**alpha

    return posterior


class BaseObserved2CosmologicalRedshift(ABC):
    """Base class for `Observed2CosmologicalRedshift`."""
    def __init__(self, calibration_samples, r_xrange):
        # Check calibration samples input.
        for i, key in enumerate(calibration_samples.keys()):
            x = calibration_samples[key]
            if not isinstance(x, (np.ndarray, jnp.ndarray)):
                raise ValueError(
                    f"Calibration sample `{key}` must be an array.")

            if x.ndim != 1 and key != "Vext":
                raise ValueError(f"Calibration samples `{key}` must be 1D.")

            if i == 0:
                ncalibratrion = len(x)

            if len(x) != ncalibratrion:
                raise ValueError(
                    "Calibration samples do not have the same length.")

            calibration_samples[key] = jnp.asarray(x)

        if "alpha" not in calibration_samples:
            print("No `alpha` calibration sample found. Setting it to 1.",
                  flush=True)
            calibration_samples["alpha"] = jnp.ones(ncalibratrion)

        if "beta" not in calibration_samples:
            print("No `beta` calibration sample found. Setting it to 1.",
                  flush=True)
            calibration_samples["beta"] = jnp.ones(ncalibratrion)

        # Get the stepsize, we need it to be constant for Simpson's rule.
        dr = np.diff(r_xrange)
        if not np.all(np.isclose(dr, dr[0], atol=1e-5)):
            raise ValueError("The radial step size must be constant.")
        dr = dr[0]

        self._calibration_samples = calibration_samples
        self._ncalibration_samples = ncalibratrion

        # It is best to JIT compile the functions right here.
        self._jit_posterior_element = jit(_posterior_element)

    def get_calibration_samples(self, key):
        """Get calibration samples for a given key."""
        if key not in self._calibration_samples:
            raise ValueError(f"Key `{key}` not found in calibration samples. "
                             f"Available keys are: `{self.calibration_keys}`.")

        return self._calibration_samples[key]

    @property
    def ncalibration_samples(self):
        """Number of calibration samples."""
        return self._ncalibration_samples

    @property
    def calibration_keys(self):
        """Calibration sample keys."""
        return list(self._calibration_samples.keys())


class Observed2CosmologicalRedshift(BaseObserved2CosmologicalRedshift):
    """
    Model to predict the cosmological redshift from the observed redshift in
    the CMB frame.

    Parameters
    ----------
    calibration_samples : dict
        Dictionary of flow calibration samples (`alpha`, `beta`, `Vext`,
        `sigma_v`, ...).
    r_xrange : 1-dimensional array
        Radial comoving distances where the fields are interpolated for each
        object.
    Omega_m : float
        Matter density parameter.
    """
    def __init__(self, calibration_samples, r_xrange, Omega_m):
        super().__init__(calibration_samples, r_xrange)
        self._r_xrange = jnp.asarray(r_xrange, dtype=jnp.float32)
        self._zcos_xrange = dist2redshift(self._r_xrange, Omega_m)
        self._Omega_m = Omega_m

        # Comoving volume element with some arbitrary normalization
        dVdOmega = gradient_redshift2dist(self._zcos_xrange, Omega_m)
        # TODO: Decide about the presence of this correction.
        dVdOmega *= self._r_xrange**2
        self._dVdOmega = dVdOmega / jnp.mean(dVdOmega)

    def posterior_mean_std(self, x, px):
        """
        Calculate the mean and standard deviation of a 1-dimensional PDF.
        """
        mu = simpson(x * px, x=x)
        std = (simpson(x**2 * px, x=x) - mu**2)**0.5
        return mu, std

    def posterior_zcosmo(self, zobs, RA, dec, los_density, los_velocity,
                         extra_sigma_v=None, verbose=True):
        """
        Calculate `p(z_cosmo | z_CMB, calibration)` for a single object.

        Parameters
        ----------
        zobs : float
            Observed redshift.
        RA, dec : float
            Right ascension and declination in radians.
        los_density : 1-dimensional array
            LOS density field.
        los_velocity : 1-dimensional array
            LOS radial velocity field.
        extra_sigma_v : float, optional
            Any additional velocity uncertainty.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        zcosmo : 1-dimensional array
            Cosmological redshift at which the PDF is evaluated.
        posterior : 1-dimensional array
            Posterior PDF.
        """
        Vext = self.get_calibration_samples("Vext")
        Vext_radial = project_Vext(*[Vext[:, i] for i in range(3)], RA, dec)

        alpha = self.get_calibration_samples("alpha")
        beta = self.get_calibration_samples("beta")
        sigma_v = self.get_calibration_samples("sigma_v")

        if extra_sigma_v is not None:
            sigma_v = jnp.sqrt(sigma_v**2 + extra_sigma_v**2)

        posterior = np.zeros((self.ncalibration_samples, len(self._r_xrange)),
                             dtype=np.float32)
        for i in trange(self.ncalibration_samples, desc="Marginalizing",
                        disable=not verbose):
            posterior[i] = self._jit_posterior_element(
                self._r_xrange, beta[i], Vext_radial[i], los_velocity,
                self._Omega_m, zobs, sigma_v[i], alpha[i], self._dVdOmega,
                los_density)

        # # Normalize the posterior for each flow sample and then stack them.
        posterior /= simpson(posterior, x=self._zcos_xrange, axis=-1)[:, None]
        posterior = jnp.nanmean(posterior, axis=0)

        return self._zcos_xrange, posterior


def stack_pzosmo_over_realizations(n, obs2cosmo_models, loaders, zobs_catname,
                                   pzcosmo_kwargs={}, verbose=True):
    """
    Stack the posterior PDFs of `z_cosmo` for a given galaxy index `n` over
    multiple constrained realizations.

    Parameters
    ----------
    n : int
        Galaxy index in the loaders' catalogue.
    obs2cosmo_models : list
        List of `Observed2CosmologicalRedshift` instances per realization.
    loaders : list
        List of DataLoader instances per realization.
    zobs_catname : str
        Name of the observed redshift column in the catalogue.
    pzcosmo_kwargs : dict, optional
        Additional keyword arguments to pass to `posterior_zcosmo`.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    zcosmo : 1-dimensional array
        Cosmological redshift at which the PDF is evaluated.
    p_zcosmo : 1-dimensional array
        Stacked posterior PDF.
    """
    # Do some standard checks of inputs
    if not isinstance(obs2cosmo_models, list):
        raise ValueError("`obs2cosmo_models` 1must be a list.")
    if not isinstance(loaders, list):
        raise ValueError("`loaders` must be a list.")
    if len(obs2cosmo_models) != len(loaders):
        raise ValueError("The number of models and loaders must be equal.")

    for i in trange(len(obs2cosmo_models), desc="Stacking",
                    disable=not verbose):
        zobs = loaders[i].cat[zobs_catname][n]
        RA = np.deg2rad(loaders[i].cat["RA"][n])
        dec = np.deg2rad(loaders[i].cat["DEC"][n])
        los_density = loaders[i].los_density[n]
        los_velocity = loaders[i].los_radial_velocity[n]

        x, y = obs2cosmo_models[i].posterior_zcosmo(
            zobs, RA, dec, los_density, los_velocity, verbose=False,
            **pzcosmo_kwargs)

        if i == 0:
            zcosmo = x
            p_zcosmo = np.empty((len(loaders), len(x)), dtype=np.float32)

        p_zcosmo[i] = y

    # Stack the posterior PDFs
    p_zcosmo = np.nanmean(p_zcosmo, axis=0)

    return zcosmo, p_zcosmo
