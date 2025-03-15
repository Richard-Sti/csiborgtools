# Copyright (C) 2025 Richard Stiskalek
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
A simple script to generate mock TFR data and forward-model it to assess any
potential biases.

Assumes a simple Hubble flow `cz = H0 * d`, that the luminosity distance is
equal to the comoving distance, and that the velocity field is that of a
constant dipole.
"""
from argparse import ArgumentParser
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from corner import corner
from h5py import File
from jax import numpy as jnp
from jax import random
from numpyro import factor, plate, sample
from numpyro.distributions import Normal, Uniform, TruncatedNormal
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
from tqdm import tqdm
from scipy.stats import norm
from quadax import cumulative_trapezoid
from h5py import File

from utils import ln_simpson
from jax.debug import print as jprint
import numpyro.distributions as dist
from numpyro.distributions.util import validate_sample

SPEED_OF_LIGHT = 299_792.458


def dist2distmod(dist):
    return 5 * jnp.log10(dist) + 25


def dist2redshift(dist):
    H0 = 100
    return H0 * dist / SPEED_OF_LIGHT


def distmod2dist(distmod):
    return 10**((distmod - 25) / 5)




###############################################################################
#                           x^2 distribution                                  #
###############################################################################


class SquaredLikeDistribution(dist.Distribution):
    """A distribution where the PDF is proportional to `x^2`."""
    reparametrized_params = ["xmin", "xmax"]
    support = dist.constraints.positive

    def __init__(self, xmin, xmax, validate_args=None):
        batch_shape, event_shape = (), ()
        self.xmin, self.xmax = xmin, xmax

        self.log_norm_const = jnp.log(3) - jnp.log(self.xmax**3 - self.xmin**3)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        u = random.uniform(key, shape=sample_shape)

        return jnp.cbrt(self.xmin**3 + u * (self.xmax**3 - self.xmin**3))

    @validate_sample
    def log_prob(self, value):
        """Compute log-probability, ensuring truncation."""
        return jnp.where(
            (value >= self.xmin) & (value <= self.xmax),
            2 * jnp.log(value) + self.log_norm_const,
            -jnp.inf)



def sample_mag_distance_mlim(gen, M, e_mag, mlim, dM=0.2):
    """
    Sample the distance and apparent magnitude for a given absolute magnitude
    and limiting apparent magnitude.
    """
    if isinstance(e_mag, (int, float)):
        e_mag = e_mag * jnp.ones_like(M)

    num_attempts_per_draw = np.zeros_like(M, dtype=int)
    mobs = np.zeros_like(M)
    mtrue = np.zeros_like(M)
    r = np.zeros_like(M)

    print("Sampling the distance and apparent magnitude.")
    for i, M_i in tqdm(enumerate(M), total=len(M), desc="Sampling"):
        # Rmax = 10**((mlim - M_i - 25 + 1) / 5)
        Rmax = 200
        n = 0

        while True:
            r_i = Rmax * np.cbrt(gen.uniform(0, 1))

            mtrue_i = M_i + dist2distmod(r_i)
            mobs_i = gen.normal(mtrue_i, e_mag[i])

            if mobs_i < mlim:
                num_attempts_per_draw[i] = n
                break

            n += 1

        num_attempts_per_draw[i] = n
        mobs[i] = mobs_i
        mtrue[i] = mtrue_i
        r[i] = r_i

    print(f"Finished sampling. Average number of attempts per draw "
          f"is {num_attempts_per_draw.mean():.2f}.")

    return mobs, mtrue, r


def key2label(key):
    x = {
        "aTFR": r"$a_{\rm TFR}$",
        "bTFR": r"$b_{\rm TFR}$",
        "sigmaTFR": r"$\sigma_{\rm TFR}$",
        "Vdip_mag": r"$V_{\rm dip}$",
        "Vdip_ra": r"$\phi_{\rm dip}$",
        "Vdip_cos_theta": r"$\cos \theta_{\rm dip}$",
        "sigma_v": r"$\sigma_v$",
        "eta_mean": r"$\mu_{\eta}$",
        "eta_std": r"$w_{\eta}$",
        "m1": r"$m_1$",
        "m2": r"$m_2$",
        "a": r"$a$",
        }

    if key in x:
        return x[key]
    return key


###############################################################################
#                           Mock data generation                              #
###############################################################################


def generate_mock_data(injected_parameters, mag_selection, run_num,
                       verbose=True, make_plots=True):
    if verbose:
        print("Injected parameters:")
        print("--------------------")
        for key, value in injected_params.items():
            print(f"{key:15s}: {value:.6g}")
        print(f"{'mag_selection':15s}: {mag_selection}")
        print()

    gen = np.random.default_rng(run_num)
    ngal = injected_parameters["ngal"]

    # Linewidth and absolute magnitude
    eta_true = gen.normal(
        injected_parameters["eta_mean"], injected_parameters["eta_std"],
        injected_parameters["ngal"])
    eta_obs = gen.normal(eta_true, injected_parameters["e_eta"])

    M = gen.normal(
        injected_parameters["aTFR"] + injected_parameters["bTFR"] * eta_true,
        injected_parameters["sigmaTFR"])

    # Distance from a truncated r^2 distribution and apparent magnitude
    if mag_selection == "mlim":
        mag_obs, mag_true, dist = sample_mag_distance_mlim(
            gen, M, injected_parameters["e_mag"], injected_parameters["mlim"])
    else:
        raise ValueError(f"Unknown magnitude selection: {mag_selection}")

    distmod = dist2distmod(dist)

    # Sky position and peculiar velocity
    phi = gen.uniform(0, 2 * np.pi, ngal)
    theta = np.pi / 2 - np.arcsin(gen.uniform(-1, 1, ngal))

    Vdip_theta = np.arccos(injected_parameters["Vdip_cos_theta"])
    Vdip_phi = injected_parameters["Vdip_ra"]
    Vdip_mag = injected_parameters["Vdip_mag"]

    Vrad = Vdip_mag * (
        + np.sin(Vdip_theta) * np.sin(theta) * np.cos(Vdip_phi - phi)
        + np.cos(Vdip_theta) * np.cos(theta)
        )

    # Observed redshift
    ztrue = (1 + dist2redshift(dist)) * (1 + Vrad / SPEED_OF_LIGHT) - 1
    zobs = gen.normal(ztrue, injected_parameters["sigma_v"] / SPEED_OF_LIGHT)

    if make_plots:
        plot_data = {
            r"$\eta_{\rm true}$": eta_true,
            r"$M$": M,
            r"$r ~ [\mathrm{Mpc} / h]$": dist,
            r"$\mu$": distmod,
            r"$m_{\rm obs}$": mag_obs,
            r"$z_{\rm obs}$": zobs,
            r"$c(z_{\rm obs} - z_{\rm true}) ~ [\mathrm{km} / \mathrm{s}]$": SPEED_OF_LIGHT * (zobs - ztrue),  # noqa
        }

        num_plots = len(plot_data)
        cols = 3
        rows = ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten()

        for i, (label, data) in enumerate(plot_data.items()):
            ax = axes[i]
            ax.hist(data, bins="auto", alpha=0.7, edgecolor='black',)
            ax.set_xlabel(label)
            ax.set_ylabel("Binned counts")

        # Hide unused subplots (if any)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        fname = f"./plots/run_{run_num}_mock_data.png"
        if verbose:
            print(f"Saving the mock data plot to `{fname}`.")
        fig.savefig(fname, dpi=450)
        plt.close()

    all_data = {
        "eta_true": eta_true,
        "eta_obs": eta_obs,
        "M": M,
        "phi": phi,
        "theta": theta,
        "mag_true": mag_true,
        "mag_obs": mag_obs,
        "dist": dist,
        "Vrad": Vrad,
        "dist": dist,
        "ztrue": ztrue,
        "zobs": zobs,
    }

    if isinstance(injected_params["e_mag"], (int, float)):
        e_mag = injected_params["e_mag"] * jnp.ones_like(M)
    else:
        e_mag = injected_params["e_mag"]

    if isinstance(injected_params["e_eta"], (int, float)):
        e_eta = injected_params["e_eta"] * jnp.ones_like(eta_obs)
    else:
        e_eta = injected_params["e_eta"]

    model_data = (eta_obs, mag_obs, phi, theta, zobs, e_eta, e_mag)

    return model_data, all_data


def read_data(name, make_plots=True, verbose=True):
    if name == "2MTF":
        fname = "/mnt/extraspace/rstiskalek/catalogs/PV_compilation.hdf5"

        with File(fname, 'r') as f:
            grp = f["2MTF"]
            ra = grp["RA"][...]
            dec = grp["DEC"][...]

            zobs = grp["z_CMB"][...]
            eta = grp["eta"][...]
            e_eta = grp["e_eta"][...]
            mag = grp["mag"][...]
            e_mag = grp["e_mag"][...]

        eta -= np.mean(eta)

        theta = np.pi / 2 - np.radians(dec)
        phi = np.radians(ra)
    else:
        raise ValueError(f"Unknown data set: {name}")

    if make_plots:
        plot_data = {
            r"$\theta$": theta,
            r"$\phi$": phi,
            r"$z_{\rm obs}$": zobs,
            r"$\eta_{\rm obs}$": eta,
            r"$m_{\rm obs}$": mag,
        }

        num_plots = len(plot_data)
        cols = 3
        rows = ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten()

        for i, (label, data) in enumerate(plot_data.items()):
            ax = axes[i]
            ax.hist(data, bins="auto", alpha=0.7, edgecolor='black',)
            ax.set_xlabel(label)
            ax.set_ylabel("Binned counts")

        # Hide unused subplots (if any)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        fname = f"./plots/{name}_data.png"
        if verbose:
            print(f"Saving the mock data plot to `{fname}`.")
        fig.savefig(fname, dpi=450)
        plt.close()

    return eta, mag, phi, theta, zobs, e_eta, e_mag


###############################################################################
#                     Forward model including distance sampling               #
###############################################################################


def interpolate_mag_smooth_norm(M, m1, m2, a, rmax, num_dist_steps_marg,
                                num_absmag_steps_marg):
    # Get a range of radial distances.
    dist_xrange = jnp.linspace(0.001, rmax, num_dist_steps_marg)
    distmod_xrange = dist2distmod(dist_xrange)
    # Similarly, get a range of absolute magnitudes.
    M_xrange = jnp.linspace(M.min(), M.max(), num_absmag_steps_marg)

    # Compute the normalisation as a function of the absolute magnitude,
    # initially the shape is `(num_absmag_steps_marg, num_dist_steps_marg)`
    # but then we marginalise over the distance steps.
    norm_per_M = ln_mag_selection(
        M_xrange[:, None] + distmod_xrange[None, :],
        m1, m2, a,
        )
    norm_per_M += 2 * jnp.log(dist_xrange)[None, :]
    norm_per_M = ln_simpson(norm_per_M, x=dist_xrange[None, :], axis=-1)

    # Interpolate the normalisation to the absolute magnitudes of the galaxies.
    return jnp.interp(M, M_xrange, norm_per_M)


def sample_distance_smooth(M, m1, m2, a, rmax, num_dist_steps_marg,
                           num_absmag_steps_marg):
    # Sample the distance for each galaxy uniformly, and only later compute the
    # log-probablity of the distance given the absolute magnitude.
    with plate("plate_dist", len(M)):
        dist = sample("dist", Uniform(0, rmax))

    distmod = dist2distmod(dist)

    lp_r = ln_mag_selection(M + distmod, m1, m2, a) + 2 * jnp.log(dist)
    lp_r -= interpolate_mag_smooth_norm(
        M, m1, m2, a, rmax, num_dist_steps_marg, num_absmag_steps_marg)
    factor("ll_mag", lp_r)

    return dist, distmod


def model_sample_dist(model_kind, obs_data, injected_params, sample_sigmaTFR,
                      sample_sigma_v, sample_TFR, mag_selection, data_kwargs):
    eta_obs, mag_obs, phi, theta, zobs, e_eta, e_mag = obs_data
    ngal = len(eta_obs)

    eta_mean = injected_params["eta_mean"]
    eta_std = injected_params["eta_std"]

    eta_mean_min = injected_params["eta_mean"] - 1
    eta_mean_max = injected_params["eta_mean"] + 1

    eta_mean = sample("eta_mean", Uniform(eta_mean_min, eta_mean_max))
    eta_std = sample("eta_std", Uniform(0, 3 * injected_params["eta_std"]))
    # factor("ll_eta_std", -jnp.log(eta_std))

    if sample_sigmaTFR:
        sigmaTFR = sample(
            "sigmaTFR", Uniform(0.1, 5 * injected_params["sigmaTFR"]))
        # factor("ll_sigma_TFR", -jnp.log(sigmaTFR))
    else:
        sigmaTFR = injected_params["sigmaTFR"]

    if sample_TFR:
        aTFR_min = injected_params["aTFR"] - 3
        aTFR_max = injected_params["aTFR"] + 3
        aTFR = sample("aTFR", Uniform(aTFR_min, aTFR_max))

        bTFR_min = injected_params["bTFR"] - 3
        bTFR_max = injected_params["bTFR"] + 3
        bTFR = sample("bTFR", Uniform(bTFR_min, bTFR_max))
    else:
        aTFR, bTFR = injected_params["aTFR"], injected_params["bTFR"]

    with plate("plate_eta", ngal):
        eta_true = sample("xtrue_eta", Normal(eta_mean, eta_std))
        sample("ll_eta", Normal(eta_true, e_eta), obs=eta_obs)

    with plate("plate_M", ngal):
        M = sample("xtrue_M", Normal(aTFR + bTFR * eta_true, sigmaTFR))

    if mag_selection == "mlim":
        Rmax = 200
        # with plate("plate_dist", len(M)):
        #     dist = sample("xtrue_dist", Uniform(0, Rmax))
        # factor("ll_dist", 2 * jnp.log(dist))
        with plate("plate_dist", len(M)):
            dist = sample("xtrue_dist", SquaredLikeDistribution(0, 200))

        distmod = dist2distmod(dist)

        with plate("plate_mag", ngal):
            sample(
                "mag_obs", TruncatedNormal(
                    M + distmod, e_mag, high=injected_params["mlim"]),
                obs=mag_obs)
            # sample("mag_obs", Normal(M + distmod, e_mag,), obs=mag_obs)
    else:
        raise ValueError(f"Unknown magnitude selection: {mag_selection}")

    Vdip_mag = sample("Vdip_mag", Uniform(0, 10 * injected_params["Vdip_mag"]))
    Vdip_ra = sample("Vdip_ra", Uniform(0, 2 * np.pi))
    Vdip_cos_theta = sample("Vdip_cos_theta", Uniform(-1, 1))
    Vdip_theta = jnp.arccos(Vdip_cos_theta)

    Vpec = Vdip_mag * (
        + jnp.sin(Vdip_theta) * jnp.sin(theta) * jnp.cos(Vdip_ra - phi)
        + jnp.cos(Vdip_theta) * jnp.cos(theta))

    zpred = (1 + dist2redshift(dist)) * (1 + Vpec / SPEED_OF_LIGHT) - 1

    if sample_sigma_v:
        sigma_v = sample("sigma_v", Uniform(0, 5 * injected_params["sigma_v"]))
        # factor("ll_sigma_v", -jnp.log(sigma_v))
    else:
        sigma_v = injected_params["sigma_v"]

    with plate("plate_zobs", ngal):
        sample("zobs", Normal(zpred, sigma_v / SPEED_OF_LIGHT), obs=zobs)


# def model_sample_dist(model_kind, obs_data, injected_params, sample_sigmaTFR,
#                       sample_sigma_v, sample_TFR, mag_selection, data_kwargs):
#     eta_obs, mag_obs, phi, theta, zobs, e_eta, e_mag = obs_data
#     ngal = len(eta_obs)

#     if mag_selection == "smooth":
#         m2 = sample("m2", Uniform(12, 15))
#         m1 = sample("m1", Uniform(10, m2))
#         a = sample("a", Uniform(-0.5, 0.0))
#         m1 = injected_params["m1"]
#         m2 = injected_params["m2"]
#         a = injected_params["a"]

#     # The delta-prior models don't need these hyperpriors.
#     if "eta_delta" not in model_kind:
#         eta_mean = injected_params["eta_mean"]
#         eta_std = injected_params["eta_std"]

#         eta_mean_min = injected_params["eta_mean"] - 1
#         eta_mean_max = injected_params["eta_mean"] + 1

#         eta_mean = sample("eta_mean", Uniform(eta_mean_min, eta_mean_max))
#         eta_std = sample("eta_std", Uniform(0, 3 * injected_params["eta_std"]))
#         factor("ll_eta_std", -jnp.log(eta_std))

#     if sample_sigmaTFR:
#         sigmaTFR = sample(
#             "sigmaTFR", Uniform(0, 5 * injected_params["sigmaTFR"]))
#         factor("ll_sigma_TFR", -jnp.log(sigmaTFR))
#     else:
#         sigmaTFR = injected_params["sigmaTFR"]

#     if sample_TFR:
#         aTFR_min = injected_params["aTFR"] - 3
#         aTFR_max = injected_params["aTFR"] + 3
#         aTFR = sample("aTFR", Uniform(aTFR_min, aTFR_max))

#         bTFR_min = injected_params["bTFR"] - 3
#         bTFR_max = injected_params["bTFR"] + 3
#         bTFR = sample("bTFR", Uniform(bTFR_min, bTFR_max))
#     else:
#         aTFR, bTFR = injected_params["aTFR"], injected_params["bTFR"]

#     if model_kind in ["full", "M_marginalised"]:
#         with plate("plate_eta", ngal):
#             eta_true = sample("xtrue_eta", Normal(eta_mean, eta_std))
#         factor("ll_eta", Normal(eta_true, e_eta).log_prob(eta_obs))

#     if model_kind == "full":
#         with plate("plate_M", ngal):
#             M = sample("xtrue_M", Normal(aTFR + bTFR * eta_true, sigmaTFR))

#         if mag_selection == "mlim":
#             dist = sample_distance_mlim(M, injected_params["mlim"])
#             distmod = dist2distmod(dist)

#             with plate("plate_mag", ngal):
#                 sample(
#                     "mag_obs", TruncatedNormal(
#                         M + distmod, e_mag, high=injected_params["mlim"]),
#                     obs=mag_obs)
#         elif mag_selection == "smooth":
#             dist, distmod = sample_distance_smooth(
#                 M, m1, m2, a, data_kwargs["rmax"],
#                 data_kwargs["num_dist_steps_marg"],
#                 data_kwargs["num_absmag_steps_marg"])

#             with plate("plate_mag", ngal):
#                 sample("mag_obs", Normal(M + distmod, e_mag), obs=mag_obs)
#         else:
#             raise ValueError(f"Unknown magnitude selection: {mag_selection}")

#     elif model_kind == "eta_delta":
#         sigma = jnp.sqrt(sigmaTFR**2 + (bTFR * e_eta)**2)

#         with plate("plate_M", ngal):
#             M = sample("M", Normal(aTFR + bTFR * eta_obs, sigma))

#         if mag_selection == "mlim":
#             dist = sample_distance_mlim(M, injected_params["mlim"])
#             distmod = dist2distmod(dist)

#             with plate("plate_mag", ngal):
#                 sample(
#                     "mag_obs", TruncatedNormal(
#                         M + distmod, e_mag,
#                         high=injected_params["mlim"]), obs=mag_obs)
#         elif mag_selection == "smooth":
#             dist, distmod = sample_distance_smooth(
#                 M, m1, m2,
#                 a, data_kwargs["rmax"],
#                 data_kwargs["num_dist_steps_marg"],
#                 data_kwargs["num_absmag_steps_marg"])

#             with plate("plate_mag", ngal):
#                 sample(
#                     "mag_obs", Normal(M + distmod, e_mag), obs=mag_obs)
#         else:
#             raise ValueError(f"Unknown magnitude selection: {mag_selection}")

#     elif model_kind == "eta_delta_M_delta":
#         # This should be set to some relatively large value with a uniform
#         # prior.
#         with plate("dist_plate", ngal):
#             dist = sample("dist", Uniform(0, data_kwargs["rmax"]))

#         distmod = dist2distmod(dist)
#         M = mag_obs - distmod

#         sigma = jnp.sqrt(sigmaTFR**2 + (bTFR * e_eta)**2 + e_mag**2)

#         factor("ll_TFR", Normal(aTFR + bTFR * eta_obs, sigma).log_prob(M))

#         if mag_selection == "mlim":
#             Rmax = 10**((injected_params["mlim"] - M - 25) / 5)
#             factor("ll_dist",
#                    jnp.log(3) - 3 * jnp.log(Rmax) + 2 * jnp.log(dist))

#         elif mag_selection == "smooth":
#             ll_dist = ln_mag_selection(
#                 M + distmod, m1, m2,
#                 a) + 2 * jnp.log(dist)
#             ll_dist -= interpolate_mag_smooth_norm(
#                 M, m1, m2,
#                 a, data_kwargs["rmax"],
#                 data_kwargs["num_dist_steps_marg"],
#                 data_kwargs["num_absmag_steps_marg"])
#             factor("ll_dist", ll_dist)
#         else:
#             raise NotImplementedError(
#                 "Smooth selection not implemented for `eta_delta_M_delta.")
#     else:
#         raise ValueError(f"Unknown model kind: {model_kind}")

#     Vdip_mag = sample("Vdip_mag", Uniform(0, 10 * injected_params["Vdip_mag"]))
#     Vdip_ra = sample("Vdip_ra", Uniform(0, 2 * np.pi))
#     Vdip_cos_theta = sample("Vdip_cos_theta", Uniform(-1, 1))
#     Vdip_theta = jnp.arccos(Vdip_cos_theta)

#     Vpec = Vdip_mag * (
#         + jnp.sin(Vdip_theta) * jnp.sin(theta) * jnp.cos(Vdip_ra - phi)
#         + jnp.cos(Vdip_theta) * jnp.cos(theta))

#     zpred = (1 + dist2redshift(dist)) * (1 + Vpec / SPEED_OF_LIGHT) - 1

#     if sample_sigma_v:
#         sigma_v = sample("sigma_v", Uniform(0, 5 * injected_params["sigma_v"]))
#         factor("ll_sigma_v", -jnp.log(sigma_v))
#     else:
#         sigma_v = injected_params["sigma_v"]

#     with plate("plate_zobs", ngal):
#         sample("zobs", Normal(zpred, sigma_v / SPEED_OF_LIGHT), obs=zobs)

###############################################################################
#                     Forward model including distance sampling               #
###############################################################################


# def model_marg_dist(model_kind, obs_data, injected_params, sample_sigmaTFR,
#                     sample_sigma_v, sample_TFR, num_dist_steps, ):
#     eta_obs, mag_obs, phi, theta, zobs = obs_data
#     ngal = len(eta_obs)
#     raise NotImplementedError("model_marg_dist not implemented yet.")

#     eta_mean = injected_params["eta_mean"]
#     eta_std = injected_params["eta_std"]

#     eta_mean_min = injected_params["eta_mean"] - 1
#     eta_mean_max = injected_params["eta_mean"] + 1

#     eta_mean = sample("eta_mean", Uniform(eta_mean_min, eta_mean_max))
#     eta_std = sample("eta_std", Uniform(0, 3 * injected_params["eta_std"]))
#     factor("ll_eta_std", -jnp.log(eta_std))

#     if sample_sigmaTFR:
#         sigmaTFR = sample(
#             "sigmaTFR", Uniform(0, 5 * injected_params["sigmaTFR"]))
#         factor("ll_sigma_TFR", -jnp.log(sigmaTFR))
#     else:
#         sigmaTFR = injected_params["sigmaTFR"]

#     if sample_TFR:
#         aTFR_min = injected_params["aTFR"] - 3
#         aTFR_max = injected_params["aTFR"] + 3
#         aTFR = sample("aTFR", Uniform(aTFR_min, aTFR_max))

#         bTFR_min = injected_params["bTFR"] - 3
#         bTFR_max = injected_params["bTFR"] + 3
#         bTFR = sample("bTFR", Uniform(bTFR_min, bTFR_max))
#     else:
#         aTFR, bTFR = injected_params["aTFR"], injected_params["bTFR"]

#     # It is important that the prior range here matches the range used to
#     # generate the mock data. We add a small number to the minimum distance
#     # to avoid log(0).
#     dist_range = jnp.linspace(
#         1e-3, injected_params["dist_max_marg"], num_dist_steps)

#     distmod_range = dist2distmod(dist_range)

#     if model_kind in ["full", "M_marginalised"]:
#         with plate("plate_eta", ngal):
#             eta_true = sample("eta_true", Normal(eta_mean, eta_std))
#         factor(
#             "ll_eta",
#             Normal(eta_true, injected_params["e_eta"]).log_prob(eta_obs)
#             )

#     if model_kind == "full":
#         with plate("plate_M", ngal):
#             M = sample("M", Normal(aTFR + bTFR * eta_true, sigmaTFR))

#         # Log-likelihood of shape `(ngal, num_dist_steps)``
#         ll = Normal(
#             M[:, None] + distmod_range[None, :],
#             injected_params["e_mag"]).log_prob(mag_obs[:, None])
#     elif model_kind == "M_marginalised":
#         M = aTFR + bTFR * eta_true
#         sigma = jnp.sqrt(sigmaTFR**2 + injected_params["e_mag"]**2)

#         # Log-likelihood of shape `(ngal, num_dist_steps)``
#         ll = Normal(
#             M[:, None] + distmod_range[None, :],
#             sigma).log_prob(mag_obs[:, None])
#     elif model_kind == "M_eta_marginalised":
#         Sigma2 = sigmaTFR**2 + injected_params["e_mag"]**2
#         alpha = aTFR - mag_obs[:, None] + distmod_range[None, :]
#         eta_var = eta_std**2

#         e2_eta = injected_params["e_eta"]**2

#         SigmaTot2 = (
#             Sigma2 * e2_eta + eta_var * (Sigma2 + bTFR**2 * e2_eta))

#         ll = - (
#             + eta_var * (alpha + bTFR * eta_obs[:, None])**2
#             + Sigma2 * (eta_obs[:, None] - eta_mean)**2
#             + e2_eta * (alpha + bTFR * eta_mean)**2
#             ) / (2 * SigmaTot2)
#         ll -= 0.5 * jnp.log(SigmaTot2)
#     else:
#         raise ValueError(f"Unknown model kind: {model_kind}")

#     # Malmquist bias
#     ll += 2 * jnp.log(dist_range)[None, :]

#     Vdip_mag = sample("Vdip_mag", Uniform(0, 10 * injected_params["Vdip_mag"]))
#     Vdip_ra = sample("Vdip_ra", Uniform(0, 2 * np.pi))
#     Vdip_cos_theta = sample("Vdip_cos_theta", Uniform(-1, 1))
#     Vdip_theta = jnp.arccos(Vdip_cos_theta)

#     Vpec = Vdip_mag * (
#         + jnp.sin(Vdip_theta) * jnp.sin(theta) * jnp.cos(Vdip_ra - phi)
#         + jnp.cos(Vdip_theta) * jnp.cos(theta))

#     # Predicted redshift of shape `(ngal, num_dist_steps)`
#     zpred = (1 + dist2redshift(dist_range)[None, :]) * (1 + Vpec[:, None] / SPEED_OF_LIGHT) - 1  # noqa

#     if sample_sigma_v:
#         sigma_v = sample("sigma_v", Uniform(0, 5 * injected_params["sigma_v"]))
#         factor("ll_sigma_v", -jnp.log(sigma_v))
#     else:
#         sigma_v = injected_params["sigma_v"]

#     # Add the log-likelihood of the observed redshifts, shape remains
#     # `(ngal, num_dist_steps)`
#     ll += Normal(zpred, sigma_v / SPEED_OF_LIGHT).log_prob(zobs[:, None])

#     # Marginalise over the distance, shape `(ngal,)`
#     ll = ln_simpson(ll, x=dist_range[None, :], axis=-1)
#     factor("ll_zobs", ll)


###############################################################################
#                           Corner plot                                       #
###############################################################################


def plot_corner(mcmc_samples, injected_params, params_to_plot, run_num, kind,
                sample_distance, verbose=True):
    params_to_plot = [p for p in params_to_plot if p in mcmc_samples]
    samples = np.array([mcmc_samples[param] for param in params_to_plot]).T
    labels = [key2label(param) for param in params_to_plot]
    if injected_params is not None:
        truths = [injected_params.get(param, None) for param in params_to_plot]
    else:
        truths = None

    # Generate the corner plot with truth values
    fig = corner(
        samples, labels=labels, show_titles=True, truths=truths,
        truth_color="red", title_kwargs={"fontsize": 12}, smooth=1)

    fname = f"./plots/run_{run_num}_{kind}_corner.png"
    if sample_distance:
        fname = fname.replace(f"{kind}", f"distsample_{kind}")
    if verbose:
        print(f"Saving the corner plot to `{fname}`.")
    fig.savefig(fname, dpi=450)

    plt.close()


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("run_num", type=int, help="Run number")
    args = argparser.parse_args()

    Path("./plots").mkdir(parents=True, exist_ok=True,)
    Path("./data").mkdir(parents=True, exist_ok=True,)

    nwarm, nsamp = 1500, 4500
    # nwarm, nsamp = 500, 500
    save_samples = False
    sample_distance = True
    mag_selection = "mlim"
    dist_spacing = 0.5
    data_name = None
    print(f"Running {nwarm} warmup and {nsamp} sampling steps.")
    if sample_distance:
        print("Sampling distances.")
    else:
        print(f"Numerically marginalising over distances with a step size "
              f"of {dist_spacing} Mpc / h.")

    injected_params = {
        "ngal": 500,

        "mlim": 13.5,
        "m1": 10.92,
        "m2": 13.47,
        "a": -0.12,

        "aTFR": -20,
        "bTFR": -7,
        "sigmaTFR": 0.2,

        "Vdip_mag": 250,
        "Vdip_ra": 5 / 4 * np.pi,
        "Vdip_cos_theta": 0.3,
        "sigma_v": 250,

        "eta_mean": 0.0,
        "eta_std": 0.08,
        "e_eta": 0.025,

        "e_mag": 0.05,

        "dist_max_marg": 750,
    }

    data_kwargs = {
        "rmax": injected_params["dist_max_marg"],
        "num_dist_steps_marg": 1000,
        "num_absmag_steps_marg": 250,
    }

    kind = "full"
    sample_sigmaTFR = True
    sample_sigma_v = True
    sample_TFR = True
    num_dist_steps = int(injected_params["dist_max_marg"] / dist_spacing)

    if not sample_distance:
        print(f"We have {num_dist_steps} distance steps.")

    print(f"Running the model `{kind}`.")

    print()
    if sample_sigmaTFR:
        print("Sampling sigma_TFR.")
    else:
        print("Fixing sigma_TFR to the injected value.")

    if sample_sigma_v:
        print("Sampling sigma_v.")
    else:
        print("Fixing sigma_v to the injected value.")

    if sample_TFR:
        print("Sampling aTFR, bTFR.")
    else:
        print("Fixing aTFR, bTFR to the injected values.")
    print()

    params_plot = ["Vdip_mag", "Vdip_ra", "Vdip_cos_theta",
                   "sigma_v", "eta_mean", "eta_std", "sigmaTFR",
                   "aTFR", "bTFR", "m1", "m2", "a"]

    if data_name is None:
        model_data, all_data = generate_mock_data(
            injected_params, mag_selection, args.run_num)
    else:
        model_data = read_data(data_name)
        injected_params["ngal"] = len(model_data[0])

    rng_key = random.PRNGKey(args.run_num)
    model_args = (kind, model_data, injected_params, sample_sigmaTFR,
                  sample_sigma_v, sample_TFR, mag_selection, data_kwargs)
    if sample_distance:
        model = model_sample_dist
    else:
        model_args += (num_dist_steps,)
        model = model_marg_dist

    # print("Quitting...")
    # quit()

    kernel = NUTS(model, init_strategy=init_to_median(num_samples=5))
    mcmc = MCMC(kernel, num_warmup=nwarm, num_samples=nsamp,)

    print(f"Running the MCMC for the model `{kind}`.")
    mcmc.run(rng_key, *model_args)
    mcmc.print_summary()

    mcmc_samples = mcmc.get_samples()
    print(f"Sampled parameters are: {list(mcmc_samples.keys())}")
    for key in mcmc_samples:
        x = mcmc_samples[key]
        if x.ndim > 1:
            continue
        print(f"{key:20s}: {x.mean():.6g} ± {x.std():.6g}")

    plot_corner(mcmc_samples, injected_params, params_plot,
                run_num=args.run_num if data_name is None else data_name,
                kind=kind, sample_distance=sample_distance)

    plot_diff = {
        # r"$\Delta d / \sigma_d$": (mcmc_samples["dist"], all_data["dist"], True, 1),  # noqa
        # r"$\Delta d$": (mcmc_samples["dist"], all_data["dist"], False, None),
    }

    if kind == "full" and data_name is None:
        if "M" in mcmc_samples:
            plot_diff[r"$\Delta M / \sigma_M$"] = (mcmc_samples["M"], all_data["M"], True, 1)  # noqa

    if len(plot_diff) > 0:
        cols = 3
        rows = ceil(len(plot_diff) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = axes.flatten()

        for i, (label, (samples, true, norm_bias, bias_std)) in enumerate(plot_diff.items()):  # noqa
            ax = axes[i]
            mu, std = np.mean(samples, axis=0), np.std(samples, axis=0)
            if norm_bias:
                dx = (mu - true) / std
            else:
                dx = mu - true

            ax.hist(dx, bins="auto", alpha=0.7, edgecolor='black', density=1)
            if bias_std is not None:
                x = np.linspace(dx.min(), dx.max(), 1000)
                ax.plot(x, norm(0, bias_std).pdf(x), color="k", )
            ax.axvline(0, color="red", linestyle="--")
            ax.set_xlabel(label)
            ax.set_ylabel("Binned counts")

        # Hide unused subplots (if any)
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        fname = f"./plots/run_{args.run_num}_{kind}_diff.png"
        if sample_distance:
            fname = fname.replace(f"{kind}", f"distsample_{kind}")

        print(f"Saving the mock data plot to `{fname}`.")
        fig.savefig(fname, dpi=450)
        plt.close()

    if save_samples:
        fname = f"./data/run_{args.run_num}_{kind}.hdf5"
        if sample_distance:
            fname = fname.replace(f"{kind}", f"distsample_{kind}")
        print(f"Saving the data to `{fname}`.")
        with File(fname, 'w') as f:
            grp = f.create_group("data")
            for key, value in all_data.items():
                grp.create_dataset(key, data=value)

            grp = f.create_group("samples")
            for key, value in mcmc_samples.items():
                grp.create_dataset(key, data=value)
