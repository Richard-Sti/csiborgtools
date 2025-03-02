# Copyright (C) 2023 Richard Stiskalek
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
from numpyro.distributions import Normal, Uniform
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
from scipy.stats import norm

SPEED_OF_LIGHT = 299_792.458


def dist2distmod(dist):
    return 5 * jnp.log10(dist) + 25


def dist2redshift(dist):
    H0 = 100
    return H0 * dist / SPEED_OF_LIGHT

###############################################################################
#                           Mock data generation                              #
###############################################################################


def generate_mock_data(injected_parameters, run_num, verbose=True,
                       make_plots=True):
    if verbose:
        print("Injected parameters:")
        print("--------------------")
        for key, value in injected_params.items():
            print(f"{key:15s}: {value:.6g}")
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

    # Distance and apparent magnitude
    dist = gen.uniform(
        injected_parameters["dist_min"],
        injected_parameters["dist_max"],
        ngal)

    distmod = dist2distmod(dist)

    mag_true = M + distmod
    mag_obs = gen.normal(mag_true, injected_parameters["e_mag"])

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
        "mag_obs": mag_obs,
        "dist": dist,
        "Vrad": Vrad,
        "dist": dist,
        "ztrue": ztrue,
        "zobs": zobs,
    }

    model_data = (eta_obs, mag_obs, phi, theta, zobs)

    return model_data, all_data


def model(obs_data, injected_params, sample_sigmaTFR, sample_sigma_v,
          sample_TFR):
    eta_obs, mag_obs, phi, theta, zobs = obs_data
    ngal = len(eta_obs)

    eta_mean = injected_params["eta_mean"]
    eta_std = injected_params["eta_std"]

    eta_mean_min = injected_params["eta_mean"] - 1
    eta_mean_max = injected_params["eta_mean"] + 1

    eta_mean = sample("eta_mean", Uniform(eta_mean_min, eta_mean_max))
    eta_std = sample("eta_std", Uniform(0, 3 * injected_params["eta_std"]))
    factor("ll_eta_std", -jnp.log(eta_std))

    with plate("plate_eta", ngal):
        eta_true = sample("eta_true", Normal(eta_mean, eta_std))
    factor(
        "ll_eta",
        Normal(eta_true, injected_params["e_eta"]).log_prob(eta_obs)
        )

    if sample_sigmaTFR:
        sigmaTFR = sample(
            "sigmaTFR", Uniform(0, 5 * injected_params["sigmaTFR"]))
        factor("ll_sigma_TFR", -jnp.log(sigmaTFR))
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

    with plate("plate_M", ngal):
        M = sample("M", Normal(aTFR + bTFR * eta_true, sigmaTFR))

    # It is important that the prior range here matches the range used to
    # generate the mock data.
    with plate("plate_dist", ngal):
        dist = sample(
            "dist",
            Uniform(injected_params["dist_min"], injected_params["dist_max"]))

    distmod = dist2distmod(dist)

    with plate("plate_mag", ngal):
        sample(
            "mag_true", Normal(M + distmod, injected_params["e_mag"]),
            obs=mag_obs)

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
        factor("ll_sigma_v", -jnp.log(sigma_v))
    else:
        sigma_v = injected_params["sigma_v"]

    with plate("plate_zobs", ngal):
        sample("zobs", Normal(zpred, sigma_v / SPEED_OF_LIGHT), obs=zobs)


def plot_corner(mcmc_samples, injected_params, params_to_plot, run_num,
                verbose=True):
    params_to_plot = [p for p in params_to_plot if p in mcmc_samples]
    samples = np.array([mcmc_samples[param] for param in params_to_plot]).T
    truths = [injected_params.get(param, None) for param in params_to_plot]

    # Generate the corner plot with truth values
    fig = corner(
        samples, labels=params_to_plot, show_titles=True, truths=truths,
        truth_color="red", title_kwargs={"fontsize": 12}, smooth=1)

    fname = f"./plots/run_{run_num}.png"
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

    nwarm, nsamp = 1500, 5000

    injected_params = {
        "ngal": 1000,
        "dist_min": 30,
        "dist_max": 150,

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
    }

    sample_sigmaTFR = True
    sample_sigma_v = True
    sample_TFR = True

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
                   "aTFR", "bTFR",]

    model_data, all_data = generate_mock_data(injected_params, args.run_num)

    rng_key = random.PRNGKey(args.run_num)
    kernel = NUTS(model, init_strategy=init_to_median(num_samples=1000))
    mcmc = MCMC(kernel, num_warmup=nwarm, num_samples=nsamp,)
    mcmc.run(rng_key, model_data, injected_params, sample_sigmaTFR,
             sample_sigma_v, sample_TFR)

    mcmc_samples = mcmc.get_samples()
    print(f"Sampled parameters are: {list(mcmc_samples.keys())}")
    for key in mcmc_samples:
        x = mcmc_samples[key]
        if x.ndim > 1:
            continue
        print(f"{key:20s}: {x.mean():.6g} ± {x.std():.6g}")

    plot_corner(mcmc_samples, injected_params, params_plot,
                run_num=args.run_num)

    plot_diff = {
        r"$\Delta M / \sigma_M$": (mcmc_samples["M"], all_data["M"], True, 1),
        r"$\Delta d / \sigma_d$": (mcmc_samples["dist"], all_data["dist"], True, 1),  # noqa
        r"$\Delta d$": (mcmc_samples["dist"], all_data["dist"], False, None),
    }

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
        fname = f"./plots/run_{args.run_num}_samples_diff.png"
        print(f"Saving the mock data plot to `{fname}`.")
        fig.savefig(fname, dpi=450)
        plt.close()

    fname = f"./data/run_{args.run_num}.hdf5"
    print(f"Saving the data to `{fname}`.")
    with File(fname, 'w') as f:
        grp = f.create_group("data")
        for key, value in all_data.items():
            grp.create_dataset(key, data=value)

        grp = f.create_group("samples")
        for key, value in mcmc_samples.items():
            grp.create_dataset(key, data=value)
