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
Script to run the PV validation model on various catalogues and simulations.
The script is not MPI parallelised, instead it is best run on a GPU.
"""
from argparse import ArgumentParser, ArgumentTypeError


def none_or_int(value):
    if value.lower() == "none":
        return None

    if "_" in value:
        args = value.split("_")
        if len(args) == 2:
            k0, kf = args
            dk = 1
        elif len(args) == 3:
            k0, kf, dk = args
        else:
            raise ArgumentTypeError(f"Invalid length of arguments: `{value}`.")

        return [int(k) for k in range(int(k0), int(kf), int(dk))]

    try:
        return int(value)
    except ValueError:
        raise ArgumentTypeError(f"Invalid value: {value}. Must be an integer or 'none'.")  # noqa


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, required=True,
                        help="Simulation name.")
    parser.add_argument("--catalogue", type=str, required=True,
                        help="PV catalogues.")
    parser.add_argument("--ksmooth", type=int, default=0,
                        help="Smoothing index.")
    parser.add_argument("--ksim", type=none_or_int, default=None,
                        help="IC iteration number. If 'None', all IC realizations are used.")  # noqa
    parser.add_argument("--ndevice", type=int, default=1,
                        help="Number of devices to request.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use.")
    args = parser.parse_args()

    # Convert the catalogue to a list of catalogues
    args.catalogue = args.catalogue.split(",")

    return args


ARGS = parse_args()
# This must be done before we import JAX etc.
from numpyro import set_host_device_count, set_platform                         # noqa

set_platform(ARGS.device)                                                       # noqa
set_host_device_count(ARGS.ndevice)                                             # noqa

import sys                                                                      # noqa
from os.path import join                                                        # noqa

import csiborgtools                                                             # noqa
import jax                                                                      # noqa
import numpy as np                                                              # noqa
from csiborgtools import fprint                                                 # noqa
from h5py import File                                                           # noqa
from numpyro.infer import MCMC, NUTS, init_to_median                            # noqa


def print_variables(names, variables):
    for name, variable in zip(names, variables):
        print(f"{name:<20} {variable}", flush=True)
    print(flush=True)


def get_models(ksim, get_model_kwargs, mag_selection, void_kwargs,
               wo_num_dist_marginalisation, verbose=True):
    """Load the data and create the NumPyro models."""
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    folder = "/mnt/extraspace/rstiskalek/catalogs/"

    nsims = paths.get_ics(ARGS.simname)
    if ksim is None:
        nsim_iterator = [i for i in range(len(nsims))]
    else:
        nsim_iterator = [ksim]
        nsims = [nsims[ksim]]

    if verbose:
        print(f"{'Simulation:':<20} {ARGS.simname}")
        print(f"{'Catalogue:':<20} {ARGS.catalogue}")
        print(f"{'Num. realisations:':<20} {len(nsims)}")
        print(flush=True)

    # Get models
    models = [None] * len(ARGS.catalogue)
    for i, cat in enumerate(ARGS.catalogue):
        if cat == "A2":
            fpath = join(folder, "A2.h5")
        elif cat in ["LOSS", "Foundation", "Pantheon+", "SFI_gals",
                     "2MTF", "SFI_groups", "SFI_gals_masked",
                     "Pantheon+_groups", "Pantheon+_groups_zSN",
                     "Pantheon+_zSN"]:
            fpath = join(folder, "PV_compilation.hdf5")
        elif "CF4_TFR" in cat:
            fpath = join(folder, "PV/CF4/CF4_TF-distances.hdf5")
        elif cat in ["CF4_GroupAll"]:
            fpath = join(folder, "PV/CF4/CF4_GroupAll.hdf5")
        elif "IndranilVoidTFRMock" in cat:
            fpath = None
        else:
            raise ValueError(f"Unsupported catalogue: `{ARGS.catalogue}`.")

        loader = csiborgtools.flow.DataLoader(ARGS.simname, nsim_iterator,
                                              cat, fpath, paths,
                                              ksmooth=ARGS.ksmooth)
        models[i] = csiborgtools.flow.get_model(
            loader, mag_selection=mag_selection[i], void_kwargs=void_kwargs,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation,
            **get_model_kwargs)

    fprint(f"num. radial steps is {len(loader.rdist)}")
    return models


def get_harmonic_evidence(samples, log_posterior, nchains_harmonic, epoch_num):
    """Compute evidence using the `harmonic` package."""
    data, names = csiborgtools.dict_samples_to_array(samples)
    data = data.reshape(nchains_harmonic, -1, len(names))
    log_posterior = log_posterior.reshape(nchains_harmonic, -1)

    return csiborgtools.harmonic_evidence(
        data, log_posterior, return_flow_samples=False, epochs_num=epoch_num)


def run_model(model, nsteps, nburn,  model_kwargs, out_folder,
              calculate_harmonic, nchains_harmonic, epoch_num, kwargs_print,
              fname_kwargs):
    """Run the NumPyro model and save output to a file."""
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    fname = paths.flow_validation(out_folder, ARGS.simname, ARGS.catalogue,
                                  **fname_kwargs)

    try:
        ndata = sum(model.ndata for model in model_kwargs["models"])
    except AttributeError as e:
        raise AttributeError("The models must have an attribute `ndata` "
                             "indicating the number of data points.") from e

    nuts_kernel = NUTS(model, init_strategy=init_to_median(num_samples=10000))
    mcmc = MCMC(nuts_kernel, num_warmup=nburn, num_samples=nsteps)
    rng_key = jax.random.PRNGKey(42)

    mcmc.run(rng_key, extra_fields=("potential_energy",), **model_kwargs)
    samples = mcmc.get_samples()

    log_posterior = -mcmc.get_extra_fields()["potential_energy"]
    BIC, AIC = csiborgtools.BIC_AIC(samples, log_posterior, ndata)
    print(f"{'BIC':<20} {BIC}")
    print(f"{'AIC':<20} {AIC}")
    mcmc.print_summary()

    if calculate_harmonic:
        print("Calculating the evidence using `harmonic`.", flush=True)
        neg_ln_evidence, neg_ln_evidence_err = get_harmonic_evidence(
            samples, log_posterior, nchains_harmonic, epoch_num)
        print(f"{'-ln(Z_h)':<20} {neg_ln_evidence}")
        print(f"{'-ln(Z_h) error':<20} {neg_ln_evidence_err}")
    else:
        neg_ln_evidence = jax.numpy.nan
        neg_ln_evidence_err = (jax.numpy.nan, jax.numpy.nan)

    fname = join(out_folder, fname)
    print(f"Saving results to `{fname}`.")
    with File(fname, "w") as f:
        # Write samples
        grp = f.create_group("samples")
        for key, value in samples.items():
            grp.create_dataset(key, data=value)

        # Write log likelihood and posterior
        f.create_dataset("log_posterior", data=log_posterior)

        # Write goodness of fit
        grp = f.create_group("gof")
        grp.create_dataset("BIC", data=BIC)
        grp.create_dataset("AIC", data=AIC)
        grp.create_dataset("neg_lnZ_harmonic", data=neg_ln_evidence)
        grp.create_dataset("neg_lnZ_harmonic_err", data=neg_ln_evidence_err)

    fname_summary = fname.replace(".hdf5", ".txt")
    print(f"Saving summary to `{fname_summary}`.")
    with open(fname_summary, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f

        print("User parameters:")
        for kwargs in kwargs_print:
            print_variables(kwargs.keys(), kwargs.values())

        print("HMC summary:")
        print(f"{'BIC':<20} {BIC}")
        print(f"{'AIC':<20} {AIC}")
        print(f"{'-ln(Z)':<20} {neg_ln_evidence}")
        print(f"{'-ln(Z) error':<20} {neg_ln_evidence_err}")
        mcmc.print_summary(exclude_deterministic=False)
        sys.stdout = original_stdout


###############################################################################
#                        Command line interface                               #
###############################################################################

def get_distmod_hyperparams(catalogue, sample_alpha, sample_mag_dipole):
    alpha_min = -10 if "IndranilVoid" in ARGS.simname else -1.0
    alpha_max = 10.0

    if catalogue in ["LOSS", "Foundation", "Pantheon+", "Pantheon+_groups", "Pantheon+_zSN"]:  # noqa
        return {"e_mu_min": 0.001, "e_mu_max": 1.0,
                "mag_cal_mean": -18.25, "mag_cal_std": 2.0,
                "alpha_cal_mean": 0.148, "alpha_cal_std": 1.0,
                "beta_cal_mean": 3.112, "beta_cal_std": 2.0,
                "alpha_min": alpha_min, "alpha_max": alpha_max,
                "sample_alpha": sample_alpha
                }
    elif catalogue in ["SFI_gals", "2MTF"] or "CF4_TFR" in catalogue or "IndranilVoidTFRMock" in catalogue:  # noqa
        return {"e_mu_min": 0.001, "e_mu_max": 1.0,
                "a_mean": -21., "a_std": 5.0,
                "b_mean": -5.95, "b_std": 4.0,
                "c_mean": 0., "c_std": 20.0,
                "a_dipole_mean": 0., "a_dipole_std": 1.0,
                "sample_a_dipole": sample_mag_dipole,
                "alpha_min": alpha_min, "alpha_max": alpha_max,
                "sample_alpha": sample_alpha,
                }
    elif catalogue in ["CF4_GroupAll"]:
        return {"e_mu_min": 0.001, "e_mu_max": 1.0,
                "dmu_min": -3.0, "dmu_max": 3.0,
                "dmu_dipole_mean": 0., "dmu_dipole_std": 1.0,
                "sample_dmu_dipole": sample_mag_dipole,
                "alpha_min": alpha_min, "alpha_max": alpha_max,
                "sample_alpha": sample_alpha,
                }
    else:
        raise ValueError(f"Unsupported catalogue: `{ARGS.catalogue}`.")


def get_toy_selection(catalogue):
    """Toy magnitude selection coefficients."""
    if catalogue == "SFI_gals":
        kind = "soft"
        # m1, m2, a
        coeffs = [11.467, 12.906, -0.231]
    elif "CF4_TFR" in catalogue and "_i" in catalogue:
        kind = "soft"
        coeffs = [13.043, 14.423, -0.129]
    elif "CF4_TFR" in catalogue and "w1" in catalogue:
        kind = "soft"
        coeffs = [11.731, 14.189, -0.118]
    elif catalogue == "2MTF":
        kind = "hard"
        coeffs = 11.25
    else:
        fprint(f"found no selection coefficients for {catalogue}.")
        return None

    return {"kind": kind,
            "coeffs": coeffs}


if __name__ == "__main__":
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    out_folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity"  # noqa
    print(f"{'Num. devices:':<20} {jax.device_count()}")
    print(f"{'Devices:':<20} {jax.devices()}")

    ###########################################################################
    #                        Fixed user parameters                            #
    ###########################################################################

    # `None` means default behaviour
    nsteps = 2_000
    nburn = 2_000
    zcmb_min = None
    zcmb_max = 0.05
    nchains_harmonic = 10
    num_epochs = 50
    inference_method = "mike"
    mag_selection = None
    sample_alpha = False if (ARGS.simname == "no_field" or "IndranilVoid" in ARGS.simname) else True  # noqa
    sample_beta = None
    no_Vext = None
    sample_Vmag_vax = False
    sample_Vmono = False
    sample_mag_dipole = False
    wo_num_dist_marginalisation = False
    absolute_calibration = None
    calculate_harmonic = (False if inference_method == "bayes" else True) and (not wo_num_dist_marginalisation)  # noqa
    sample_h = True if absolute_calibration is not None else False

    fname_kwargs = {"inference_method": inference_method,
                    "smooth": ARGS.ksmooth,
                    "nsim": ARGS.ksim,
                    "zcmb_min": zcmb_min,
                    "zcmb_max": zcmb_max,
                    "mag_selection": mag_selection,
                    "sample_alpha": sample_alpha,
                    "sample_beta": sample_beta,
                    "no_Vext": no_Vext,
                    "sample_Vmag_vax": sample_Vmag_vax,
                    "sample_Vmono": sample_Vmono,
                    "sample_mag_dipole": sample_mag_dipole,
                    "absolute_calibration": absolute_calibration,
                    }

    main_params = {"nsteps": nsteps, "nburn": nburn,
                   "zcmb_min": zcmb_min,
                   "zcmb_max": zcmb_max,
                   "mag_selection": mag_selection,
                   "calculate_harmonic": calculate_harmonic,
                   "nchains_harmonic": nchains_harmonic,
                   "num_epochs": num_epochs,
                   "inference_method": inference_method,
                   "sample_mag_dipole": sample_mag_dipole,
                   "wo_dist_marg": wo_num_dist_marginalisation,
                   "absolute_calibration": absolute_calibration,
                   "sample_h": sample_h,
                   }
    print_variables(main_params.keys(), main_params.values())

    if sample_beta is None:
        sample_beta = ARGS.simname == "Carrick2015"

    if mag_selection and inference_method != "bayes":
        raise ValueError("Magnitude selection is only supported with `bayes` inference.")   # noqa

    if "IndranilVoid" in ARGS.simname:
        if ARGS.ksim is not None:
            raise ValueError(
                "`IndranilVoid` does not have multiple realisations.")

        profile = ARGS.simname.split("_")[-1]
        h = csiborgtools.flow.select_void_h(profile)
        rdist = np.arange(0, 165, 0.5)
        void_kwargs = {"profile": profile, "h": h, "order": 1, "rdist": rdist}
    else:
        void_kwargs = None
        h = 1.

    if inference_method != "bayes":
        mag_selection = [None] * len(ARGS.catalogue)
    elif mag_selection is None or mag_selection:
        mag_selection = [get_toy_selection(cat) for cat in ARGS.catalogue]

    if nsteps % nchains_harmonic != 0:
        raise ValueError(
            "The number of steps must be divisible by the number of chains.")

    calibration_hyperparams = {"Vext_min": -3000, "Vext_max": 3000,
                               "Vmono_min": -1000, "Vmono_max": 1000,
                               "beta_min": -10.0, "beta_max": 10.0,
                               "sigma_v_min": 1.0, "sigma_v_max": 1000 if "IndranilVoid_" in ARGS.simname else 750.,  # noqa
                               "h_min": 0.01, "h_max": 5.0,
                               "no_Vext": False if no_Vext is None else no_Vext,        # noqa
                               "sample_Vmag_vax": sample_Vmag_vax,
                               "sample_Vmono": sample_Vmono,
                               "sample_beta": sample_beta,
                               "sample_h": sample_h,
                               "sample_rLG": "IndranilVoid" in ARGS.simname,
                               "rLG_min": 0.0, "rLG_max": 500 * h,
                               }
    print_variables(
        calibration_hyperparams.keys(), calibration_hyperparams.values())

    distmod_hyperparams_per_catalogue = []
    for cat in ARGS.catalogue:
        x = get_distmod_hyperparams(cat, sample_alpha, sample_mag_dipole)
        print(f"\n{cat} hyperparameters:")
        print_variables(x.keys(), x.values())
        distmod_hyperparams_per_catalogue.append(x)

    kwargs_print = (main_params, calibration_hyperparams,
                    *distmod_hyperparams_per_catalogue)

    ###########################################################################

    get_model_kwargs = {
        "zcmb_min": zcmb_min,
        "zcmb_max": zcmb_max,
        "absolute_calibration": absolute_calibration,
        "calibration_fpath": "/mnt/extraspace/rstiskalek/catalogs/PV/CF4/CF4_TF_calibration.hdf5",  # noqa
        }

    # In case we want to run multiple simulations independently.
    if not isinstance(ARGS.ksim, list):
        ksim_iterator = [ARGS.ksim]
    else:
        ksim_iterator = ARGS.ksim

    for i, ksim in enumerate(ksim_iterator):
        if len(ksim_iterator) > 1:
            print(f"{'Current simulation:':<20} {i + 1} ({ksim}) out of {len(ksim_iterator)}.")  # noqa

        fname_kwargs["nsim"] = ksim
        models = get_models(ksim, get_model_kwargs, mag_selection, void_kwargs,
                            wo_num_dist_marginalisation)
        model_kwargs = {
            "models": models,
            "field_calibration_hyperparams": calibration_hyperparams,
            "distmod_hyperparams_per_model": distmod_hyperparams_per_catalogue,
            "inference_method": inference_method,
            }

        model = csiborgtools.flow.PV_validation_model

        run_model(model, nsteps, nburn, model_kwargs, out_folder,
                  calculate_harmonic, nchains_harmonic, num_epochs,
                  kwargs_print, fname_kwargs)
