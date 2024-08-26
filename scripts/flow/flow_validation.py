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
    parser.add_argument("--ksmooth", type=int, default=1,
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
from h5py import File                                                           # noqa
from numpyro.infer import MCMC, NUTS, init_to_median                            # noqa


def print_variables(names, variables):
    for name, variable in zip(names, variables):
        print(f"{name:<20} {variable}", flush=True)
    print(flush=True)


def get_models(get_model_kwargs, toy_selection, verbose=True):
    """Load the data and create the NumPyro models."""
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    folder = "/mnt/extraspace/rstiskalek/catalogs/"

    nsims = paths.get_ics(ARGS.simname)
    if ARGS.ksim is None:
        nsim_iterator = [i for i in range(len(nsims))]
    else:
        nsim_iterator = [ARGS.ksim]
        nsims = [nsims[ARGS.ksim]]

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
        else:
            raise ValueError(f"Unsupported catalogue: `{ARGS.catalogue}`.")

        loader = csiborgtools.flow.DataLoader(ARGS.simname, nsim_iterator,
                                              cat, fpath, paths,
                                              ksmooth=ARGS.ksmooth)
        models[i] = csiborgtools.flow.get_model(
            loader, toy_selection=toy_selection[i], **get_model_kwargs)

    print(f"\n{'Num. radial steps':<20} {len(loader.rdist)}\n", flush=True)
    return models


def get_harmonic_evidence(samples, log_posterior, nchains_harmonic, epoch_num):
    """Compute evidence using the `harmonic` package."""
    data, names = csiborgtools.dict_samples_to_array(samples)
    data = data.reshape(nchains_harmonic, -1, len(names))
    log_posterior = log_posterior.reshape(nchains_harmonic, -1)

    return csiborgtools.harmonic_evidence(
        data, log_posterior, return_flow_samples=False, epochs_num=epoch_num)


def run_model(model, nsteps, nburn,  model_kwargs, out_folder, sample_beta,
              calculate_harmonic, nchains_harmonic, epoch_num, kwargs_print):
    """Run the NumPyro model and save output to a file."""
    try:
        ndata = sum(model.ndata for model in model_kwargs["models"])
    except AttributeError as e:
        raise AttributeError("The models must have an attribute `ndata` "
                             "indicating the number of data points.") from e

    nuts_kernel = NUTS(model, init_strategy=init_to_median(num_samples=1000))
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

    fname = f"samples_{ARGS.simname}_{'+'.join(ARGS.catalogue)}_ksmooth{ARGS.ksmooth}.hdf5"  # noqa
    if ARGS.ksim is not None:
        fname = fname.replace(".hdf5", f"_nsim{ARGS.ksim}.hdf5")

    if sample_beta:
        fname = fname.replace(".hdf5", "_sample_beta.hdf5")

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
    alpha_min = -1.0
    alpha_max = 3.0

    if catalogue in ["LOSS", "Foundation", "Pantheon+", "Pantheon+_groups", "Pantheon+_zSN"]:  # noqa
        return {"e_mu_min": 0.001, "e_mu_max": 1.0,
                "mag_cal_mean": -18.25, "mag_cal_std": 2.0,
                "alpha_cal_mean": 0.148, "alpha_cal_std": 1.0,
                "beta_cal_mean": 3.112, "beta_cal_std": 2.0,
                "alpha_min": alpha_min, "alpha_max": alpha_max,
                "sample_alpha": sample_alpha
                }
    elif catalogue in ["SFI_gals", "2MTF"] or "CF4_TFR" in catalogue:
        return {"e_mu_min": 0.001, "e_mu_max": 1.0,
                "a_mean": -21., "a_std": 5.0,
                "b_mean": -5.95, "b_std": 4.0,
                "c_mean": 0., "c_std": 20.0,
                "sample_curvature": False,
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


def get_toy_selection(toy_selection, catalogue):
    if not toy_selection:
        return None

    if catalogue == "SFI_gals":
        return [1.221e+01, 1.297e+01, -2.708e-01]
    else:
        raise ValueError(f"Unsupported catalogue: `{ARGS.catalogue}`.")


if __name__ == "__main__":
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    out_folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity"  # noqa
    print(f"{'Num. devices:':<20} {jax.device_count()}")
    print(f"{'Devices:':<20} {jax.devices()}")

    ###########################################################################
    #                        Fixed user parameters                            #
    ###########################################################################

    nsteps = 1000
    nburn = 500
    zcmb_min = 0
    zcmb_max = 0.05
    nchains_harmonic = 10
    num_epochs = 50
    inference_method = "bayes"
    calculate_harmonic = True if inference_method == "mike" else False
    maxmag_selection = None
    sample_alpha = False
    sample_beta = True
    sample_Vmono = False
    sample_mag_dipole = False
    toy_selection = True

    if toy_selection and inference_method == "mike":
        raise ValueError("Toy selection is not supported with `mike` inference.")  # noqa

    if nsteps % nchains_harmonic != 0:
        raise ValueError(
            "The number of steps must be divisible by the number of chains.")

    main_params = {"nsteps": nsteps, "nburn": nburn,
                   "zcmb_min": zcmb_min,
                   "zcmb_max": zcmb_max,
                   "maxmag_selection": maxmag_selection,
                   "calculate_harmonic": calculate_harmonic,
                   "nchains_harmonic": nchains_harmonic,
                   "num_epochs": num_epochs,
                   "inference_method": inference_method,
                   "sample_mag_dipole": sample_mag_dipole,
                   "toy_selection": toy_selection}
    print_variables(main_params.keys(), main_params.values())

    calibration_hyperparams = {"Vext_min": -1000, "Vext_max": 1000,
                               "Vmono_min": -1000, "Vmono_max": 1000,
                               "beta_min": -1.0, "beta_max": 3.0,
                               "sigma_v_min": 1.0, "sigma_v_max": 750.,
                               "sample_Vmono": sample_Vmono,
                               "sample_beta": sample_beta,
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

    get_model_kwargs = {"zcmb_min": zcmb_min, "zcmb_max": zcmb_max,
                        "maxmag_selection": maxmag_selection}

    toy_selection = [get_toy_selection(toy_selection, cat)
                     for cat in ARGS.catalogue]

    models = get_models(get_model_kwargs, toy_selection)
    model_kwargs = {
        "models": models,
        "field_calibration_hyperparams": calibration_hyperparams,
        "distmod_hyperparams_per_model": distmod_hyperparams_per_catalogue,
        "inference_method": inference_method,
        }

    model = csiborgtools.flow.PV_validation_model

    run_model(model, nsteps, nburn, model_kwargs, out_folder,
              calibration_hyperparams["sample_beta"], calculate_harmonic,
              nchains_harmonic, num_epochs, kwargs_print)
