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
    parser.add_argument(
        "--ksim", type=none_or_int, default=None,
        help="IC iteration number. If 'None', all IC realizations are used.")
    parser.add_argument("--ndevice", type=int, default=1,
                        help="Number of devices to request.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use.")
    parser.add_argument(
        "--aux_name", type=str, default="",
        help="Auxiliary argument name to overwrite any choices.")
    parser.add_argument(
        "--aux_arg", type=str, default="",
        help="Auxiliary argument to overwrite any choices.")
    parser.add_argument(
        "--aux_type", type=str, default="str", choices=["str", "int", "float"],
        help="Auxiliary argument type to overwrite any choices.")

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
from interpax import Interpolator1D                                             # noqa
from numpyro.infer import (MCMC, NUTS, init_to_median)                          # noqa


def print_variables(names, variables):
    for name, variable in zip(names, variables):
        print(f"{name:<20} {variable}", flush=True)
    print(flush=True)


def get_models(ksim, get_model_kwargs, mag_selection, void_kwargs,
               wo_num_dist_marginalisation, verbose=True):
    """Load the data and create the NumPyro models."""
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    folder = "/mnt/extraspace/rstiskalek/catalogs/"

    nsims = paths.get_ics(ARGS.simname, subsample=True)
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
        elif cat in ["LOSS", "Foundation", "SFI_gals", "2MTF", "SFI_groups",
                     "SFI_gals_masked"]:
            fpath = join(folder, "PV_compilation.hdf5")
        elif "Pantheon+" in cat:
            fpath = join(folder, "PV", "Pantheon+SH0ES.dat")
        elif "Carrick2MTFmock" in cat:
            ki = cat.split("_")[-1]
            fpath =f"/mnt/extraspace/rstiskalek/csiborg_postprocessing/flow_mock/Carrick2MTFmock_seed{ki}.hdf5"  # noqa
        elif "CF4_TFR" in cat:
            fpath = join(folder, "PV/CF4/CF4_TFR.hdf5")
        elif cat in ["CF4_GroupAll"]:
            fpath = join(folder, "PV/CF4/CF4_GroupAll.hdf5")
        elif "IndranilVoidTFRMock" in cat:
            fpath = None
        elif cat in ["SDSS-FP"]:
            fpath = join(folder, "PV/CF4/SDSS-FP-LOWZ.hdf5")
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
    data, names = csiborgtools.dict_samples_to_array(
        samples, exclude_deterministic=True)
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

    nuts_kernel = NUTS(model,
                       init_strategy=init_to_median(num_samples=1000),
                       )
    mcmc = MCMC(nuts_kernel, num_warmup=nburn, num_samples=nsteps)
    rng_key = jax.random.PRNGKey(42)

    mcmc.run(rng_key, extra_fields=("potential_energy",), **model_kwargs)
    samples = mcmc.get_samples()

    log_posterior = -mcmc.get_extra_fields()["potential_energy"]
    BIC, AIC = csiborgtools.BIC_AIC(samples, log_posterior, ndata)
    print(f"{'BIC':<20} {BIC}")
    print(f"{'AIC':<20} {AIC}")
    mcmc.print_summary(exclude_deterministic=False)

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
    print(f"Saving results: `{fname}`.")
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

    fname_config = fname.replace(".hdf5", "_config.txt")
    print(f"Saving configuration: `{fname_config}`.")
    with open(fname_config, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f

        print("User parameters:")
        for kwargs in kwargs_print:
            print_variables(kwargs.keys(), kwargs.values())
        sys.stdout = original_stdout

    fname_summary = fname.replace(".hdf5", "_summary.txt")
    print(f"Saving summary: `{fname_summary}`.")
    with open(fname_summary, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f

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

def get_distmod_hyperparams(catalogue, sample_alpha, sample_mag_dipole,
                            dust_model, Rdust_fixed):
    alpha_min = -10 if "IndranilVoid" in ARGS.simname else -1.0
    alpha_max = 10.0

    if catalogue in ["LOSS", "Foundation"]:
        return {"e_mu_min": 0.005, "e_mu_max": 1.0,
                "mag_cal_mean": -18.25, "mag_cal_std": 2.0,
                "alpha_cal_mean": 0.148, "alpha_cal_std": 1.0,
                "beta_cal_mean": 3.112, "beta_cal_std": 2.0,
                "alpha_min": alpha_min, "alpha_max": alpha_max,
                "sample_alpha": sample_alpha
                }
    elif catalogue in ["Pantheon+", "Pantheon+_groups", "Pantheon+_zSN"]:
        return {"e_mu_min": 0.001, "e_mu_max": 1.0,
                "mag_cal_mean": -18.5, "mag_cal_std": 2.0,
                "alpha_mean": 1.0, "alpha_std": 0.5,
                "sample_alpha": sample_alpha
                }
    elif catalogue in ["SFI_gals", "2MTF"] or "CF4_TFR" in catalogue or "IndranilVoidTFRMock" in catalogue or "Carrick2MTFmock" in catalogue:  # noqa
        return {"e_mu_min": 0.005, "e_mu_max": 1.0,
                "a_mean": -22.0, "a_std": 5.0,
                "b_mean": -7.0, "b_std": 4.0,
                "c_mean": 0., "c_std": 20.0,
                "a_dipole_mean": 0., "a_dipole_std": 1.0,
                "sample_a_dipole": sample_mag_dipole,
                "alpha_min": alpha_min, "alpha_max": alpha_max,
                "sample_alpha": sample_alpha,
                "sample_curvature": False if "Carrick2MTFmock" in catalogue else True,  # noqa
                "Rdust_min": 0,
                "Rdust_max": 1.0,
                "Rdust_fixed": Rdust_fixed,
                }
    elif catalogue in ["CF4_GroupAll"]:
        return {"e_mu_min": 0.005, "e_mu_max": 1.0,
                "dmu_min": -3.0, "dmu_max": 3.0,
                "dmu_dipole_mean": 0., "dmu_dipole_std": 1.0,
                "sample_dmu_dipole": sample_mag_dipole,
                "alpha_min": alpha_min, "alpha_max": alpha_max,
                "sample_alpha": sample_alpha,
                }
    elif catalogue in ["SDSS-FP"]:
        return {"e_mu_min": 0.005, "e_mu_max": 10.0,
                "a_mean": 0.0, "a_std": 2.0,
                "b_mean": 0.0, "b_std": 2.0,
                "c_mean": 0.0, "c_std": 2.0,
                "alpha_min": alpha_min, "alpha_max": alpha_max,
                "sample_alpha": sample_alpha}
    else:
        raise ValueError(f"Unsupported catalogue: `{ARGS.catalogue}`.")


def get_toy_selection(catalogue):
    """Toy magnitude selection coefficients."""
    if catalogue == "SFI_gals":
        mag_kind = "soft"
        # m1, m2, a
        mag_coeffs = [11.602, 12.948, -0.233]
        eta_coeffs = [None, None]
        eta_kind = None
    elif "CF4_TFR" in catalogue and "_i" in catalogue:
        mag_kind = "soft"
        mag_coeffs = [12.010, 13.879, -0.158]
        eta_coeffs = [-0.3, None]
        eta_kind = "lower_hard"
    elif "CF4_TFR" in catalogue and "w1" in catalogue:
        mag_kind = "soft"
        mag_coeffs = [10.921, 13.471, -0.118]
        eta_kind = "lower_hard"
        eta_coeffs = [-0.3, None]
    elif "CF4_TFR" in catalogue and "w2" in catalogue:
        raise RuntimeError("Need to calculate W2 coefficients.")
        # mag_kind = "soft"
        # mag_coeffs = [10.921, 13.471, -0.118]
        # eta_kind = "lower_hard"
        # eta_coeffs = [-0.3, None]
    elif catalogue == "2MTF":
        mag_kind = "hard"
        mag_coeffs = 11.25
        eta_coeffs = [-0.1, 0.2]
        eta_kind = "hard"
    else:
        fprint(f"found no selection coefficients for {catalogue}.")
        return None

    return {"mag_kind": mag_kind,
            "mag_coeffs": mag_coeffs,
            "eta_kind": eta_kind,
            "eta_coeffs": eta_coeffs,
            }


if __name__ == "__main__":
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    out_folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity"  # noqa
    print(f"{'Num. devices:':<20} {jax.device_count()}")
    print(f"{'Devices:':<20} {jax.devices()}")

    ###########################################################################
    #                        Fixed user parameters                            #
    ###########################################################################

    # `None` means default behaviour
    nsteps = 1500
    nburn = 1500
    zcmb_min = None
    zcmb_max = 0.05
    nchains_harmonic = 10
    num_epochs = 50
    inference_method = "mike"
    mag_selection = None
    sample_alpha = False if (ARGS.simname == "no_field" or "IndranilVoid" in ARGS.simname) else True  # noqa
    sample_beta = None
    sample_h_e_int = False
    no_Vext = None
    sample_Vmono = False
    sample_mag_dipole = False
    dust_model = None
    Rdust_fixed = None  # Default for W1 is 0.186 and for W2 = 0.123
    wo_num_dist_marginalisation = False
    absolute_calibration = None
    calculate_harmonic = (False if (inference_method == "bayes") else True) and (not wo_num_dist_marginalisation)  # noqa
    sample_h = True if absolute_calibration is not None else False
    which_void_size_run = "zoom"

    if ARGS.aux_name != "":
        if ARGS.aux_type == "int":
            ARGS.aux_arg = int(ARGS.aux_arg)
        elif ARGS.aux_type == "float":
            ARGS.aux_arg = float(ARGS.aux_arg)
        elif ARGS.aux_type != "str":
            raise ValueError(f"Unsupported auxiliary type: `{ARGS.aux_type}`.")

        fprint(f"setting {ARGS.aux_name} to {ARGS.aux_arg}")
        globals()[ARGS.aux_name] = ARGS.aux_arg

    # Overwrite if if not running a varying void size simulation.
    if "IndranilVoidSizeVar_" not in ARGS.simname:
        which_void_size_run = None

    if any("Pantheon+" in cat for cat in ARGS.catalogue):
        calculate_harmonic = False

    # These mocks are generated without a density field, so there is no
    # inhomogeneous Malmquist and we also do not need evidences.
    for catalogue in ARGS.catalogue:
        if "Carrick2MTFmock" in catalogue:
            sample_alpha = False
            calculate_harmonic = False

    fname_kwargs = {"inference_method": inference_method,
                    "smooth": ARGS.ksmooth,
                    "nsim": ARGS.ksim,
                    "zcmb_min": zcmb_min,
                    "zcmb_max": zcmb_max,
                    "mag_selection": mag_selection,
                    "sample_alpha": sample_alpha,
                    "sample_beta": sample_beta,
                    "no_Vext": no_Vext,
                    "sample_Vmono": sample_Vmono,
                    "sample_mag_dipole": sample_mag_dipole,
                    "absolute_calibration": absolute_calibration,
                    "sample_h_e_int": sample_h_e_int,
                    "which_void_size_run": which_void_size_run,
                    "dust_model": dust_model,
                    "Rdust_fixed": Rdust_fixed,
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
                   "dust_model": dust_model,
                   "Rdust_fixed": Rdust_fixed,
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

        # Check whether running at some fixed void size or varying void size
        if "SizeVar" not in ARGS.simname:
            size_indx = ''.join(
                char for char in ARGS.simname if char.isdigit()).zfill(3)

            # This indicates the fiducial run
            if size_indx == "000":
                size_indx = None
        else:
            size_indx = None

        profile = ARGS.simname.split("_")[-1]

        # This is the radial distance over which to intergrate along the LOS.
        # 165 Mpc / h should be sufficient
        rdist = np.arange(0, 165, 0.5)

        # Create the interpolator of void size to void Hubble parameter
        void_size, h_void = csiborgtools.flow.select_void_h(
            None, profile, return_all=True)

        # Check if the mapping is in H0 or h and sizes in percent
        void_size_to_h_void = Interpolator1D(
            void_size / 100, h_void, method="linear",
            extrap=(h_void[0], h_void[-1]))

        is_fiducial = "IndranilVoidSizeVar" not in ARGS.simname

        void_kwargs = {
            "profile": profile, "void_size_to_h_void": void_size_to_h_void,
            "which_void_size_run": which_void_size_run, "order": 1,
            "rdist": rdist, "is_fiducial": is_fiducial,
            "size_indx": size_indx}

        if which_void_size_run == "zoom":
            void_size_min, void_size_max = 0.01, 0.2
        elif which_void_size_run == "coarse":
            void_size_min, void_size_max = 0.1, 3.0
        elif is_fiducial:
            void_size_min, void_size_max = None, None
        else:
            raise ValueError(
                f"Unsupported void size run: `{which_void_size_run}`.")

    else:
        void_kwargs = None
        void_size_min, void_size_max = None, None

    if inference_method != "bayes":
        mag_selection = [None] * len(ARGS.catalogue)
    elif mag_selection is None or mag_selection:
        mag_selection = [get_toy_selection(cat) for cat in ARGS.catalogue]

    if nsteps % nchains_harmonic != 0:
        raise ValueError(
            "The number of steps must be divisible by the number of chains.")

    Vext_i_lim = 1000
    num_dust_maps = len(dust_model.split(",")) if dust_model is not None else 0
    sample_void_size = "IndranilVoidSizeVar" in ARGS.simname
    calibration_hyperparams = {"Vext_i_min": -Vext_i_lim,
                               "Vext_i_max": Vext_i_lim,
                               "Vmono_min": -1000, "Vmono_max": 1000,
                               "beta_min": -10.0, "beta_max": 10.0,
                               "sigma_v_min": 10., "sigma_v_max": 750.,
                               "h_min": 0.25, "h_max": 5.,
                               "no_Vext": no_Vext is not None,
                               "sample_Vmono": sample_Vmono,
                               "sample_beta": sample_beta,
                               "sample_h": sample_h,
                               "sample_h_e_int": sample_h_e_int,
                               "sample_rLG": "IndranilVoid" in ARGS.simname,
                               "sample_void_size": sample_void_size,
                               "void_size_min": void_size_min,
                               "void_size_max": void_size_max,
                               "rLG_min": -50, "rLG_max": 50,
                               "sample_dust": dust_model is not None,
                               "num_dust_maps": num_dust_maps,
                               }
    print_variables(
        calibration_hyperparams.keys(), calibration_hyperparams.values())

    distmod_hyperparams_per_catalogue = []
    for cat in ARGS.catalogue:
        x = get_distmod_hyperparams(
            cat, sample_alpha, sample_mag_dipole, dust_model, Rdust_fixed)
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
        "dust_model": dust_model,
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
