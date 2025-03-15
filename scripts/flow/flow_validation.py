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
        "--aux_type", type=str, default="str",
        choices=["str", "int", "float", "bool"],
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
from numpyro.infer import MCMC, NUTS, init_to_median                            # noqa


def print_variables(names, variables):
    for name, variable in zip(names, variables):
        print(f"{name:<20} {variable}", flush=True)
    print(flush=True)


def get_models(ksim, get_model_kwargs, selection, void_kwargs,
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
            ki = cat.split("_")[-1]
            fpath = f"/mnt/extraspace/rstiskalek/csiborg_postprocessing/flow_mock/void_mock_seed{ki}.hdf5"  # noqa
        elif cat in ["SDSS-FP"]:
            fpath = join(folder, "PV/CF4/SDSS-FP-LOWZ.hdf5")
        else:
            raise ValueError(f"Unsupported catalogue: `{ARGS.catalogue}`.")

        loader = csiborgtools.flow.DataLoader(ARGS.simname, nsim_iterator,
                                              cat, fpath, paths,
                                              ksmooth=ARGS.ksmooth)
        models[i] = csiborgtools.flow.get_model(
            loader, selection=selection[i], void_kwargs=void_kwargs,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation,
            **get_model_kwargs)

    fprint(f"num. radial steps is {len(loader.rdist)} from {loader.rdist[0]} "
           f"to {loader.rdist[-1]} Mpc / h.")

    return models


def get_harmonic_evidence(samples, log_posterior, nchains_harmonic, epoch_num):
    """Compute evidence using the `harmonic` package."""
    data, names = csiborgtools.dict_samples_to_array(
        samples, exclude_deterministic=True)
    fprint(f"computing harmonic evidence from {len(names)} parameters: {names}")  # noqa
    data = data.reshape(nchains_harmonic, -1, len(names))
    log_posterior = log_posterior.reshape(nchains_harmonic, -1)

    return csiborgtools.harmonic_evidence(
        data, log_posterior, return_flow_samples=False, epochs_num=epoch_num)


def get_laplace_evidence(samples, log_posterior):
    data, names = csiborgtools.dict_samples_to_array(
        samples, exclude_deterministic=True)
    fprint(f"computing Laplace evidence from {len(names)} parameters: {names}")
    data = data.reshape(nchains_harmonic, -1, len(names))
    log_posterior = log_posterior.reshape(nchains_harmonic, -1)

    return csiborgtools.laplace_evidence(data, log_posterior)


def run_model(model, nsteps, nburn,  model_kwargs, out_folder,
              calculate_harmonic, calculate_laplace, nchains_harmonic,
              epoch_num, kwargs_print, fname_kwargs):
    """Run the NumPyro model and save output to a file."""
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    fname = paths.flow_validation(out_folder, ARGS.simname, ARGS.catalogue,
                                  **fname_kwargs)

    try:
        ndata = sum(model.ndata for model in model_kwargs["models"])
    except AttributeError as e:
        raise AttributeError("The models must have an attribute `ndata` "
                             "indicating the number of data points.") from e

    nuts_kernel = NUTS(model, init_strategy=init_to_median(num_samples=100))
    mcmc = MCMC(nuts_kernel, num_warmup=nburn, num_samples=nsteps)
    rng_key = jax.random.PRNGKey(42)

    mcmc.run(rng_key, extra_fields=("potential_energy",), **model_kwargs)
    samples = mcmc.get_samples()

    fprint("recomputing the log-density in the constrained space")
    log_posterior = csiborgtools.flow.PV_validation_model_log_density(
        samples, model, model_kwargs)

    BIC, AIC = csiborgtools.BIC_AIC(samples, log_posterior, ndata)
    fprint(f"{'BIC':<20} {BIC}")
    fprint(f"{'AIC':<20} {AIC}")
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

    if calculate_laplace:
        ln_evidence_laplace, ln_evidence_laplace_err = get_laplace_evidence(
            samples, log_posterior)
        print(f"{'-ln(Z_l)':<20} {-ln_evidence_laplace}")
        print(f"{'-ln(Z_l) error':<20} {ln_evidence_laplace_err}")
    else:
        ln_evidence_laplace = jax.numpy.nan
        ln_evidence_laplace_err = jax.numpy.nan

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
        grp.create_dataset("lnZ_laplace", data=ln_evidence_laplace)
        grp.create_dataset("lnZ_laplace_err", data=ln_evidence_laplace_err)

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

def get_distmod_hyperparams(catalogue, sample_alpha, Rdust_fixed,):
    alpha_mean = 1.0
    alpha_std = 1.0

    if catalogue in ["LOSS", "Foundation"]:
        return {"e_mu_min": 0.005, "e_mu_max": 1.0,
                "mag_cal_mean": -18.25, "mag_cal_std": 2.0,
                "alpha_cal_mean": 0.148, "alpha_cal_std": 1.0,
                "beta_cal_mean": 3.112, "beta_cal_std": 2.0,
                "alpha_mean": alpha_mean, "alpha_std": alpha_std,
                "sample_alpha": sample_alpha,
                }
    elif catalogue in ["Pantheon+", "Pantheon+_groups", "Pantheon+_zSN"]:
        return {"e_mu_min": 0.001, "e_mu_max": 1.0,
                "mag_cal_mean": -18.5, "mag_cal_std": 2.0,
                "alpha_mean": alpha_mean, "alpha_std": alpha_std,
                "sample_alpha": sample_alpha,
                }
    elif catalogue in ["SFI_gals", "2MTF"] or "CF4_TFR" in catalogue or "IndranilVoidTFRMock" in catalogue or "Carrick2MTFmock" in catalogue:  # noqa
        return {"e_mu_min": 0.005, "e_mu_max": 1.0,
                "a_mean": -22.0, "a_std": 5.0,
                "b_mean": -7.0, "b_std": 5.0,
                "c_mean": 10., "c_std": 20.0,
                "alpha_mean": alpha_mean, "alpha_std": alpha_std,
                "sample_alpha": sample_alpha,
                "sample_curvature": True if "2MTF" not in catalogue else False,
                "Rdust_min": 0,
                "Rdust_max": 1.0,
                "Rdust_fixed": Rdust_fixed,
                }
    elif catalogue in ["CF4_GroupAll"]:
        return {"e_mu_min": 0.005, "e_mu_max": 1.0,
                "dmu_min": -3.0, "dmu_max": 3.0,
                "alpha_mean": alpha_mean, "alpha_std": alpha_std,
                "sample_alpha": sample_alpha,
                }
    elif catalogue in ["SDSS-FP"]:
        return {"e_mu_min": 0.005, "e_mu_max": 10.0,
                "a_mean": 0.0, "a_std": 2.0,
                "b_mean": 0.0, "b_std": 2.0,
                "c_mean": 0.0, "c_std": 2.0,
                "alpha_mean": alpha_mean, "alpha_std": alpha_std,
                "sample_alpha": sample_alpha,
                }
    else:
        raise ValueError(f"Unsupported catalogue: `{ARGS.catalogue}`.")


def get_selection(catalogue, zcmb_max):
    """Toy magnitude selection coefficients."""
    if catalogue == "SFI_gals":
        mag_kind = "soft"
        # m1, m2, a
        mag_coeffs = [11.602, 12.948, -0.233]
        eta_coeffs = [None, None]
        eta_kind = None
    elif "CF4_TFR" in catalogue and "_i" in catalogue:
        mag_kind = "soft"
        mag_coeffs = [12.243, 13.898, -0.173]
        eta_coeffs = [-0.3, None]
        eta_kind = "lower_hard"
    elif "CF4_TFR" in catalogue and "w1" in catalogue:
        mag_kind = "soft"
        mag_coeffs = [11.206, 13.203, -0.152]
        eta_kind = "lower_hard"
        eta_coeffs = [-0.3, None]
    elif "CF4_TFR" in catalogue and "w2" in catalogue:
        mag_kind = "soft"
        mag_coeffs = [11.752, 13.772, -0.150]
        eta_kind = "lower_hard"
        eta_coeffs = [-0.3, None]
    elif catalogue == "2MTF":
        mag_kind = "hard"
        mag_coeffs = 11.25
        eta_kind = "hard"
        eta_coeffs = [-0.09859945625066757, 0.2007037103176117]
    elif "Carrick2MTFmock" in catalogue:
        mag_kind = "soft"
        # Make sure these match what was used to generate the mock.
        mag_coeffs = [11.8, 13.4, -0.19]
        eta_kind = None
        eta_coeffs = [None, None]
    else:
        print(f"found no selection coefficients for `{catalogue}`.")
        return None

    print(f"{catalogue}: mag_kind={mag_kind}, "
          f"mag_coeffs={mag_coeffs}, eta_kind={eta_kind}, "
          f"eta_coeffs={eta_coeffs}, "
          f"zcmb_max={zcmb_max}."
          )

    return {"mag_kind": mag_kind,
            "mag_coeffs": mag_coeffs,
            "eta_kind": eta_kind,
            "eta_coeffs": eta_coeffs,
            "zcmb_max": zcmb_max,
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
    nsteps = 2500
    nburn = 1000
    zcmb_min = None
    zcmb_max = 0.05

    nchains_harmonic = 10
    num_epochs = 50
    inference_method = "mike"
    sample_alpha = False if ("no_field" in ARGS.simname or "IndranilVoid" in ARGS.simname) else True  # noqa
    sample_beta = None
    sample_h_e_int = False
    no_Vext = None
    Vext_prior_kind = None
    sample_Vmono = False
    sample_mag_dipole = False
    mag_dipole_prior_kind = "fixed"  # Defaults to `None` if not sampled.
    dust_model = None
    Rdust_fixed = None  # Default for W1 is 0.186 and for W2 = 0.123
    wo_num_dist_marginalisation = False
    absolute_calibration = None
    which_void_size_run = "zoom"
    remove_CF4_outliers = False

    if ARGS.aux_name != "none":
        if ARGS.aux_type == "int":
            ARGS.aux_arg = int(ARGS.aux_arg)
        elif ARGS.aux_type == "float":
            ARGS.aux_arg = float(ARGS.aux_arg)
        elif ARGS.aux_type == "bool":
            ARGS.aux_arg = int(ARGS.aux_arg)
            if ARGS.aux_arg not in [0, 1]:
                raise ValueError(f"Unsupported boolean value: `{ARGS.aux_arg}`.")  # noqa
            ARGS.aux_arg = bool(ARGS.aux_arg)
        elif ARGS.aux_type != "str":
            raise ValueError(f"Unsupported auxiliary type: `{ARGS.aux_type}`.")

        fprint(f"setting `{ARGS.aux_name}` to `{ARGS.aux_arg}`.")
        globals()[ARGS.aux_name] = ARGS.aux_arg

    calculate_harmonic = (False if (inference_method == "bayes") else True) and (not wo_num_dist_marginalisation)  # noqa
    calculate_laplace = calculate_harmonic
    sample_h = True if absolute_calibration is not None else False

    if not sample_mag_dipole:
        mag_dipole_prior_kind = None

    if Vext_prior_kind not in [None, "fixed"]:
        raise ValueError(f"Unsupported Vext prior kind: `{Vext_prior_kind}`.")

    if mag_dipole_prior_kind not in [None, "fixed"]:
        raise ValueError(f"Unsupported mag dipole prior kind: `{mag_dipole_prior_kind}`.")  # noqa

    # Overwrite if if not running a varying void size simulation.
    if "IndranilVoidSizeVar_" not in ARGS.simname:
        which_void_size_run = None

    if "IndranilVoid" in ARGS.simname and no_Vext:
        raise ValueError("`Vext` must be sampled for the void, as it is "
                         "needed to define the void axis.")

    if any("Pantheon+" in cat for cat in ARGS.catalogue):
        calculate_harmonic = False
        calculate_laplace = False

    # These mocks are generated without a density field, so there is no
    # inhomogeneous Malmquist.
    for catalogue in ARGS.catalogue:
        if "Carrick2MTFmock" in catalogue:
            sample_alpha = False

    fname_kwargs = {"inference_method": inference_method,
                    "smooth": ARGS.ksmooth,
                    "nsim": ARGS.ksim,
                    "zcmb_min": zcmb_min,
                    "zcmb_max": zcmb_max,
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
                    "Vext_prior_kind": Vext_prior_kind,
                    "mag_dipole_prior_kind": mag_dipole_prior_kind,
                    "remove_CF4_outliers": remove_CF4_outliers,
                    }

    main_params = {"nsteps": nsteps, "nburn": nburn,
                   "zcmb_min": zcmb_min,
                   "zcmb_max": zcmb_max,
                   "calculate_harmonic": calculate_harmonic,
                   "calculate_laplace": calculate_laplace,
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
        # 250 Mpc / h should be sufficient (plus the grid only extends to
        # 400 Mpc)
        rdist = np.arange(0, 250, 0.5)

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

    print("Selection:")
    selection = [get_selection(cat, zcmb_max) for cat in ARGS.catalogue]
    print()

    if nsteps % nchains_harmonic != 0:
        raise ValueError(
            "The number of steps must be divisible by the number of chains.")

    num_dust_maps = len(dust_model.split(",")) if dust_model is not None else 0
    sample_void_size = "IndranilVoidSizeVar" in ARGS.simname
    calibration_hyperparams = {"Vext_mag_min": 0,
                               "Vext_mag_max": 2000,
                               "Vmono_min": -1000, "Vmono_max": 1000,
                               "e_mu_h_min": 0.001, "e_mu_h_max": 1.0,
                               "beta_min": -10.0, "beta_max": 10.0,
                               "sigma_v_min": 10., "sigma_v_max": 5000.,
                               "h_min": 0.25, "h_max": 5.,
                               "no_Vext": False if no_Vext is None else no_Vext,  # noqa
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
                               "Vext_prior_kind": Vext_prior_kind,
                               "mag_dipole_min": 0.0, "mag_dipole_max": 0.25,
                               "sample_mag_dipole": sample_mag_dipole,
                               "mag_dipole_prior_kind": mag_dipole_prior_kind,
                               }
    print_variables(
        calibration_hyperparams.keys(), calibration_hyperparams.values())

    distmod_hyperparams_per_catalogue = []
    for cat in ARGS.catalogue:
        x = get_distmod_hyperparams(cat, sample_alpha, Rdust_fixed,)
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
        "remove_CF4_outliers": remove_CF4_outliers,
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
        models = get_models(ksim, get_model_kwargs, selection, void_kwargs,
                            wo_num_dist_marginalisation)
        model_kwargs = {
            "models": models,
            "field_calibration_hyperparams": calibration_hyperparams,
            "distmod_hyperparams_per_model": distmod_hyperparams_per_catalogue,
            "inference_method": inference_method,
            }

        model = csiborgtools.flow.PV_validation_model

        run_model(model, nsteps, nburn, model_kwargs, out_folder,
                  calculate_harmonic, calculate_laplace, nchains_harmonic,
                  num_epochs, kwargs_print, fname_kwargs)
