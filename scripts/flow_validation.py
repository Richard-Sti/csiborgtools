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
                        help="PV catalogue.")
    parser.add_argument("--ksmooth", type=int, default=1,
                        help="Smoothing index.")
    parser.add_argument("--ksim", type=none_or_int, default=None,
                        help="IC iteration number. If 'None', all IC realizations are used.")  # noqa
    parser.add_argument("--ndevice", type=int, default=1,
                        help="Number of devices to request.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use.")
    return parser.parse_args()


ARGS = parse_args()
# This must be done before we import JAX etc.
from numpyro import set_host_device_count, set_platform                         # noqa
set_platform(ARGS.device)                                                       # noqa
set_host_device_count(ARGS.ndevice)                                             # noqa

import sys                                                                      # noqa
from os.path import join                                                        # noqa

import jax                                                                      # noqa
from h5py import File                                                           # noqa
from mpi4py import MPI                                                          # noqa
from numpyro.infer import MCMC, NUTS, init_to_median                            # noqa

import csiborgtools                                                             # noqa


def print_variables(names, variables):
    for name, variable in zip(names, variables):
        print(f"{name:<20} {variable}", flush=True)
    print(flush=True)


def get_model(paths, get_model_kwargs, verbose=True):
    """Load the data and create the NumPyro model."""
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

    if ARGS.catalogue == "A2":
        fpath = join(folder, "A2.h5")
    elif ARGS.catalogue in ["LOSS", "Foundation", "Pantheon+", "SFI_gals",
                            "2MTF", "SFI_groups", "SFI_gals_masked",
                            "Pantheon+_groups", "Pantheon+_groups_zSN",
                            "Pantheon+_zSN"]:
        fpath = join(folder, "PV_compilation.hdf5")
    else:
        raise ValueError(f"Unsupported catalogue: `{ARGS.catalogue}`.")

    loader = csiborgtools.flow.DataLoader(ARGS.simname, nsim_iterator,
                                          ARGS.catalogue, fpath, paths,
                                          ksmooth=ARGS.ksmooth)

    return csiborgtools.flow.get_model(loader, **get_model_kwargs)


def run_model(model, nsteps, nburn,  model_kwargs, out_folder, kwargs_print):
    """Run the NumPyro model and save output to a file."""
    nuts_kernel = NUTS(model, init_strategy=init_to_median(num_samples=1000))
    mcmc = MCMC(nuts_kernel, num_warmup=nburn, num_samples=nsteps)
    rng_key = jax.random.PRNGKey(42)

    mcmc.run(rng_key, **model_kwargs)
    BIC, AIC = csiborgtools.numpyro_gof(model, mcmc)
    print(f"{'BIC':<20} {BIC}")
    print(f"{'AIC':<20} {AIC}")

    mcmc.print_summary()
    samples = mcmc.get_samples()

    fname = f"samples_{ARGS.simname}_{ARGS.catalogue}_ksmooth{ARGS.ksmooth}.hdf5"  # noqa
    if ARGS.ksim is not None:
        fname = fname.replace(".hdf5", f"_nsim{ARGS.ksim}.hdf5")

    fname = join(out_folder, fname)
    print(f"Saving results to `{fname}`.")
    with File(fname, "w") as f:
        # Write samples
        grp = f.create_group("samples")
        for key, value in samples.items():
            grp.create_dataset(key, data=value)

        # Write goodness of fit
        grp = f.create_group("gof")
        grp.create_dataset("BIC", data=BIC)
        grp.create_dataset("AIC", data=AIC)

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
        mcmc.print_summary(exclude_deterministic=False)
        sys.stdout = original_stdout


###############################################################################
#                        Command line interface                               #
###############################################################################


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
    zcmb_max = 0.06
    sample_alpha = True
    sample_beta = True

    main_params = {"nsteps": nsteps, "nburn": nburn, "zcmb_max": zcmb_max,
                   "sample_alpha": sample_alpha, "sample_beta": sample_beta}

    print_variables(main_params.keys(), main_params.values())

    calibration_hyperparams = {"Vext_std": 250,
                               "alpha_mean": 1.0, "alpha_std": 0.5,
                               "beta_mean": 1.0, "beta_std": 0.5,
                               "sigma_v_mean": 200., "sigma_v_std": 100.,
                               "sample_alpha": True, "sample_beta": True,
                               }
    print_variables(
        calibration_hyperparams.keys(), calibration_hyperparams.values())

    if ARGS.catalogue in ["LOSS", "Foundation", "Pantheon+", "Pantheon+_groups"]:  # noqa
        distmod_hyperparams = {"e_mu_mean": 0.1, "e_mu_std": 0.05,
                               "mag_cal_mean": -18.25, "mag_cal_std": 0.5,
                               "alpha_cal_mean": 0.148, "alpha_cal_std": 0.05,
                               "beta_cal_mean": 3.112, "beta_cal_std": 1.0,
                               }
    elif ARGS.catalogue in ["SFI_gals", "2MTF"]:
        distmod_hyperparams = {"e_mu_mean": 0.3, "e_mu_std": 0.15,
                               "a_mean": -21., "a_std": 0.5,
                               "b_mean": -5.95, "b_std": 0.25,
                               }
    else:
        raise ValueError(f"Unsupported catalogue: `{ARGS.catalogue}`.")

    print_variables(
        distmod_hyperparams.keys(), distmod_hyperparams.values())

    kwargs_print = (main_params, calibration_hyperparams, distmod_hyperparams)
    ###########################################################################

    model_kwargs = {"calibration_hyperparams": calibration_hyperparams,
                    "distmod_hyperparams": distmod_hyperparams}
    get_model_kwargs = {"zcmb_max": zcmb_max}

    model = get_model(paths, get_model_kwargs, )
    run_model(model, nsteps, nburn, model_kwargs, out_folder, kwargs_print)
