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
The script is MPI parallelized over the IC realizations.
"""
from argparse import ArgumentParser
from datetime import datetime
from os import makedirs, remove, rmdir
from os.path import exists, join

import csiborgtools
import jax
import numpy as np
from h5py import File
from mpi4py import MPI
from numpyro.infer import MCMC, NUTS, init_to_sample
from taskmaster import work_delegation  # noqa


def get_model(args, nsim_iterator, get_model_kwargs):
    """
    Load the data and create the NumPyro model.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    nsim_iterator : int
        Simulation index, not the IC index. Ranges from 0, ... .
    get_model_kwargs : dict
        Keyword arguments for reading in the data for the model
        (`csiboorgtools.flow.get_model`).

    Returns
    -------
    numpyro model
    """
    folder = "/mnt/extraspace/rstiskalek/catalogs/"
    if args.catalogue == "A2":
        fpath = join(folder, "A2.h5")
    elif args.catalogue in ["LOSS", "Foundation", "Pantheon+", "SFI_gals",
                            "2MTF", "SFI_groups", "SFI_gals_masked"]:
        fpath = join(folder, "PV_compilation_Supranta2019.hdf5")
    else:
        raise ValueError(f"Unknown catalogue: `{args.catalogue}`.")

    loader = csiborgtools.flow.DataLoader(args.simname, nsim_iterator,
                                          args.catalogue, fpath, paths,
                                          ksmooth=args.ksmooth)

    return csiborgtools.flow.get_model(loader, **get_model_kwargs)


def run_model(model, nsteps, nburn, nchains, nsim, dump_folder,
              model_kwargs, show_progress=True):
    """
    Run the NumPyro model and save the thinned samples to a temporary file.

    Parameters
    ----------
    model : jax.numpyro.Primitive
        Model to be run.
    nsteps : int
        Number of steps.
    nburn : int
        Number of burn-in steps.
    nchains : int
        Number of chains.
    nsim : int
        Simulation index.
    dump_folder : str
        Folder where the temporary files are stored.
    show_progress : bool
        Whether to show the progress bar.

    Returns
    -------
    None
    """
    nuts_kernel = NUTS(model, init_strategy=init_to_sample)
    mcmc = MCMC(nuts_kernel, num_warmup=nburn, num_samples=nsteps,
                chain_method="sequential", num_chains=nchains,
                progress_bar=show_progress)
    rng_key = jax.random.PRNGKey(42)
    mcmc.run(rng_key, **model_kwargs)

    if show_progress:
        print(f"Summary of the MCMC run of simulation indexed {nsim}:")
        mcmc.print_summary()

    samples = mcmc.get_samples()
    thinned_samples = csiborgtools.thin_samples_by_acl(samples)

    # Calculate the chi2
    keys = list(thinned_samples.keys())
    nsamples = len(thinned_samples[keys[0]])
    try:
        zobs_mean, zobs_std = model.predict_zobs(thinned_samples)
        nu = model.ndata - len(keys)
        chi2 = [np.sum((zobs_mean[:, i] - model._z_obs)**2 / zobs_std[:, i]**2) / nu  # noqa
                for i in range(nsamples)]
    except NotImplementedError:
        chi2 = [0. for _ in range(nsamples)]

    gof = csiborgtools.numpyro_gof(model, mcmc, model_kwargs)

    # Save the samples to the temporary folder.
    fname = join(dump_folder, f"samples_{nsim}.npz")
    np.savez(fname, **thinned_samples, **gof, chi2=chi2)


def combine_from_simulations(catalogue_name, simname, nsims, outfolder,
                             dumpfolder, ksmooth):
    """
    Combine the results from individual simulations into a single file.

    Parameters
    ----------
    catalogue_name : str
        Catalogue name.
    simname : str
        Simulation name.
    nsims : list
        List of IC realisations.
    outfolder : str
        Output folder.
    dumpfolder : str
        Dumping folder where the temporary files are stored.
    ksmooth : int
        Smoothing index.

    Returns
    -------
    None
    """
    fname_out = join(
        outfolder,
        f"flow_samples_{catalogue_name}_{simname}_smooth_{ksmooth}.hdf5")
    print(f"Combining results from invidivual simulations to `{fname_out}`.")

    if exists(fname_out):
        remove(fname_out)

    for nsim in nsims:
        fname = join(dumpfolder, f"samples_{nsim}.npz")
        data = np.load(fname)

        with File(fname_out, 'a') as f:
            grp = f.create_group(f"sim_{nsim}")
            for key in data.files:
                grp.create_dataset(key, data=data[key])

        # Remove the temporary file.
        remove(fname)

    # Remove the dumping folder.
    rmdir(dumpfolder)
    print("Finished combining results.")

###############################################################################
#                        Command line interface                               #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, required=True,
                        help="Simulation name.")
    parser.add_argument("--catalogue", type=str, required=True,
                        help="PV catalogue.")
    parser.add_argument("--ksmooth", type=int, required=True,
                        help="Smoothing index.")
    parser.add_argument("--nchains", type=int, default=4,
                        help="Number of chains.")
    parser.add_argument("--nsteps", type=int, default=2500,
                        help="Number of post burn-n steps.")
    parser.add_argument("--nburn", type=int, default=500,
                        help="Number of burn-in steps.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    out_folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity"  # noqa

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(args.simname)

    get_model_kwargs = {"zcmb_max": 0.06}
    model_kwargs = {"sample_alpha": True}

    # Create the dumping folder.
    if comm.Get_rank() == 0:
        dump_folder = join(out_folder,
                           f"temp_{str(datetime.now())}".replace(" ", "_"))
        print(f"Creating folder `{dump_folder}`.")
        makedirs(dump_folder)
    else:
        dump_folder = None
    dump_folder = comm.bcast(dump_folder, root=0)

    def main(i):
        model = get_model(args, i, get_model_kwargs)
        run_model(model, args.nsteps, args.nburn, args.nchains, nsims[i],
                  dump_folder, model_kwargs, show_progress=size == 1)

    work_delegation(main, [i for i in range(len(nsims))], comm,
                    master_verbose=True)
    comm.Barrier()

    if rank == 0:
        combine_from_simulations(args.catalogue, args.simname, nsims,
                                 out_folder, dump_folder, args.ksmooth)
