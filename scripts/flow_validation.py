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
from numpyro.infer import MCMC, NUTS
from taskmaster import work_delegation  # noqa


def get_model(args, nsim):
    """
    Load the data and create the NumPyro model.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    nsim : int
        Simulation index.

    Returns
    -------
    numpyro.Primitive
    """
    folder = "/mnt/extraspace/rstiskalek/catalogs/"
    if args.catalogue == "A2":
        fpath = join(folder, "A2.h5")
    elif args.catalogue == "LOSS" or args.catalogue == "Foundation":
        raise NotImplementedError("To be implemented..")
    else:
        raise ValueError(f"Unknown catalogue: `{args.catalogue}`.")

    loader = csiborgtools.flow.DataLoader(args.simname, args.catalogue, fpath,
                                          paths, ksmooth=args.ksmooth)
    Omega_m = csiborgtools.simname2Omega_m(args.simname)

    # Read in the data from the loader.
    los_overdensity = loader.los_density[:, nsim, :]
    los_velocity = loader.los_radial_velocity[:, nsim, :]

    RA = loader.cat["RA"]
    dec = loader.cat["DEC"]
    z_obs = loader.cat["z_obs"]

    r_hMpc = loader.cat["r_hMpc"]
    e_r_hMpc = loader.cat["e_rhMpc"]

    return csiborgtools.flow.SD_PV_validation_model(
        los_overdensity, los_velocity, RA, dec, z_obs, r_hMpc, e_r_hMpc,
        loader.rdist, Omega_m)


def run_model(model, nsteps, nchains, nsim, dump_folder, show_progress=True):
    """
    Run the NumPyro model and save the thinned samples to a temporary file.

    Parameters
    ----------
    model : jax.numpyro.Primitive
        Model to be run.
    nsteps : int
        Number of steps.
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
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=nsteps // 2, num_samples=nsteps // 2,
                chain_method="sequential", num_chains=nchains,
                progress_bar=show_progress)
    rng_key = jax.random.PRNGKey(42)
    mcmc.run(rng_key)

    if show_progress:
        print(f"Summary of the MCMC run of simulation indexed {nsim}:")
        mcmc.print_summary()

    samples = mcmc.get_samples()
    thinned_samples = csiborgtools.thin_samples_by_acl(samples)

    # Save the samples to the temporary folder.
    fname = join(dump_folder, f"samples_{nsim}.npz")
    np.savez(fname, **thinned_samples)


def combine_from_simulations(catalogue_name, simname, nsims, outfolder,
                             dumpfolder):
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

    Returns
    -------
    None
    """
    fname_out = join(outfolder,
                     f"flow_samples_{catalogue_name}_{simname}.hdf5")
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
    parser.add_argument("--ksmooth", type=int, default=None,
                        help="Smoothing index.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    out_folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity"  # noqa

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(args.simname)

    nsteps = 5000
    nchains = 1

    # Create the dumping folder.
    if comm.Get_rank() == 0:
        dump_folder = join(out_folder,
                           f"temp_{str(datetime.now())}".replace(" ", "_"))
        print(f"Creating folder `{dump_folder}`.")
        makedirs(dump_folder)
    else:
        dump_folder = None
    dump_folder = comm.bcast(dump_folder, root=0)

    def main(nsim):
        model = get_model(args, nsim)
        run_model(model, nsteps, nchains, nsim, dump_folder,
                  show_progress=size == 1)

    work_delegation(main, nsims, comm, master_verbose=True)
    comm.Barrier()

    if rank == 0:
        combine_from_simulations(args.catalogue, args.simname, nsims,
                                 out_folder, dump_folder)
