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
Script to calculate cosmological redshifts from observed redshifts assuming
the Carrick+2015 peculiar velocity model. In the future this may be extended
to include other peculiar velocity models.

TODO: Switch to the zSN calibration.
"""
from datetime import datetime
from os import remove
from os.path import join

import csiborgtools
import numpy as np
from h5py import File
from mpi4py import MPI
from taskmaster import work_delegation  # noqa
from tqdm import tqdm

SPEED_OF_LIGHT = 299792.458  # km / s


def t():
    return datetime.now().strftime("%H:%M:%S")


def load_calibration(catalogue, simname, nsim, ksmooth, verbose=False):
    """Load the pre-computed calibration samples."""
    fname = f"/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity/flow_samples_{catalogue}_{simname}_smooth_{ksmooth}.hdf5"  # noqa
    keys = ["Vext_x", "Vext_y", "Vext_z", "alpha", "beta", "sigma_v"]

    calibration_samples = {}
    with File(fname, 'r') as f:
        for key in keys:
            # NOTE: here the posterior samples are down-sampled
            calibration_samples[key] = f[f"sim_{nsim}/{key}"][:][::10]

    if verbose:
        k = list(calibration_samples.keys())[0]
        nsamples = len(calibration_samples[k])
        print(f"{t()}: found {nsamples} calibration posterior samples.",
              flush=True)

    return calibration_samples


def main(loader, model, indxs, fdir, fname, num_split, verbose):
    out = np.full(
        len(indxs), np.nan,
        dtype=[("mean_zcosmo", float), ("std_zcosmo", float)])

    # Process each galaxy in this split
    for i, n in enumerate(tqdm(indxs, desc=f"Split {num_split}",
                               disable=not verbose)):
        x, y = model.posterior_zcosmo(
            loader.cat["zcmb"][n], loader.cat["RA"][n], loader.cat["DEC"][n],
            loader.los_density[n], loader.los_radial_velocity[n],
            extra_sigma_v=loader.cat["e_zcmb"][n] * SPEED_OF_LIGHT,
            verbose=False)

        mu, std = model.posterior_mean_std(x, y)
        out["mean_zcosmo"][i], out["std_zcosmo"][i] = mu, std

    # Save the results of this rank
    fname = join(fdir, f"{fname}_{num_split}.hdf5")
    with File(fname, 'w') as f:
        f.create_dataset("mean_zcosmo", data=out["mean_zcosmo"])
        f.create_dataset("std_zcosmo", data=out["std_zcosmo"])
        f.create_dataset("indxs", data=indxs)


###############################################################################
#                           Command line interface                            #
###############################################################################


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    # Calibration parameters
    simname = "Carrick2015"
    ksmooth = 0
    nsim = 0
    catalogue_calibration = "Pantheon+"

    # Galaxy sample parameters
    catalogue = "UPGLADE"
    fpath_data = "/mnt/users/rstiskalek/csiborgtools/data/upglade_z_0p05_all_PROCESSED.h5"  # noqa

    # Number of splits for MPI
    nsplits = 10

    # Folder to save the results
    fdir = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity/UPGLADE"  # noqa
    fname = f"zcosmo_{catalogue}"

    # Load in the data, calibration samples and the model
    loader = csiborgtools.flow.DataLoader(
        simname, nsim, catalogue, fpath_data, paths, ksmooth=ksmooth,
        verbose=rank == 0)
    calibration_samples = load_calibration(
        catalogue_calibration, simname, nsim, ksmooth, verbose=rank == 0)
    model = csiborgtools.flow.Observed2CosmologicalRedshift(
        calibration_samples, loader.rdist, loader._Omega_m)
    if rank == 0:
        print(f"{t()}: the catalogue size is {loader.cat['zcmb'].size}.")
        print(f"{t()}: loaded calibration samples and model.", flush=True)

    # Decide how to split up the job
    if rank == 0:
        indxs = np.arange(loader.cat["zcmb"].size)
        split_indxs = np.array_split(indxs, nsplits)
    else:
        indxs = None
        split_indxs = None
    indxs = comm.bcast(indxs, root=0)
    split_indxs = comm.bcast(split_indxs, root=0)

    # Process all splits with MPI, the rank 0 delegates the jobs.
    def main_wrapper(n):
        main(loader, model, split_indxs[n], fdir, fname, n, verbose=size == 1)

    comm.Barrier()
    work_delegation(
        main_wrapper, list(range(nsplits)), comm, master_verbose=True)
    comm.Barrier()

    # Combine the results to a single file
    if rank == 0:
        print("Combining results from all ranks.", flush=True)
        mean_zcosmo = np.full(loader.cat["zcmb"].size, np.nan)
        std_zcosmo = np.full_like(mean_zcosmo, np.nan)

        for n in range(nsplits):
            fname_current = join(fdir, f"{fname}_{n}.hdf5")
            with File(fname_current, 'r') as f:
                mask = f["indxs"][:]
                mean_zcosmo[mask] = f["mean_zcosmo"][:]
                std_zcosmo[mask] = f["std_zcosmo"][:]

            remove(fname_current)

        # Save the results
        fname = join(fdir, f"{fname}.hdf5")
        print(f"Saving results to `{fname}`.")
        with File(fname, 'w') as f:
            f.create_dataset("mean_zcosmo", data=mean_zcosmo)
            f.create_dataset("std_zcosmo", data=std_zcosmo)
            f.create_dataset("indxs", data=indxs)
