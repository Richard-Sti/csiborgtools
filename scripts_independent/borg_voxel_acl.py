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
Script to calculate the ACL of BORG voxels.
"""
from argparse import ArgumentParser
from glob import glob
from os.path import join
from re import search

import numpy as np
from h5py import File
from numba import jit
from tqdm import tqdm, trange

###############################################################################
#                             BORG voxels I/O                                 #
###############################################################################


def find_mcmc_files(basedir):
    """
    Find the MCMC files in the BORG run directory. Checks that the samples
    are consecutive.

    Parameters
    ----------
    basedir : str
        The base directory of the BORG run.

    Returns
    -------
    files : list of str
    """
    files = glob(join(basedir, "mcmc_*"))
    print(f"Found {len(files)} BORG samples.")

    # Sort the files by the MCMC iteration number.
    indxs = [int(search(r"mcmc_(\d+)", f).group(1)) for f in files]
    argsort_indxs = np.argsort(indxs)
    indxs = [indxs[i] for i in argsort_indxs]
    files = [files[i] for i in argsort_indxs]

    if not all((indxs[i] - indxs[i - 1]) == 1 for i in range(1, len(indxs))):
        raise ValueError("MCMC iteration numbers are not consecutive.")

    return files


def load_borg_voxels(basedir, frac=0.25):
    """
    Load the BORG density field samples of the central `frac` of the box.

    Parameters
    ----------
    basedir : str
        The base directory of the BORG run.
    frac : float
        The fraction of the box to load. Must be <= 1.0.

    Returns
    -------
    4-dimensional array of shape (n_samples, n_voxels, n_voxels, n_voxels)
    """
    if frac > 1.0:
        raise ValueError("`frac` must be <= 1.0")

    files = find_mcmc_files(basedir)

    start, end, x = None, None, None
    for n, fpath in enumerate(tqdm(files, desc="Loading BORG samples")):
        with File(fpath, 'r') as f:
            if n == 0:
                grid = f["scalars/BORG_final_density"].shape[0]
                ncentral = int(grid * frac)

                start = (grid - ncentral) // 2
                end = (grid + ncentral) // 2
                nvoxel = end - start
                shape = (len(files), nvoxel, nvoxel, nvoxel)
                x = np.full(shape, np.nan, dtype=np.float32)

            x[n] = f["scalars/BORG_final_density"][start:end, start:end, start:end]  # noqa

    return x


def load_borg_galaxy_nmean(basedir):
    """
    Load the BORG `galaxy_nmean` samples.

    Parameters
    ----------
    basedir : str
        The base directory of the BORG run.

    Returns
    -------
    samples : 2-dimensional array of shape (n_samples, 10)
    """
    files = find_mcmc_files(basedir)

    x = np.full((len(files), 10), np.nan, dtype=np.float32)
    for n, fpath in enumerate(tqdm(files, desc="Loading BORG samples")):
        with File(fpath, 'r') as f:
            for i in range(10):
                nmean = f[f"scalars/galaxy_nmean_{i}"][...]
                if nmean.size > 1:
                    raise ValueError(f"Unknown shape of `galaxy_nmean_{i}`, {nmean.shape}.")  # noqa

                x[n, i] = nmean[0]

    return x


###############################################################################
#                           ACL & ACF calculation                             #
###############################################################################


def calculate_acf(data):
    """
    Calculates the autocorrelation of some data. Taken from `epsie` package
    written by Collin Capano.

    Parameters
    ----------
    data : 1-dimensional array
        The data to calculate the autocorrelation of.

    Returns
    -------
    acf : 1-dimensional array
    """
    # zero the mean
    data = data - data.mean()
    # zero-pad to 2 * nearest power of 2
    newlen = int(2**(1 + np.ceil(np.log2(len(data)))))
    x = np.zeros(newlen)
    x[:len(data)] = data[:]
    # correlate
    acf = np.correlate(x, x, mode='full')
    # drop corrupted region
    acf = acf[len(acf)//2:]
    # normalize
    acf /= acf[0]
    return acf


def calculate_acl(data):
    """
    Calculate the autocorrelation length of some data. Taken from `epsie`
    package written by Collin Capano. Algorithm used is from:
        N. Madras and A.D. Sokal, J. Stat. Phys. 50, 109 (1988).

    Parameters
    ----------
    data : 1-dimensional array
        The data to calculate the autocorrelation length of.

    Returns
    -------
    acl : int
    """
    # calculate the acf
    acf = calculate_acf(data)
    # now the ACL: Following from Sokal, this is estimated
    # as the first point where M*tau[k] <= k, where
    # tau = 2*cumsum(acf) - 1, and M is a tuneable parameter,
    # generally chosen to be = 5 (which we use here)
    m = 5
    cacf = 2. * np.cumsum(acf) - 1.
    win = m * cacf <= np.arange(len(cacf))
    if win.any():
        acl = int(np.ceil(cacf[np.where(win)[0][0]]))
    else:
        # data is too short to estimate the ACL, just choose
        # the length of the data
        acl = len(data)
    return acl


def voxel_acl(borg_voxels):
    """
    Calculate the ACL of each voxel in the BORG samples.

    Parameters
    ----------
    borg_voxels : 4-dimensional array of shape (n_samples, nvox, nvox, nvox)
        The BORG density field samples.

    Returns
    -------
    voxel_acl : 3-dimensional array of shape (nvox, nvox, nvox)
        The ACL of each voxel.
    """
    ngrid = borg_voxels.shape[1]
    voxel_acl = np.zeros((ngrid, ngrid, ngrid), dtype=np.float32)
    for i in trange(ngrid):
        for j in range(ngrid):
            for k in range(ngrid):
                voxel_acl[i, j, k] = calculate_acl(borg_voxels[:, i, j, k])

    return voxel_acl


def galaxy_nmean_acl(galaxy_nmean):
    nbias = galaxy_nmean.shape[1]

    for i in range(nbias):
        x = galaxy_nmean[:, i]
        mu = np.mean(x)
        sigma = np.std(x)
        acl = calculate_acl(x)

        print(f"<galaxy_nmean_{i}> = {mu} +- {sigma}")
        print(f"ACL                = {acl}")

    return


def enclosed_density_acl(borg_voxels):
    """
    Calculate the ACL of the enclosed overdensity of the BORG samples.

    Parameters
    ----------
    borg_voxels : 4-dimensional array of shape (n_samples, nvox, nvox, nvox)
        The BORG density field samples.

    Returns
    -------
    acl : int
    """
    # Calculate the mean overdensity of the voxels.
    x = np.asanyarray([np.mean(borg_voxels[i] + 1) - 1
                       for i in range(len(borg_voxels))])

    mu = np.mean(x)
    sigma = np.std(x)
    acl = calculate_acl(x)

    print("Calculating the boxed overdensity ACL.")
    print(f"<delta_box> = {mu} +- {sigma}")
    print(f"ACL         = {acl}")

    return acl


###############################################################################
#                       Voxel distance from the centre                        #
###############################################################################


@jit(nopython=True, boundscheck=False, fastmath=True)
def calculate_voxel_distance_from_center(grid, voxel_size):
    """
    Calculate the distance in `Mpc / h` of each voxel from the centre of the
    box.

    Parameters
    ----------
    grid : int
        The number of voxels in each dimension. Assumed to be centered on the
        box centre.
    voxel_size : float
        The size of each voxel in `Mpc / h`.

    Returns
    -------
    voxel_dist : 3-dimensional array of shape (grid, grid, grid)
    """
    x0 = grid // 2
    dist = np.zeros((grid, grid, grid), dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            for k in range(grid):
                dist[i, j, k] = ((i - x0)**2 + (j - x0)**2 + (k - x0)**2)**0.5

    return dist * voxel_size


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("kind", choices=["BORG1", "BORG2"],
                        help="The BORG run.", type=str)
    parser.add_argument("--frac", help="The fraction of the box to load.",
                        default=0.25, type=float)
    args = parser.parse_args()

    dumpdir = "/mnt/extraspace/rstiskalek/dump"
    outdir = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/ACL"
    if args.kind == "BORG1":
        basedir = "/mnt/users/hdesmond/BORG_final"
        grid = 256
        boxsize = 677.6
    elif args.kind == "BORG2":
        basedir = "/mnt/extraspace/rstiskalek/BORG_STOPYRA_2023"
        grid = 256
        boxsize = 676.6
    else:
        raise ValueError(f"Unknown BORG run: `{args.kind}`.")

    # First try to load the BORG samples from a dump file. If that fails, load
    # them directly from the BORG samples.
    fname = join(dumpdir, f"{args.kind}_{args.frac}.hdf5")
    try:
        with File(fname, 'r') as f:
            print(f"Loading BORG samples from `{fname}`.")
            borg_voxels = f["borg_voxels"][...]
    except FileNotFoundError:
        print("Loading directly from BORG samples.")
        borg_voxels = load_borg_voxels(basedir, frac=args.frac)

        with File(fname, 'w') as f:
            print(f"Saving BORG samples to to `{fname}`.")
            f.create_dataset("borg_voxels", data=borg_voxels)

    enclosed_density_acl(borg_voxels)

    # Calculate the voxel distance from the centre and their ACLs.
    voxel_size = boxsize / grid
    voxel_dist = calculate_voxel_distance_from_center(
        borg_voxels.shape[1], voxel_size)
    voxel_acl = voxel_acl(borg_voxels)

    # Save the voxel distance and ACLs to a file.
    fout = join(outdir, f"{args.kind}_{args.frac}.hdf5")
    print(f"Writting voxel distance and ACLs to `{fout}`.")
    with File(fout, 'w') as f:
        f.create_dataset("voxel_dist", data=voxel_dist)
        f.create_dataset("voxel_acl", data=voxel_acl)

    # Now load the galaxy_nmean samples.
    fname = join(dumpdir, f"{args.kind}_galaxy_nmean_{args.frac}.hdf5")
    try:
        with File(fname, 'r') as f:
            print(f"Loading BORG `galaxy_nmean` samples from `{fname}`.")
            galaxy_nmean = f["borg_voxels"][...]
    except FileNotFoundError:
        print("Loading `galaxy_nmean` directly from BORG samples.")
        galaxy_nmean = load_borg_galaxy_nmean(basedir)

        with File(fname, 'w') as f:
            print(f"Saving `galaxy_nmean` BORG samples to to `{fname}`.")
            f.create_dataset("galaxy_nmean", data=galaxy_nmean)

    galaxy_nmean_acl(galaxy_nmean)
