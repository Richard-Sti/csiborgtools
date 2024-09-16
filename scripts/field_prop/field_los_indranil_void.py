# Copyright (C) 2024 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Script to interpolate the Indranil void profiles for lines of sight."""
from argparse import ArgumentParser
from os.path import join, exists
from os import remove

import csiborgtools
import numpy as np
from astropy.coordinates import SkyCoord, angular_separation
from field_los import get_los
from h5py import File
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator
from tqdm import trange


def interpolate_indranil_void(kind, nsims, RA, dec, rmax, dr, dump_folder,
                              catalogue):
    if kind not in ["exp", "gauss", "mb"]:
        raise ValueError(f"Unknown void kind: `{kind}`.")

    kind = kind.upper()
    fdir = join("/mnt/extraspace/rstiskalek/catalogs", "IndranilVoid",
                f"{kind}profile")

    fname_out = join(
        dump_folder, f"los_{catalogue}_IndranilVoid_{kind.lower()}.hdf5")
    if exists(fname_out):
        print(f"Fname `{fname_out}` already exists. Removing.")
        remove(fname_out)

    print(f"Writing to `{fname_out}`.")
    for k in trange(len(nsims), desc="LG observers"):
        nsim = nsims[k]
        # These are only velocities.
        fname = join(fdir, f"v_pec_{kind}profile_rLG_{nsim}.dat")
        data = np.loadtxt(fname)

        # The grid is in Mpc
        r_grid = np.arange(0, 251)
        phi_grid = np.arange(0, 181)
        # The input is in Mpc/h, so we need to convert to Mpc
        r_eval = np.arange(0, rmax, dr).astype(float) / 0.674

        model_axis = SkyCoord(l=117, b=4, frame='galactic', unit='deg').icrs
        coords = SkyCoord(ra=RA, dec=dec, unit='deg').icrs

        # Get angular separation in degrees
        phi = angular_separation(coords.ra.rad, coords.dec.rad,
                                 model_axis.ra.rad, model_axis.dec.rad)
        phi *= 180 / np.pi

        # Get the interpolator
        f = RegularGridInterpolator((r_grid, phi_grid), data.T)
        # Get the dummy x-values to evaluate for each LOS
        x_dummy = np.ones((len(r_eval), 2))
        x_dummy[:, 0] = r_eval

        result = np.full((len(RA), len(r_eval)), np.nan)
        for i in range(len(RA)):
            x_dummy[:, 1] = phi[i]
            result[i] = f(x_dummy)

        # Write the output, homogenous density.
        density = np.ones_like(result)
        with File(fname_out, 'a') as f_out:
            f_out.create_dataset(f"rdist_{k}", data=r_eval * 0.674)
            f_out.create_dataset(f"density_{k}", data=density)
            f_out.create_dataset(f"velocity_{k}", data=result)


###############################################################################
#                           Command line interface                            #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--kind", type=str, choices=["exp", "gauss", "mb"],
                        help="Kind of void profile.")
    parser.add_argument("--catalogue", type=str, help="Catalogue name.")
    args = parser.parse_args()

    rmax = 165
    dr = 0.75

    comm = MPI.COMM_WORLD
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(f"IndranilVoid_{args.kind}")
    out_folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/field_los"

    print(f"Running kind `{args.kind}` for catalogue `{args.catalogue}`.")
    RA, dec = get_los(args.catalogue, "", comm).T
    interpolate_indranil_void(
        args.kind, nsims, RA, dec, rmax, dr, out_folder, args.catalogue)
