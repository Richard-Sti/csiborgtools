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
"""
MPI script to interpolate the density and velocity fields along the line of
sight.
"""
from os.path import join

import csiborgtools
import numpy as np
from astropy.coordinates import SkyCoord, angular_separation
from h5py import File
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator

from field_los import get_los


def interpolate_indranil_void(kind, RA, dec, rmax, dr, dump_folder, catalogue):
    fdir = "/mnt/extraspace/rstiskalek/catalogs"
    if kind == "exp":
        fname = join(fdir, "v_pec_EXP_IndranilVoid.dat")
    elif kind == "gauss":
        fname = join(fdir, "v_pec_GAUSS_IndranilVoid.dat")
    else:
        raise ValueError("Invalid void kind.")

    # These are only velocities.
    data = np.loadtxt(fname)
    fname_out = join(dump_folder, f"los_{catalogue}_IndranilVoid_{kind}.hdf5")

    r_grid = np.arange(0, 251)
    phi_grid = np.arange(0, 181)
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
    print(f"Writing to `{fname_out}`.")
    with File(fname_out, 'w') as f_out:
        f_out.create_dataset("rdist_0", data=r_eval * 0.674)
        f_out.create_dataset("density_0", data=density)
        f_out.create_dataset("velocity_0", data=result)


###############################################################################
#                           Command line interface                            #
###############################################################################


if __name__ == "__main__":
    kind = "exp"
    rmax = 165
    dr = 1

    comm = MPI.COMM_WORLD
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    out_folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/field_los"

    for catalogue in ["LOSS", "Foundation", "2MTF", "SFI_gals", "CF4_TFR", "CF4_GroupAll"]:  # noqa
        print(f"Running kind `{kind}` for catalogue `{catalogue}`.")

        RA, dec = get_los(catalogue, "", comm).T
        interpolate_indranil_void(
            kind, RA, dec, rmax, dr, out_folder, catalogue)
