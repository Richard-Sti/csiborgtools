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



A script to calculate the bulk flow in Quijote simulations from either
particles or FoF haloes and to also save the resulting smaller halo catalogues.



"""
import csiborgtools
import healpy as hp
import numpy as np
from h5py import File
from tqdm import tqdm


def load_field(nsim, MAS, grid, paths):
    """
    Load the precomputed radial velocity field from the Quijote simulations.
    """
    reader = csiborgtools.read.QuijoteField(nsim, paths)
    return reader.radial_velocity_field(MAS, grid)


def skymap_coordinates(nside, R, boxsize):
    """Generate 3D pixel positions at a given radius in box units."""
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), )
    pos = R * np.vstack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)]).T

    # Move to box units and center
    pos /= boxsize
    pos += 0.5
    # Quijote expects float32, otherwise it will crash
    return pos.astype(np.float32)


def make_skymap(radvel_field, map_pos):
    """
    Make a skymap of the radial velocity field at the given 3D positions which
    correspond to the pixels.
    """
    return csiborgtools.field.evaluate_cartesian_cic(
        radvel_field, pos=map_pos, smooth_scales=None)


def main(nsims, nside, radii, boxsize, MAS, grid, fname):
    """Calculate the sky maps and C_ell."""
    # 3D pixel positions at each radius in box units
    map_pos = [skymap_coordinates(nside, R, boxsize) for R in radii]
    ell_max = 16

    print(f"Writing to `{fname}`...")
    f = File(fname, 'w')
    f.create_dataset("ell", data=np.arange(ell_max + 1))
    f.create_dataset("radii", data=radii)

    for nsim in tqdm(nsims, desc="Simulations"):
        radvel_field = load_field(nsim, MAS, grid, paths)

        grp = f.create_group(f"nsim_{str(nsim).zfill(5)}")
        C_ell = np.zeros((len(radii), ell_max + 1))

        for n in range(len(radii)):
            skymap = make_skymap(radvel_field, map_pos[n])
            C_ell[n] = hp.sphtfunc.anafast(skymap, lmax=ell_max)
            grp.create_dataset(f"skymap_{n}", data=skymap)

        grp.create_dataset("C_ell", data=C_ell)

    print(f"Closing `{fname}`.")
    f.close()


if __name__ == "__main__":
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    nside = 256
    boxsize = 1000
    MAS = "PCS"
    grid = 512
    radii = np.linspace(50, 500, 10)
    fname = "/mnt/extraspace/rstiskalek/BBF/Quijote_Cell/C_ell_fiducial.h5"
    nsims = list(range(50))

    main(nsims, nside, radii, boxsize, MAS, grid, fname)
