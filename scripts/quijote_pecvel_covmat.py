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
from csiborgtools.field import evaluate_cartesian_cic
from h5py import File
from tqdm import tqdm


def load_field(nsim, MAS, grid, paths):
    """Load the precomputed velocity field from the Quijote simulations."""
    reader = csiborgtools.read.QuijoteField(nsim, paths)
    return reader.velocity_field(MAS, grid)


def skymap_coordinates(nside, R):
    """Generate 3D pixel positions at a given radius."""
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), )
    pos = R * np.vstack([np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)]).T

    # Quijote expects float32, otherwise it will crash
    return pos.astype(np.float32)


def make_radvel_skymap(velocity_field, pos, observer, boxsize):
    """
    Make a skymap of the radial velocity field at the given 3D positions which
    correspond to the pixels.
    """
    # Velocities on the shell
    Vx, Vy, Vz = [evaluate_cartesian_cic(velocity_field[i], pos=pos / boxsize,
                                         smooth_scales=None) for i in range(3)]

    # Observer velocity
    obs = np.asarray(observer).reshape(1, 3) / boxsize
    Vx_obs, Vy_obs, Vz_obs = [evaluate_cartesian_cic(
        velocity_field[i], pos=obs, smooth_scales=None)[0] for i in range(3)]

    # Subtract observer velocity
    Vx -= Vx_obs
    Vy -= Vy_obs
    Vz -= Vz_obs

    # Radial velocity
    norm_pos = pos - observer
    norm_pos /= np.linalg.norm(norm_pos, axis=1).reshape(-1, 1)
    Vrad = Vx * norm_pos[:, 0] + Vy * norm_pos[:, 1] + Vz * norm_pos[:, 2]

    return Vrad


def main(nsims, observers, nside, ell_max, radii, boxsize, MAS, grid, fname):
    """Calculate the sky maps and C_ell."""
    # 3D pixel positions at each radius in box units
    map_pos = [skymap_coordinates(nside, R) for R in radii]

    print(f"Writing to `{fname}`...")
    f = File(fname, 'w')
    f.create_dataset("ell", data=np.arange(ell_max + 1))
    f.create_dataset("radii", data=radii)
    f.attrs["num_simulations"] = len(nsims)
    f.attrs["num_observers"] = len(observers)
    f.attrs["num_radii"] = len(radii)
    f.attrs["npix_per_map"] = hp.nside2npix(nside)

    for nsim in tqdm(nsims, desc="Simulations"):
        grp_sim = f.create_group(f"nsim_{nsim}")
        velocity_field = load_field(nsim, MAS, grid, paths)

        for n in range(len(observers)):
            grp_observer = grp_sim.create_group(f"observer_{n}")

            for i in range(len(radii)):
                pos = map_pos[i] + observers[n]

                skymap = make_radvel_skymap(velocity_field, pos, observers[n],
                                            boxsize)
                C_ell = hp.sphtfunc.anafast(skymap, lmax=ell_max)

                grp_observer.create_dataset(f"skymap_{i}", data=skymap)
                grp_observer.create_dataset(f"C_ell_{i}", data=C_ell)

    print(f"Closing `{fname}`.")
    f.close()


if __name__ == "__main__":
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    MAS = "PCS"
    grid = 512
    nside = 256
    ell_max = 16
    boxsize = 1000
    Rmax = 200
    radii = np.linspace(100, 150, 5)
    fname = "/mnt/extraspace/rstiskalek/BBF/Quijote_Cell/C_ell_fiducial.h5"
    nsims = list(range(50))
    observers = csiborgtools.read.fiducial_observers(boxsize, Rmax)

    main(nsims, observers, nside, ell_max, radii, boxsize, MAS, grid, fname)
