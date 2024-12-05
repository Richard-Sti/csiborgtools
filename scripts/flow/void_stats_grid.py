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
Script to calculate the void enclosed density, monopole and bulk flow for all
void size and observer offset combinations.
"""
from argparse import ArgumentParser

import numpy as np
from csiborgtools.flow import (load_void_size_variation, void_bulk_flow,
                               void_monopole)
from h5py import File


def main(profile):
    print(f"Calculating void stats for profile `{profile}`.")

    ngrid = 64
    fname_out = f"/mnt/extraspace/rstiskalek/catalogs/IndranilVoid/SizeVariation_new/void_stats_{profile}.hdf5"  # noqa

    # Load the data
    sizes, rLG, vx = load_void_size_variation(profile, "vx")
    sizes, rLG, vy = load_void_size_variation(profile, "vy")
    sizes, rLG, vrad = load_void_size_variation(profile, "vrad")
    sizes, rLG, rho = load_void_size_variation(profile, "density")

    # The void grids
    r_grid = np.arange(0, 251).astype(np.float32)
    phi_grid = np.arange(0, 181).astype(np.float32)
    # Compute over 100 steps
    r = np.linspace(0, r_grid.max(), 100)

    num_rLG = len(rLG)
    num_sizes = len(sizes)

    # Allocate arrays
    enclosed_density = np.zeros((num_sizes, num_rLG, len(r)))
    monopole = np.zeros((num_sizes, num_rLG, len(r)))
    bulk_flow = np.zeros((num_sizes, num_rLG, len(r), 3))

    for i in range(num_sizes):
        for j in range(num_sizes):
            enclosed_density[i, j, ...] = void_monopole(
                r, rho[i, j], ngrid, r_grid, phi_grid)
            monopole[i, j, ...] = void_monopole(
                r, vrad[i, j], ngrid, r_grid, phi_grid)
            bulk_flow[i, j, ...] = void_bulk_flow(
                r, vx[i, j], vy[i, j], ngrid, r_grid, phi_grid, in_icrs=True)

    with File(fname_out, "w") as f:
        # Write down the grid
        grp = f.create_group("grid")
        grp.create_dataset("sizes", data=sizes)
        grp.create_dataset("rLG", data=rLG)
        grp.create_dataset("r_grid", data=r_grid)
        grp.create_dataset("phi_grid", data=phi_grid)

        # Write down the void stats
        grp = f.create_group("void_stats")
        f.create_dataset("r", data=r)
        f.create_dataset("enclosed_density", data=enclosed_density)
        f.create_dataset("monopole", data=monopole)
        f.create_dataset("bulk_flow", data=bulk_flow)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("profile", type=str)
    args = parser.parse_args()

    main(args.profile)
