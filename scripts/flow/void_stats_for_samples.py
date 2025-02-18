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
from h5py import File
from joblib import Parallel, delayed
from tqdm import trange

from csiborgtools import radec_to_cartesian
from csiborgtools.flow import (load_void_size_variation, void_bulk_flow,
                               void_monopole)


def read_void_info(fname):
    with File(fname, "r") as f:
        grp = f["samples"]
        if "Vext_mag" in grp:
            Vext = grp["Vext_mag"][...]
            Vext = radec_to_cartesian(np.vstack([
                Vext,
                np.rad2deg(grp["Vext_phi"][...]),
                np.rad2deg(np.pi / 2 - np.arccos(grp["Vext_cos_theta"][...])),
            ]).T)
        else:
            Vext = grp["Vext"][...]

        rLG = grp["rLG"][...]

        if "SizeVar" in fname:
            void_size = grp["void_size"][...]
        else:
            void_size = np.ones_like(rLG)

    return void_size, rLG, Vext


def stats_per_sample(void_size, rLG, Vext, void_data):
    is_negative_Roffset = rLG < 0

    i = np.argmin(np.abs(void_size - void_data["sizes"]))
    j = np.argmin(np.abs(np.abs(rLG) - void_data["rLG"]))

    rho = void_monopole(
        void_data["r"], void_data["rho"][i, j], void_data["ngrid"],
        void_data["r_grid"], void_data["phi_grid"], Vext, is_negative_Roffset,
        verbose=False)
    vmono = void_monopole(
        void_data["r"], void_data["vrad"][i, j], void_data["ngrid"],
        void_data["r_grid"], void_data["phi_grid"], Vext, is_negative_Roffset,
        verbose=False)
    vbulk = void_bulk_flow(
        void_data["r"], void_data["vx"][i, j], void_data["vx"][i, j],
        void_data["ngrid"], void_data["r_grid"], void_data["phi_grid"],
        Vext, is_negative_Roffset, verbose=False)

    return rho, vmono, vbulk


def main(fname, profile, njobs=1):
    print(f"Calculating void stats for profile `{profile}` with samples "
          f"from `{fname}`.\n")

    ngrid = 64
    fname_out = fname.replace(".hdf5", "_stats.hdf5")

    # Load the data
    sizes, rLG, vx = load_void_size_variation(profile, "vx")
    sizes, rLG, vy = load_void_size_variation(profile, "vy")
    sizes, rLG, vrad = load_void_size_variation(profile, "vrad")
    sizes, rLG, rho = load_void_size_variation(profile, "density")

    # The void grids
    r_grid = np.arange(0, 401).astype(np.float32)
    phi_grid = np.arange(0, 181).astype(np.float32)

    # Compute the void statistics over 100 steps
    r = np.linspace(0, r_grid.max(), 100)

    void_data = {"sizes": sizes, "rLG": rLG,
                 "vx": vx, "vy": vy, "vrad": vrad, "rho": rho,
                 "r_grid": r_grid, "phi_grid": phi_grid, "r": r,
                 "ngrid": ngrid}

    void_size, rLG, Vext = read_void_info(fname)

    # Allocate arrays
    rho = np.full((len(rLG), len(r)), np.nan)
    vmono = np.full((len(rLG), len(r)), np.nan)
    vbulk = np.full((len(rLG), len(r), 3), np.nan)

    # # Compute the void stats for each samples.
    # for i in trange(len(rLG), desc="Samples"):
    #     rho[i], vmono[i], vbulk[i] = stats_per_sample(
    #         void_size[i], rLG[i], Vext[i], void_data)
    print(f"Computing with {njobs} jobs.")
    results = Parallel(n_jobs=args.njobs)(
        delayed(stats_per_sample)(
            void_size[i], rLG[i], Vext[i], void_data)
        for i in trange(len(rLG), total=len(rLG), desc="Samples",
                        unit="sample"))

    # Store results correctly
    for i, (ri, vmi, vbi) in enumerate(results):
        rho[i], vmono[i], vbulk[i] = ri, vmi, vbi

    # Write the results
    print(f"\nWriting the results to `{fname_out}`.")
    with File(fname_out, "w") as f:
        # Write down the grid
        grp = f.create_group("grid")
        grp.create_dataset("sizes", data=sizes)
        grp.create_dataset("rLG", data=rLG)
        grp.create_dataset("r_grid", data=r_grid)
        grp.create_dataset("phi_grid", data=phi_grid)

        # Write down the void stats
        grp = f.create_group("void_stats")
        grp.create_dataset("r", data=r)
        grp.create_dataset("enclosed_density", data=rho)
        grp.create_dataset("monopole", data=vmono)
        grp.create_dataset("bulk_flow", data=vbulk)
        grp.create_dataset("Vext", data=Vext)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("fname", type=str)
    parser.add_argument("--njobs", type=int, default=1)
    args = parser.parse_args()

    if "_gauss" in args.fname:
        profile = "gauss"
    elif "_exp" in args.fname:
        profile = "exp"
    elif "_mb" in args.fname:
        profile = "mb"
    else:
        raise ValueError("Unknown profile.")

    main(args.fname, profile, args.njobs)
