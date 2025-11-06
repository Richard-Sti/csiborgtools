# Copyright (C) 2025 Richard Stiskalek
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
Compute M200c and R200c for CSiBORG1 FOF halos using shrinking sphere.
"""
import csiborgtools
import hdf5plugin  # noqa
import numpy as np
from h5py import File
from mpi4py import MPI
from tqdm import trange


if __name__ == "__main__":
    # User settings
    mass_threshold = 1e13  # Msun / h
    boxsize = 677.7  # Mpc / h
    h = 1.0
    output_dir = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/SOcat"
    fname_stem = "csiborg1"

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # File I/O
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    simulations = paths.get_ics("csiborg1")

    if rank == 0:
        print(f"Processing {len(simulations)} simulations using {size} "
              f"MPI processes per simulation")

    for i, nsim in enumerate(simulations):
        fname_out = f"{output_dir}/{fname_stem}_{nsim}.hdf5"

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Processing simulation {nsim} ({i+1}/{len(simulations)}, "
                  f"{len(simulations) - i - 1} remaining)")
            print(f"{'='*60}")

        halocat = csiborgtools.read.CSiBORG1Catalogue(nsim, paths=paths)

        snapnum = max(paths.get_snapshots(nsim, "csiborg1"))
        snapcat = csiborgtools.read.CSiBORG1Snapshot(
            nsim, snapnum, paths=paths, flip_xz=True)

        mask = halocat["totmass"] > mass_threshold
        selected_idx = np.where(mask)[0]

        if rank == 0:
            print(f"Processing {len(selected_idx)} halos with mass > "
                  f"{mass_threshold:.0e} Msun/h")
            np.random.shuffle(selected_idx)

        selected_idx = comm.bcast(selected_idx, root=0)
        my_halos = np.array_split(selected_idx, size)[rank]
        n_my_halos = len(my_halos)

        print(f"Rank {1 + rank}/{size}: processing {n_my_halos} halos")

        m200c_vals = np.full(n_my_halos, np.nan, dtype=np.float32)
        r200c_vals = np.full(n_my_halos, np.nan, dtype=np.float32)
        centers = np.full((n_my_halos, 3), np.nan, dtype=np.float32)
        total_mass = np.full(n_my_halos, np.nan, dtype=np.float32)
        fof_pos = np.full((n_my_halos, 3), np.nan, dtype=np.float32)

        for i in trange(n_my_halos, disable=size > 1):
            hid = halocat.index[my_halos[i]]

            pos = snapcat.halo_coordinates(hid)
            mass = snapcat.halo_masses(hid)

            halo = csiborgtools.halo.Halo(pos, mass)
            cm = halo.compute_center(
                boxsize, periodic=False, shrink_factor=0.95,
                npart_min=50)
            M200c, R200c = halo.compute_r200c(cm, h=h, boxsize=boxsize)

            m200c_vals[i] = M200c
            r200c_vals[i] = R200c
            centers[i] = cm
            total_mass[i] = mass.sum()
            fof_pos[i] = halocat["cartesian_pos"][my_halos[i]]

        all_m200c = comm.gather(m200c_vals, root=0)
        all_r200c = comm.gather(r200c_vals, root=0)
        all_centers = comm.gather(centers, root=0)
        all_total_mass = comm.gather(total_mass, root=0)
        all_fof_pos = comm.gather(fof_pos, root=0)

        if rank == 0:
            m200c_vals = np.concatenate(all_m200c)
            r200c_vals = np.concatenate(all_r200c)
            centers = np.vstack(all_centers)
            total_mass = np.concatenate(all_total_mass)
            fof_pos = np.vstack(all_fof_pos)

            # Sort by total mass descending
            sort_idx = np.argsort(total_mass)[::-1]
            m200c_vals = m200c_vals[sort_idx]
            r200c_vals = r200c_vals[sort_idx]
            centers = centers[sort_idx]
            total_mass = total_mass[sort_idx]
            fof_pos = fof_pos[sort_idx]

            print(f"Saving results to {fname_out}")

            with File(fname_out, 'w') as f:
                f.create_dataset("M200c", data=m200c_vals)
                f.create_dataset("R200c", data=r200c_vals)
                f.create_dataset("Position", data=centers)
                f.create_dataset("TotalMass", data=total_mass)
                f.create_dataset("FOFPosition", data=fof_pos)
                f.attrs["nsim"] = nsim
                f.attrs["boxsize"] = boxsize
                f.attrs["h"] = h
                f.attrs["mass_threshold"] = mass_threshold

            print(f"Processed {len(m200c_vals)} halos successfully")
