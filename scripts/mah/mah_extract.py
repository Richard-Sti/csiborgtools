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
Script to extract the main progenitor branch information from the CSiBORG
simulations and save redshift and mass assembly histories of massive halos
to an HDF5 file.
"""
import numpy as np
import ytree
from tqdm import tqdm
import h5py
import argparse


def load_trees(input_path):
    """Load tree data from a ytree HDF5 file."""
    return ytree.load(input_path)


def select_massive_halos(trees, mass_def, mass_threshold, unit_scale):
    """Select indices of halos with mass above a threshold."""
    if mass_def not in trees.field_list:
        raise ValueError(f"Mass field '{mass_def}' not found in tree fields.")

    M = trees[mass_def].value * unit_scale
    return np.where(M > mass_threshold)[0]


def extract_mass_assembly_histories(trees, ks, mass_def, unit_scale, nsteps):
    """Extract redshift and mass histories for given halo indices."""
    redshifts = np.full((len(ks), nsteps), np.nan)
    masses = np.full((len(ks), nsteps), np.nan)

    for i, k in enumerate(tqdm(ks, desc="Processing halos")):
        Mi = trees[k]["prog", mass_def].value * unit_scale
        zi = trees[k]["prog", "redshift"].value

        N = len(Mi)
        if N > nsteps:
            print(f"Warning: Tree {k} has {N} steps, truncating to {nsteps}")

        redshifts[i, :min(N, nsteps)] = zi[:nsteps]
        masses[i, :min(N, nsteps)] = Mi[:nsteps]

    return redshifts, masses


def save_to_hdf5(output_path, redshifts, masses, root_pos):
    """Save redshift and mass data to an HDF5 file."""
    with h5py.File(output_path, "w") as f:
        f.create_dataset("redshift", data=redshifts)
        f.create_dataset("mass", data=masses)
        f.create_dataset("root_pos", data=root_pos)
    print(f"Saved MAH data to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Path to ytree HDF5 input file.")
    parser.add_argument("--output_file", help="Path to HDF5 output file.")
    parser.add_argument("--mass_def", type=str, default="SubhaloMass",
                        help="Mass definition to use (default: SubhaloMass).")
    parser.add_argument("--mass_threshold", type=float, default=1e14,
                        help="Minimum halo mass to include.")
    parser.add_argument("--nsteps", type=int, default=131,
                        help="Max number of steps per halo.")
    parser.add_argument("--unit_scale", type=float, default=1e10,
                        help="Scale factor for mass units.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    trees = load_trees(args.input_file)
    ks = select_massive_halos(
        trees, args.mass_def, args.mass_threshold, args.unit_scale)
    redshifts, masses = extract_mass_assembly_histories(
        trees, ks, args.mass_def, args.unit_scale, args.nsteps)

    root_pos = np.vstack([trees[k]["position"].value for k in ks])
    save_to_hdf5(args.output_file, redshifts, masses, root_pos)
