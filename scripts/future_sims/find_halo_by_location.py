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
Find the most massive halo near a specific location on the sky in galactic
coordinates.
"""
import argparse
import os

import csiborgtools
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Find most massive halo near a sky location')
parser.add_argument('--steps', type=int, nargs='+',
                    default=[i for i in range(50)],
                    help='Simulation step numbers to process')
parser.add_argument('--snap', type=int, default=4,
                    help='Snapshot group number (default: 4)')
args = parser.parse_args()

# Configuration
steps = args.steps
snap = args.snap
# Target location and search parameters
target_l = 325  # Galactic longitude in degrees
target_b = -3  # Galactic latitude in degrees
target_dist = 50.0  # Distance in cMpc/h
ang_tol = 25.0  # Angular tolerance in degrees
dist_tol = 10.0  # Distance tolerance in cMpc/h
min_mass = 1e14  # Minimum halo mass in Msun/h
boxsize = 681.0  # cMpc/h
box_center = np.array([boxsize / 2, boxsize / 2, boxsize / 2])

# Snapshot index mapping for CSiBORG3 reader
snapshot_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4}
nsnap = snapshot_map.get(snap, snap)

# Base path pattern
base_path = ("/mnt/home/rstiskalek/ceph/CSiBORG/"
             "2MPP_MULTIBIN_N256_DES_V2/N256_future")

# Initialize paths object
paths = csiborgtools.read.Paths(**csiborgtools.paths_rusty)

# Setup cache and results directories
script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "cache")
results_dir = os.path.join(script_dir, "results")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

print("=" * 80)
print("HALO FINDER BY SKY LOCATION")
print("=" * 80)
print(f"Target location: l={target_l:.2f}°, b={target_b:.2f}°")
print(f"Target distance: {target_dist:.1f} cMpc/h")
print(f"Angular tolerance: ±{ang_tol:.1f}°")
print(f"Distance tolerance: ±{dist_tol:.1f} cMpc/h")
print(f"Minimum mass: {min_mass:.2e} Msun/h")
print(f"Snapshot: {snap:03d} (nsnap={nsnap})")
print(f"Processing {len(steps)} realization(s): {steps}")
print("=" * 80)


def angular_separation(l1, b1, l2, b2):
    """
    Calculate angular separation between two points on the sky.

    Parameters
    ----------
    l1, b1 : float or array
        Galactic longitude and latitude of first point(s) in degrees
    l2, b2 : float or array
        Galactic longitude and latitude of second point(s) in degrees

    Returns
    -------
    sep : float or array
        Angular separation in degrees
    """
    # Convert to radians
    l1_rad = np.radians(l1)
    b1_rad = np.radians(b1)
    l2_rad = np.radians(l2)
    b2_rad = np.radians(b2)

    # Haversine formula
    dlat = b2_rad - b1_rad
    dlon = l2_rad - l1_rad

    a = (np.sin(dlat / 2)**2 +
         np.cos(b1_rad) * np.cos(b2_rad) * np.sin(dlon / 2)**2)
    c = 2 * np.arcsin(np.sqrt(a))

    return np.degrees(c)


# Storage for results
results = []

# Loop over steps (realizations)
for step in steps:
    print(f"\nProcessing realization {step}...")

    # Read FoF catalogue
    fof_path = (f"{base_path}/step_{step}/output/groups_{snap:03d}/"
                f"fof_subhalo_tab_{snap:03d}.hdf5")

    try:
        catalogue = csiborgtools.read.CSiBORG3Catalogue(
            nsim=0, nsnap=nsnap, paths=paths, fpath_override=fof_path,
            verbose=False)
    except Exception as e:
        print(f"  Warning: Failed to load catalogue for step {step}: {e}")
        continue

    # Get halo properties
    group_mass = catalogue.Group_M_Crit200  # Msun/h
    group_coords = catalogue.coordinates  # Halo positions
    num_groups = len(catalogue)

    print(f"  Total halos: {num_groups:,}")

    # Apply minimum mass cut
    mass_mask = group_mass >= min_mass
    group_mass = group_mass[mass_mask]
    group_coords = group_coords[mass_mask]

    print(f"  Halos above mass threshold: {len(group_mass):,}")

    if len(group_mass) == 0:
        print("  No halos found above mass threshold")
        continue

    # Compute distances from box center
    group_distances = np.sqrt(np.sum((group_coords - box_center)**2, axis=1))

    # Convert halo positions to galactic coordinates
    radec = csiborgtools.cartesian_to_radec(group_coords, origin=box_center)
    ra, dec = radec[:, 1], radec[:, 2]
    gal_l, gal_b = csiborgtools.radec_to_galactic(ra, dec)

    # Find halos within angular and distance tolerance
    ang_sep = angular_separation(target_l, target_b, gal_l, gal_b)
    dist_diff = np.abs(group_distances - target_dist)

    # Create mask for halos within tolerance
    within_tol = (ang_sep <= ang_tol) & (dist_diff <= dist_tol)
    n_candidates = np.sum(within_tol)

    print(f"  Halos within tolerance: {n_candidates}")

    if n_candidates == 0:
        print("  No halos found within tolerance")
        results.append({
            'step': step,
            'halo_id': -1,
            'mass': np.nan,
            'distance': np.nan,
            'gal_l': np.nan,
            'gal_b': np.nan,
            'ang_sep': np.nan,
            'dist_diff': np.nan
        })
        continue

    # Get indices of candidate halos (need to map back to original indices)
    original_indices = np.where(mass_mask)[0]
    candidate_indices = original_indices[within_tol]

    # Among candidates, find the most massive
    candidate_masses = group_mass[within_tol]
    most_massive_idx = np.argmax(candidate_masses)

    # Get the halo ID (index in full catalogue)
    halo_id = candidate_indices[most_massive_idx]

    # Store results
    results.append({
        'step': step,
        'halo_id': int(halo_id),
        'mass': group_mass[within_tol][most_massive_idx],
        'distance': group_distances[within_tol][most_massive_idx],
        'gal_l': gal_l[within_tol][most_massive_idx],
        'gal_b': gal_b[within_tol][most_massive_idx],
        'ang_sep': ang_sep[within_tol][most_massive_idx],
        'dist_diff': dist_diff[within_tol][most_massive_idx]
    })

    print(f"  Most massive halo: ID={halo_id}, "
          f"M={results[-1]['mass']:.2e} Msun/h")
    print(f"    Position: l={results[-1]['gal_l']:.2f}°, "
          f"b={results[-1]['gal_b']:.2f}°, "
          f"d={results[-1]['distance']:.1f} cMpc/h")
    print(f"    Offsets: Δθ={results[-1]['ang_sep']:.2f}°, "
          f"Δd={results[-1]['dist_diff']:.1f} cMpc/h")

# Print summary table
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
print(f"{'Step':>6} {'Halo ID':>10} {'Mass [Msun/h]':>15} "
      f"{'Dist [cMpc/h]':>14} {'l [deg]':>10} {'b [deg]':>10} "
      f"{'Δθ [deg]':>10} {'Δd [cMpc/h]':>12}")
print("-" * 80)

for result in results:
    if result['halo_id'] == -1:
        print(f"{result['step']:6d} {'N/A':>10} {'N/A':>15} "
              f"{'N/A':>14} {'N/A':>10} {'N/A':>10} "
              f"{'N/A':>10} {'N/A':>12}")
    else:
        print(f"{result['step']:6d} {result['halo_id']:10d} "
              f"{result['mass']:15.2e} "
              f"{result['distance']:14.2f} "
              f"{result['gal_l']:10.2f} {result['gal_b']:10.2f} "
              f"{result['ang_sep']:10.2f} {result['dist_diff']:12.2f}")

print("=" * 80)

# Save results table to text file
txt_file = os.path.join(
    results_dir,
    f"halo_finder_l{target_l:.0f}_b{target_b:.0f}_d{target_dist:.0f}.txt"
)

with open(txt_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("HALO FINDER BY SKY LOCATION - RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Target location: l={target_l:.2f}°, b={target_b:.2f}°\n")
    f.write(f"Target distance: {target_dist:.1f} cMpc/h\n")
    f.write(f"Angular tolerance: ±{ang_tol:.1f}°\n")
    f.write(f"Distance tolerance: ±{dist_tol:.1f} cMpc/h\n")
    f.write(f"Minimum mass: {min_mass:.2e} Msun/h\n")
    f.write(f"Snapshot: {snap:03d} (nsnap={nsnap})\n")
    f.write(f"Realizations processed: {len(steps)}\n")
    f.write("=" * 80 + "\n\n")

    f.write(f"{'Step':>6} {'Halo ID':>10} {'Mass [Msun/h]':>15} "
            f"{'Dist [cMpc/h]':>14} {'l [deg]':>10} {'b [deg]':>10} "
            f"{'Δθ [deg]':>10} {'Δd [cMpc/h]':>12}\n")
    f.write("-" * 80 + "\n")

    for result in results:
        if result['halo_id'] == -1:
            f.write(f"{result['step']:6d} {'N/A':>10} {'N/A':>15} "
                    f"{'N/A':>14} {'N/A':>10} {'N/A':>10} "
                    f"{'N/A':>10} {'N/A':>12}\n")
        else:
            f.write(f"{result['step']:6d} {result['halo_id']:10d} "
                    f"{result['mass']:15.2e} "
                    f"{result['distance']:14.2f} "
                    f"{result['gal_l']:10.2f} {result['gal_b']:10.2f} "
                    f"{result['ang_sep']:10.2f} {result['dist_diff']:12.2f}\n")

    f.write("=" * 80 + "\n")

print(f"\nResults table saved to: {txt_file}")

# Save results to numpy file
output_file = os.path.join(
    cache_dir,
    f"halo_finder_l{target_l:.0f}_b{target_b:.0f}_d{target_dist:.0f}.npz"
)

# Convert results to arrays for saving
save_data = {
    'steps': np.array([r['step'] for r in results]),
    'halo_ids': np.array([r['halo_id'] for r in results]),
    'masses': np.array([r['mass'] for r in results]),
    'distances': np.array([r['distance'] for r in results]),
    'gal_l': np.array([r['gal_l'] for r in results]),
    'gal_b': np.array([r['gal_b'] for r in results]),
    'ang_sep': np.array([r['ang_sep'] for r in results]),
    'dist_diff': np.array([r['dist_diff'] for r in results]),
    'target_l': target_l,
    'target_b': target_b,
    'target_dist': target_dist,
    'ang_tol': ang_tol,
    'dist_tol': dist_tol,
    'min_mass': min_mass,
    'snap': snap,
}

np.savez(output_file, **save_data)
print(f"Numpy data saved to: {output_file}")
