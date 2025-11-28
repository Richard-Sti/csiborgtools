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
Track particles from the box center at a=1 to their halos at a=10.
"""
import numpy as np
import csiborgtools
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Track particles from box center to halos')
parser.add_argument('--steps', type=int, nargs='+',
                    default=[i for i in range(10)],
                    help='Simulation step numbers (0-49), can specify multiple')  # noqa
parser.add_argument('--unbound-only', action='store_true',
                    help='Only track particles that are unbound at a=1')
args = parser.parse_args()

# Configuration
steps = args.steps
radii = [5.0, 7.5, 10, 15, 20, 30]  # cMpc/h - fixed radii
boxsize = 681.0  # cMpc/h
box_center = np.array([boxsize / 2, boxsize / 2, boxsize / 2])

# Base path pattern
base_path = ("/mnt/home/rstiskalek/ceph/CSiBORG/"
             "2MPP_MULTIBIN_N256_DES_V2/N256_far_future")

# Initialize paths object
paths = csiborgtools.read.Paths(**csiborgtools.paths_rusty)

# Setup cache directory
script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)

print(f"Processing {len(steps)} step(s): {steps}")
print(f"Radii: {radii} cMpc/h")
print(f"Unbound particles only at a=1: {args.unbound_only}")
print(f"Box center: {box_center} cMpc/h")
print("=" * 80)

# Dictionary to store results for all steps
all_steps_results = {
    'steps': np.array(steps),
    'unbound_only': args.unbound_only,
    'radii': np.array(radii),
}

# Loop over steps
for step in steps:
    print(f"\n{'=' * 80}")
    print(f"PROCESSING STEP {step}")
    print(f"{'=' * 80}")

    # Step 1: Load a=1 snapshot and identify particles near box center
    print("\n1. Loading a=1 snapshot (group 001)...")
    snap_a1_path = (f"{base_path}/step_{step}/output/snapdir_001/"
                    f"snapshot_001.0.hdf5")

    snap_a1 = csiborgtools.read.CSiBORG3Snapshot(
        0, 1, paths, fpath_override=snap_a1_path)

    print("   Reading particle coordinates and IDs...")
    coords_a1 = snap_a1.coordinates()
    pids_a1 = snap_a1.particle_ids()

    print(f"   Total particles at a=1: {len(pids_a1):,}")

    # Filter for unbound particles only if requested
    if args.unbound_only:
        print("   Identifying unbound particles at a=1...")
        cat_a1_path = (f"{base_path}/step_{step}/output/groups_001/"
                       f"fof_subhalo_tab_001.hdf5")
        cat_a1 = csiborgtools.read.CSiBORG3Catalogue(
            nsim=0, nsnap=1, paths=paths, fpath_override=cat_a1_path,
            verbose=False)

        # Get halo IDs for all particles at a=1
        hids_a1 = snap_a1.particle_halo_ids(False)

        # Keep only unbound particles (halo_id == -1)
        unbound_mask = hids_a1 == -1
        coords_a1 = coords_a1[unbound_mask]
        pids_a1 = pids_a1[unbound_mask]

        print(f"   Unbound particles at a=1: {len(pids_a1):,} "
              f"({100 * len(pids_a1) / len(hids_a1):.1f}%)")

    # Compute distances from box center
    print("   Computing distances from box center...")
    distances = np.sqrt(np.sum((coords_a1 - box_center)**2, axis=1))

    # Identify particles within each radius
    tracked_pids_dict = {}

    print("\n   Identifying particles within each radius:")
    for radius in radii:
        mask_center = distances < radius
        tracked_pids_dict[radius] = pids_a1[mask_center]
        print(f"   Radius {radius:6.1f} cMpc/h "
              f"→ {len(tracked_pids_dict[radius]):,} particles")

    # Clean up coordinates
    del coords_a1

    # Step 2: Load a=10 snapshot
    print("\n2. Loading a=100 snapshot (group 006)...")
    snap_a10_path = (f"{base_path}/step_{step}/output/snapdir_006/"
                     f"snapshot_006.0.hdf5")

    snap_a10 = csiborgtools.read.CSiBORG3Snapshot(
        0, 4, paths, fpath_override=snap_a10_path)

    print("   Reading particle IDs at a=10...")
    pids_a10 = snap_a10.particle_ids()
    print(f"   Total particles at a=10: {len(pids_a10):,}")

    # Step 3: Read FoF catalogue to get particle-to-halo mapping
    print("\n3. Reading FoF catalogue at a=10...")
    fof_path = (f"{base_path}/step_{step}/output/groups_006/"
                f"fof_subhalo_tab_006.hdf5")

    # Use CSiBORG3Catalogue to read the catalogue
    catalogue = csiborgtools.read.CSiBORG3Catalogue(
        nsim=0, nsnap=4, paths=paths, fpath_override=fof_path, verbose=False)

    # Read group information using the catalogue class
    group_len = catalogue.npart
    group_mass = catalogue.Group_M_Crit200  # Already in Msun/h
    group_coords = catalogue.coordinates  # Halo positions
    num_groups = len(catalogue)

    # Compute distances of halos from box center
    group_distances = np.sqrt(np.sum((group_coords - box_center)**2, axis=1))

    # Convert halo positions to Galactic coordinates
    # First convert Cartesian to RA/Dec with origin at box center
    radec = csiborgtools.cartesian_to_radec(group_coords, origin=box_center)
    ra, dec = radec[:, 1], radec[:, 2]  # radec returns [distance, RA, Dec]

    # Then convert RA/Dec to Galactic coordinates
    gal_l, gal_b = csiborgtools.radec_to_galactic(ra, dec)

    print(f"   Total number of groups (halos): {num_groups:,}")

    # Step 4: Create particle ID to halo ID mapping
    print("\n4. Creating particle-to-halo mapping...")

    # Use the new particle_halo_ids method to get halo IDs for all particles
    hids_all = snap_a10.particle_halo_ids(False)

    # Create dictionary mapping particle ID to halo ID
    # Only include particles that are in halos (hid != -1)
    pid_to_halo = {
        pid: hid for pid, hid in zip(pids_a10, hids_all) if hid != -1}

    print(f"   Mapped {len(pid_to_halo):,} particles to halos")

    # Step 5: Match tracked particles to halos for each radius
    print("\n5. Matching tracked particles to halos at a=10...")
    results = {}

    for radius in radii:
        print(f"\n   Processing radius {radius} cMpc/h:")
        tracked_pids = tracked_pids_dict[radius]
        halo_particle_counts = {}
        field_particles = 0

        for pid in tracked_pids:
            if pid in pid_to_halo:
                halo_id = pid_to_halo[pid]
                halo_particle_counts[halo_id] = (
                    halo_particle_counts.get(halo_id, 0) + 1)
            else:
                field_particles += 1

        num_halos_matched = len(halo_particle_counts)
        particles_in_halos = len(tracked_pids) - field_particles
        fraction_in_halos = (particles_in_halos / len(tracked_pids)
                             if len(tracked_pids) > 0 else 0)

        print(f"      Tracked particles in halos: {particles_in_halos:,} "
              f"({fraction_in_halos * 100:.1f}%)")
        print(f"      Tracked particles in field: {field_particles:,} "
              f"({(1 - fraction_in_halos) * 100:.1f}%)")
        print(f"      Number of halos: {num_halos_matched:,}")

        # Store results for this radius
        results[radius] = {
            'tracked_pids': tracked_pids,
            'halo_particle_counts': halo_particle_counts,
            'field_particles': field_particles,
            'fraction_in_halos': fraction_in_halos
        }

    # Step 6: Display results for all radii
    print("\n6. Top 20 halos by tracked particle count:")

    for radius in radii:
        print(f"\n   Radius = {radius:.1f} cMpc/h:")
        print("   " + "=" * 100)
        print(f"   {'Halo ID':>8} {'N_tracked':>10} {'M200c [Msun/h]':>18} "
              f"{'Dist [cMpc/h]':>14} {'l [deg]':>10} {'b [deg]':>10} "
              f"{'Fraction':>10}")
        print("   " + "-" * 100)

        # Sort halos by particle count
        halo_particle_counts = results[radius]['halo_particle_counts']
        tracked_pids = results[radius]['tracked_pids']
        sorted_halos = sorted(halo_particle_counts.items(),
                              key=lambda x: x[1], reverse=True)

        for i, (halo_id, count) in enumerate(sorted_halos[:20]):
            mass = group_mass[halo_id]
            distance = group_distances[halo_id]
            ell = gal_l[halo_id]
            b = gal_b[halo_id]
            fraction = count / len(tracked_pids) if len(tracked_pids) > 0 else 0  # noqa
            print(f"   {halo_id:8d} {count:10d} {mass:18.2e} "
                  f"{distance:14.2f} {ell:10.2f} {b:10.2f} {fraction:10.4f}")

        print("   " + "=" * 100)

    # Store results for this step
    step_data = {}

    # Store halo catalog info (only once - same for all steps at a=10)
    if step == steps[0]:
        all_steps_results['group_mass'] = group_mass
        all_steps_results['group_distances'] = group_distances
        all_steps_results['group_coords'] = group_coords
        all_steps_results['group_gal_l'] = gal_l
        all_steps_results['group_gal_b'] = gal_b

    # Add data for each radius
    for radius in radii:
        r_str = f"r{int(radius)}"
        result = results[radius]
        step_data[f'{r_str}_tracked_pids'] = result['tracked_pids']
        step_data[f'{r_str}_halo_particle_counts'] = np.array(
            list(result['halo_particle_counts'].items()))
        step_data[f'{r_str}_field_particles'] = result['field_particles']
        step_data[f'{r_str}_fraction_in_halos'] = result['fraction_in_halos']

    # Store this step's data in the main dictionary
    all_steps_results[f'step{step}'] = step_data

# Step 7: Save all results to a single file
print(f"\n{'=' * 80}")
print("SAVING RESULTS")
print(f"{'=' * 80}")

suffix = "_unbound" if args.unbound_only else ""
if len(steps) == 1:
    output_file = os.path.join(
        cache_dir, f"tracked_particles_step{steps[0]}{suffix}.npz")
else:
    output_file = os.path.join(
        cache_dir,
        f"tracked_particles_steps{min(steps)}-{max(steps)}{suffix}.npz")

np.savez(output_file, **all_steps_results)

print(f"\nResults saved to: {output_file}")
print(f"Saved data for {len(steps)} step(s): {steps}")
print(f"Each step contains {len(radii)} radii: {radii} cMpc/h")
