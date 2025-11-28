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
Trace particles from identified halos at final snapshot back to their halos
at initial snapshot.
"""
import argparse
import os

import csiborgtools
import numpy as np

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Trace halo particles back to initial snapshot')
parser.add_argument('--input-file', type=str, required=True,
                    help='Input .npz file from find_halo_by_location.py')
parser.add_argument('--snap-initial', type=int, default=1,
                    help='Initial snapshot group number (default: 1)')
parser.add_argument('--top-n-halos', type=int, default=100,
                    help='Number of top contributing halos to report (default: 100)')    # noqa
args = parser.parse_args()

# Configuration
input_file = args.input_file
snap_initial = args.snap_initial
top_n = args.top_n_halos
boxsize = 681.0  # cMpc/h
box_center = np.array([boxsize / 2, boxsize / 2, boxsize / 2])

# Snapshot index mapping for CSiBORG3 reader
snapshot_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4}
nsnap_initial = snapshot_map.get(snap_initial, snap_initial)

# Base path pattern
base_path = ("/mnt/home/rstiskalek/ceph/CSiBORG/"
             "2MPP_MULTIBIN_N256_DES_V2/N256_future")

# Initialize paths object
paths = csiborgtools.read.Paths(**csiborgtools.paths_rusty)

# Setup output directories
script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "cache")
results_dir = os.path.join(script_dir, "results")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

print("=" * 80)
print("HALO ORIGIN TRACER")
print("=" * 80)
print(f"Input file: {input_file}")
print(f"Initial snapshot: {snap_initial:03d} (nsnap={nsnap_initial})")
print(f"Top N halos to report: {top_n}")
print("=" * 80)

# Load input data
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

data = np.load(input_file)
steps = data['steps']
halo_ids = data['halo_ids']
snap_final = int(data['snap'])
nsnap_final = snapshot_map.get(snap_final, snap_final)

print(f"\nLoaded data for {len(steps)} realization(s)")
print(f"Final snapshot: {snap_final:03d} (nsnap={nsnap_final})")

# Storage for results
all_results = {}

# Loop over realizations
for i, step in enumerate(steps):
    halo_id = halo_ids[i]

    if halo_id == -1:
        print(f"\nRealization {step}: No halo found, skipping")
        continue

    print(f"\n{'=' * 80}")
    print(f"PROCESSING REALIZATION {step}")
    print(f"Target halo ID: {halo_id}")
    print(f"{'=' * 80}")

    # Step 1: Load final snapshot and get particles in target halo
    print(f"\n1. Loading final snapshot (group {snap_final:03d})...")
    snap_final_path = (
        f"{base_path}/step_{step}/output/snapdir_{snap_final:03d}/"
        f"snapshot_{snap_final:03d}.hdf5")

    try:
        snap_final_snap = csiborgtools.read.CSiBORG3Snapshot(
            0, nsnap_final, paths, fpath_override=snap_final_path)
    except Exception as e:
        print(f"   Error loading final snapshot: {e}")
        continue

    # Get all particle IDs and their halo assignments
    print("   Reading particle IDs and halo assignments...")
    pids_final = snap_final_snap.particle_ids()
    hids_final = snap_final_snap.particle_halo_ids(False)

    # Find particles belonging to target halo
    target_halo_mask = hids_final == halo_id
    target_pids = pids_final[target_halo_mask]
    n_target_particles = len(target_pids)

    print(f"   Particles in target halo: {n_target_particles:,}")

    if n_target_particles == 0:
        print("   Warning: No particles found in target halo")
        continue

    # Clean up
    del pids_final, hids_final

    # Step 2: Load initial snapshot
    print(f"\n2. Loading initial snapshot (group {snap_initial:03d})...")
    snap_initial_path = (
        f"{base_path}/step_{step}/output/snapdir_{snap_initial:03d}/"
        f"snapshot_{snap_initial:03d}.hdf5")

    try:
        snap_initial_snap = csiborgtools.read.CSiBORG3Snapshot(
            0, nsnap_initial, paths, fpath_override=snap_initial_path)
    except Exception as e:
        print(f"   Error loading initial snapshot: {e}")
        continue

    print("   Reading particle IDs and halo assignments...")
    pids_initial = snap_initial_snap.particle_ids()
    hids_initial = snap_initial_snap.particle_halo_ids(False)

    print(f"   Total particles at initial: {len(pids_initial):,}")

    # Step 3: Create mapping from particle ID to halo ID at initial snapshot
    print("\n3. Creating particle-to-halo mapping at initial snapshot...")
    pid_to_halo_initial = {
        pid: hid for pid, hid in zip(pids_initial, hids_initial)}

    # Step 4: Match target particles to their initial halos
    print("\n4. Tracing particles back to initial snapshot...")
    initial_halo_counts = {}
    field_particles = 0

    for pid in target_pids:
        if pid in pid_to_halo_initial:
            initial_hid = pid_to_halo_initial[pid]
            if initial_hid == -1:
                field_particles += 1
            else:
                initial_halo_counts[initial_hid] = (
                    initial_halo_counts.get(initial_hid, 0) + 1)
        else:
            # Particle not found in initial snapshot (shouldn't happen)
            field_particles += 1

    n_halos_matched = len(initial_halo_counts)
    particles_in_halos = n_target_particles - field_particles

    print(f"   Particles traced to halos: {particles_in_halos:,} "
          f"({100 * particles_in_halos / n_target_particles:.1f}%)")
    print(f"   Field particles at initial: {field_particles:,} "
          f"({100 * field_particles / n_target_particles:.1f}%)")
    print(f"   Number of contributing halos: {n_halos_matched:,}")

    # Step 5: Load initial catalogue for halo properties
    print("\n5. Loading initial catalogue for halo properties...")
    cat_initial_path = (
        f"{base_path}/step_{step}/output/groups_{snap_initial:03d}/"
        f"fof_subhalo_tab_{snap_initial:03d}.hdf5")

    try:
        cat_initial = csiborgtools.read.CSiBORG3Catalogue(
            nsim=0, nsnap=nsnap_initial, paths=paths,
            fpath_override=cat_initial_path, verbose=False)
    except Exception as e:
        print(f"   Error loading initial catalogue: {e}")
        continue

    # Get halo properties
    group_mass = cat_initial.Group_M_Crit200
    group_coords = cat_initial.coordinates

    # Compute distances from box center
    group_distances = np.sqrt(np.sum((group_coords - box_center)**2, axis=1))

    # Convert to galactic coordinates
    radec = csiborgtools.cartesian_to_radec(group_coords, origin=box_center)
    ra, dec = radec[:, 1], radec[:, 2]
    gal_l, gal_b = csiborgtools.radec_to_galactic(ra, dec)

    # Step 6: Prepare results
    print("\n6. Preparing results...")
    halo_results = []

    for halo_id_init, count in initial_halo_counts.items():
        halo_results.append({
            'halo_id': halo_id_init,
            'count': count,
            'fraction': count / n_target_particles,
            'mass': group_mass[halo_id_init],
            'distance': group_distances[halo_id_init],
            'gal_l': gal_l[halo_id_init],
            'gal_b': gal_b[halo_id_init]
        })

    # Sort by particle count
    halo_results.sort(key=lambda x: x['count'], reverse=True)

    # Step 7: Display contributing halos
    n_display = min(top_n, len(halo_results))
    print(f"\n7. Showing top {n_display} of {len(halo_results)} "
          f"contributing halos:")
    print("   " + "=" * 110)
    print(f"   {'Halo ID':>8} {'N_particles':>12} {'Fraction':>10} "
          f"{'Mass [Msun/h]':>15} {'Dist [cMpc/h]':>14} "
          f"{'l [deg]':>10} {'b [deg]':>10}")
    print("   " + "-" * 110)

    for result in halo_results[:n_display]:
        print(f"   {result['halo_id']:8d} {result['count']:12d} "
              f"{result['fraction']:10.4f} {result['mass']:15.2e} "
              f"{result['distance']:14.2f} "
              f"{result['gal_l']:10.2f} {result['gal_b']:10.2f}")

    print("   " + "=" * 110)

    # Store results
    all_results[step] = {
        'target_halo_id': int(halo_id),
        'n_target_particles': n_target_particles,
        'field_particles': field_particles,
        'n_contributing_halos': n_halos_matched,
        'halo_results': halo_results
    }

# Save results
print(f"\n{'=' * 80}")
print("SAVING RESULTS")
print(f"{'=' * 80}")

# Create output filename based on input filename
input_basename = os.path.splitext(os.path.basename(input_file))[0]
output_basename = input_basename.replace('halo_finder', 'halo_origins')

# Save to text file
txt_file = os.path.join(results_dir, f"{output_basename}.txt")

with open(txt_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("HALO ORIGIN TRACER - RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Input file: {os.path.basename(input_file)}\n")
    f.write(f"Initial snapshot: {snap_initial:03d} (nsnap={nsnap_initial})\n")
    f.write(f"Final snapshot: {snap_final:03d} (nsnap={nsnap_final})\n")
    f.write(f"Realizations processed: {len(all_results)}\n")
    f.write("=" * 80 + "\n\n")

    for step, results in all_results.items():
        f.write(f"\nRealization {step}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Target halo ID at final snapshot: "
                f"{results['target_halo_id']}\n")
        f.write(f"Total particles in target halo: "
                f"{results['n_target_particles']:,}\n")
        f.write(f"Field particles at initial: "
                f"{results['field_particles']:,}\n")
        f.write(f"Contributing halos at initial: "
                f"{results['n_contributing_halos']}\n")
        f.write("\n")

        n_display = min(top_n, len(results['halo_results']))
        f.write(f"Showing top {n_display} of {len(results['halo_results'])} "
                f"contributing halos:\n")
        f.write(f"{'Halo ID':>8} {'N_particles':>12} {'Fraction':>10} "
                f"{'Mass [Msun/h]':>15} {'Dist [cMpc/h]':>14} "
                f"{'l [deg]':>10} {'b [deg]':>10}\n")
        f.write("-" * 110 + "\n")

        for result in results['halo_results'][:n_display]:
            f.write(f"{result['halo_id']:8d} {result['count']:12d} "
                    f"{result['fraction']:10.4f} {result['mass']:15.2e} "
                    f"{result['distance']:14.2f} "
                    f"{result['gal_l']:10.2f} {result['gal_b']:10.2f}\n")

        f.write(f"\nNote: All {len(results['halo_results'])} halos "
                f"saved to numpy file\n")
        f.write("\n")

print(f"\nResults saved to: {txt_file}")

# Save numpy data
npz_file = os.path.join(cache_dir, f"{output_basename}.npz")

save_data = {
    'steps': np.array(list(all_results.keys())),
    'snap_initial': snap_initial,
    'snap_final': snap_final,
}

# Add per-realization data
for step, results in all_results.items():
    save_data[f'step{step}_target_halo_id'] = results['target_halo_id']
    save_data[f'step{step}_n_target_particles'] = (
        results['n_target_particles'])
    save_data[f'step{step}_field_particles'] = results['field_particles']
    save_data[f'step{step}_n_contributing_halos'] = (
        results['n_contributing_halos'])

    # Save halo results as structured arrays
    n_halos = len(results['halo_results'])
    halo_data = np.zeros(n_halos, dtype=[
        ('halo_id', 'i4'),
        ('count', 'i4'),
        ('fraction', 'f8'),
        ('mass', 'f8'),
        ('distance', 'f8'),
        ('gal_l', 'f8'),
        ('gal_b', 'f8')
    ])

    for j, halo in enumerate(results['halo_results']):
        halo_data[j] = (
            halo['halo_id'], halo['count'], halo['fraction'],
            halo['mass'], halo['distance'], halo['gal_l'], halo['gal_b']
        )

    save_data[f'step{step}_halos'] = halo_data

np.savez(npz_file, **save_data)
print(f"Numpy data saved to: {npz_file}")
print("\nDone!")
