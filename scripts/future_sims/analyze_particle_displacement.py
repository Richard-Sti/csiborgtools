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
Analyze displacement of particles from observer vicinity between initial
and final snapshots.
"""
import argparse
import hashlib
import os
import pickle

import csiborgtools
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Analyze particle displacement between snapshots')
parser.add_argument('--steps', type=int, nargs='+',
                    default=[i for i in range(50)],
                    help='Simulation step numbers to process')
parser.add_argument('--snap-initial', type=int, default=1,
                    help='Initial snapshot group number (default: 1)')
parser.add_argument('--snap-final', type=int, default=4,
                    help='Final snapshot group number (default: 4)')
parser.add_argument('--clear-cache', action='store_true',
                    help='Clear the cache and recompute results')
args = parser.parse_args()

# Configuration
steps = args.steps
snap_initial = args.snap_initial
snap_final = args.snap_final
# Enclosed radii at initial snapshot (cMpc/h)
radii = [5.0, 7.5, 15.0]
boxsize = 681.0  # cMpc/h
box_center = np.array([boxsize / 2, boxsize / 2, boxsize / 2])
# Bootstrap parameters for uncertainty estimation
n_bootstrap = 1000

# Snapshot index mapping for CSiBORG3 reader
snapshot_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4}
nsnap_initial = snapshot_map.get(snap_initial, snap_initial)
nsnap_final = snapshot_map.get(snap_final, snap_final)

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
print("PARTICLE DISPLACEMENT ANALYSIS")
print("=" * 80)
print(f"Initial snapshot: {snap_initial:03d} (nsnap={nsnap_initial})")
print(f"Final snapshot: {snap_final:03d} (nsnap={nsnap_final})")
print(f"Enclosed radii: {radii} cMpc/h")
print(f"Bootstrap samples: {n_bootstrap}")
print(f"Processing {len(steps)} realization(s): {steps}")
print("=" * 80)


# Cache file naming
def get_cache_filename(steps, snap_initial, snap_final):
    """Generate cache filename based on input parameters."""
    steps_str = '_'.join(map(str, sorted(steps)))
    # Use a hash for very long step lists
    if len(steps_str) > 100:
        steps_hash = hashlib.md5(steps_str.encode()).hexdigest()[:8]
        steps_str = f"hash_{steps_hash}"
    return os.path.join(
        cache_dir,
        f"results_cache_snap{snap_initial}_to_{snap_final}_steps_{steps_str}.pkl"  # noqa
    )


def compute_displacement_with_periodic(coords_initial, coords_final, boxsize):
    """
    Compute displacement accounting for periodic boundary conditions.

    Parameters
    ----------
    coords_initial : array (N, 3)
        Initial coordinates
    coords_final : array (N, 3)
        Final coordinates
    boxsize : float
        Box size for periodic boundaries

    Returns
    -------
    displacement : array (N, 3)
        Displacement vectors accounting for periodic boundaries
    """
    displacement = coords_final - coords_initial

    # Handle periodic boundaries: if displacement > boxsize/2, wrap around
    displacement[displacement > boxsize / 2] -= boxsize
    displacement[displacement < -boxsize / 2] += boxsize

    return displacement


def bootstrap_direction_uncertainty(displacements, n_bootstrap=1000):
    """
    Estimate uncertainty on mean displacement direction using bootstrap.

    Parameters
    ----------
    displacements : array (N, 3)
        Displacement vectors
    n_bootstrap : int
        Number of bootstrap samples

    Returns
    -------
    l_std : float
        Standard deviation of galactic longitude (degrees)
    b_std : float
        Standard deviation of galactic latitude (degrees)
    """
    n_particles = len(displacements)
    l_samples = []
    b_samples = []

    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_particles, size=n_particles,
                                   replace=True)
        sample_displacements = displacements[indices]

        # Compute mean displacement
        mean_disp = np.mean(sample_displacements, axis=0)
        mean_disp_mag = np.sqrt(np.sum(mean_disp**2))

        if mean_disp_mag > 0:
            mean_dir = mean_disp / mean_disp_mag
            # Convert to galactic coordinates
            mean_dir_point = box_center + mean_dir * 100
            radec = csiborgtools.cartesian_to_radec(
                mean_dir_point.reshape(1, 3), origin=box_center)
            ra, dec = radec[0, 1], radec[0, 2]
            gal_l, gal_b = csiborgtools.radec_to_galactic(
                np.array([ra]), np.array([dec]))
            l_samples.append(gal_l[0])
            b_samples.append(gal_b[0])

    return np.std(l_samples), np.std(b_samples)


# Check for cached results
cache_file = get_cache_filename(steps, snap_initial, snap_final)

# Clear cache if requested
if args.clear_cache and os.path.exists(cache_file):
    print(f"\nClearing cache: {cache_file}")
    os.remove(cache_file)
    print("Cache cleared.")

# Try to load from cache
all_results = []
particle_directions_by_real = None
use_cached = False

if os.path.exists(cache_file) and not args.clear_cache:
    print(f"\nLoading cached results from: {cache_file}")
    try:
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)

        # Validate cache matches current parameters
        if (cached_data['snap_initial'] == snap_initial
                and cached_data['snap_final'] == snap_final
                and list(cached_data['radii']) == list(radii)
                and cached_data['n_bootstrap'] == n_bootstrap
                and set(cached_data['steps']) == set(steps)):
            all_results = cached_data['all_results']
            particle_directions_by_real = cached_data['particle_directions_by_real']  # noqa
            use_cached = True
            print("✓ Cache loaded successfully!")
            print(f"  Loaded {len(all_results)} realization(s)")
        else:
            print("✗ Cache parameters don't match, recomputing...")
    except Exception as e:
        print(f"✗ Error loading cache: {e}")
        print("  Recomputing...")

# Storage for results
if not use_cached:
    all_results = []
    # Storage for particle directions (for histograms, by realization)
    particle_directions_by_real = None

# Loop over realizations
if not use_cached:
    for step in steps:
        print(f"\n{'=' * 80}")
        print(f"PROCESSING REALIZATION {step}")
        print(f"{'=' * 80}")

        # Step 1: Load initial snapshot and select particles
        print(f"\n1. Loading initial snapshot (group {snap_initial:03d})...")
        snap_initial_path = (
            f"{base_path}/step_{step}/output/snapdir_{snap_initial:03d}/"
            f"snapshot_{snap_initial:03d}.0.hdf5")

        try:
            snap_initial_snap = csiborgtools.read.CSiBORG3Snapshot(
                0, nsnap_initial, paths, fpath_override=snap_initial_path)
        except Exception as e:
            print(f"   Error loading initial snapshot: {e}")
            continue

        print("   Reading particle coordinates and IDs...")
        coords_initial = snap_initial_snap.coordinates()
        pids_initial = snap_initial_snap.particle_ids()

        print(f"   Total particles at initial: {len(pids_initial):,}")

        # Compute distances from box center
        print("   Computing distances from box center...")
        distances = np.sqrt(np.sum((coords_initial - box_center)**2, axis=1))

        # Step 2: Load final snapshot
        print(f"\n2. Loading final snapshot (group {snap_final:03d})...")
        snap_final_path = (
            f"{base_path}/step_{step}/output/snapdir_{snap_final:03d}/"
            f"snapshot_{snap_final:03d}.0.hdf5")

        try:
            snap_final_snap = csiborgtools.read.CSiBORG3Snapshot(
                0, nsnap_final, paths, fpath_override=snap_final_path)
        except Exception as e:
            print(f"   Error loading final snapshot: {e}")
            continue

        print("   Reading particle coordinates and IDs...")
        coords_final = snap_final_snap.coordinates()
        pids_final = snap_final_snap.particle_ids()

        print(f"   Total particles at final: {len(pids_final):,}")

        # Step 3: Create mapping from particle ID to index/coordinates
        print("\n3. Creating particle mapping...")
        # Create mapping from particle ID to index at final snapshot
        pid_to_index_final = {pid: idx for idx, pid in enumerate(pids_final)}

        # Store results for this realization
        step_results = {'step': step, 'radii_results': {}}

        # Store particle directions for histograms (per realization)
        if step == steps[0]:
            particle_directions_by_real = {
                radius: {'l': {}, 'b': {}, 'mag': {}} for radius in radii}

        # Step 4: Process each radius bin
        print("\n4. Processing radius bins...")
        for radius in radii:
            print(f"\n   Radius {radius:.1f} cMpc/h:")

            # Select particles within this radius
            selection_mask = distances < radius
            selected_pids = pids_initial[selection_mask]
            selected_coords_initial = coords_initial[selection_mask]
            n_selected = len(selected_pids)

            print(f"      Selected particles: {n_selected:,}")

            # Find matched particles at final snapshot
            matched_mask = np.array([pid in pid_to_index_final
                                     for pid in selected_pids])
            n_matched = np.sum(matched_mask)

            print(f"      Matched particles: {n_matched:,} "
                  f"({100 * n_matched / n_selected:.1f}%)")

            if n_matched == 0:
                print("      Warning: No particles matched, skipping...")
                continue

            # Get matched particles
            matched_pids = selected_pids[matched_mask]
            matched_coords_initial = selected_coords_initial[matched_mask]

            # Get final coordinates
            matched_indices_final = np.array([pid_to_index_final[pid]
                                              for pid in matched_pids])
            matched_coords_final = coords_final[matched_indices_final]

            # Compute displacement vectors
            displacements = compute_displacement_with_periodic(
                matched_coords_initial, matched_coords_final, boxsize)

            # Compute direction for each particle's displacement
            disp_magnitudes = np.sqrt(np.sum(displacements**2, axis=1))
            nonzero_mask = disp_magnitudes > 0

            # Store magnitudes for all particles (including zero)
            particle_directions_by_real[radius]['mag'][step] = disp_magnitudes

            # Convert each particle's displacement to direction
            if np.sum(nonzero_mask) > 0:
                # Normalize displacements to unit vectors
                disp_directions = (displacements[nonzero_mask]
                                   / disp_magnitudes[nonzero_mask, np.newaxis])

                # Convert to galactic coordinates for each particle
                disp_points = box_center[np.newaxis, :] + disp_directions * 100
                radec_particles = csiborgtools.cartesian_to_radec(
                    disp_points, origin=box_center)
                ra_particles = radec_particles[:, 1]
                dec_particles = radec_particles[:, 2]
                l_particles, b_particles = csiborgtools.radec_to_galactic(
                    ra_particles, dec_particles)

                # Store for histogram (by realization)
                particle_directions_by_real[radius]['l'][step] = l_particles
                particle_directions_by_real[radius]['b'][step] = b_particles

            # Mean displacement vector
            mean_displacement = np.mean(displacements, axis=0)
            mean_displacement_magnitude = np.sqrt(np.sum(mean_displacement**2))

            print(f"      Displacement magnitude: "
                  f"{mean_displacement_magnitude:.2f} cMpc/h")

            # Convert to direction
            if mean_displacement_magnitude > 0:
                mean_direction = (mean_displacement
                                  / mean_displacement_magnitude)

                # Convert to galactic coordinates
                mean_direction_point = box_center + mean_direction * 100
                radec = csiborgtools.cartesian_to_radec(
                    mean_direction_point.reshape(1, 3), origin=box_center)
                ra, dec = radec[0, 1], radec[0, 2]
                gal_l, gal_b = csiborgtools.radec_to_galactic(
                    np.array([ra]), np.array([dec]))
                gal_l, gal_b = gal_l[0], gal_b[0]

                # Estimate uncertainty using bootstrap
                print("      Computing uncertainties...")
                l_std, b_std = bootstrap_direction_uncertainty(
                    displacements, n_bootstrap)

                print(f"      Direction: l={gal_l:.2f}±{l_std:.2f}°, "
                      f"b={gal_b:.2f}±{b_std:.2f}°")
            else:
                mean_direction = np.array([0, 0, 0])
                gal_l, gal_b = np.nan, np.nan
                l_std, b_std = np.nan, np.nan
                print("      Warning: Mean displacement is zero")

            # Store results for this radius
            step_results['radii_results'][radius] = {
                'n_selected': n_selected,
                'n_matched': n_matched,
                'mean_displacement_vector': mean_displacement,
                'mean_displacement_magnitude': mean_displacement_magnitude,
                'mean_direction_cartesian': mean_direction,
                'mean_direction_l': gal_l,
                'mean_direction_b': gal_b,
                'direction_l_std': l_std,
                'direction_b_std': b_std,
            }

        # Clean up
        del coords_initial, pids_initial, coords_final, pids_final

        # Store results for this step
        all_results.append(step_results)

    # Save cache after computation
    print(f"\nSaving cache to: {cache_file}")
    cache_data = {
        'snap_initial': snap_initial,
        'snap_final': snap_final,
        'radii': radii,
        'n_bootstrap': n_bootstrap,
        'steps': steps,
        'all_results': all_results,
        'particle_directions_by_real': particle_directions_by_real,
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print("✓ Cache saved successfully!")

# Save results
print(f"\n{'=' * 80}")
print("SAVING RESULTS")
print(f"{'=' * 80}")

# Save to text file
txt_file = os.path.join(
    results_dir,
    f"displacement_snap{snap_initial}_to_{snap_final}.txt"
)

with open(txt_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("PARTICLE DISPLACEMENT ANALYSIS - RESULTS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Initial snapshot: {snap_initial:03d} "
            f"(nsnap={nsnap_initial})\n")
    f.write(f"Final snapshot: {snap_final:03d} (nsnap={nsnap_final})\n")
    f.write(f"Radii analyzed: {radii} cMpc/h\n")
    f.write(f"Bootstrap samples: {n_bootstrap}\n")
    f.write(f"Realizations processed: {len(all_results)}\n")
    f.write("=" * 80 + "\n\n")

    # Results by radius
    for radius in radii:
        f.write(f"\n{'=' * 80}\n")
        f.write(f"RADIUS: {radius:.1f} cMpc/h\n")
        f.write(f"{'=' * 80}\n\n")

        f.write(f"{'Step':>6} {'N_sel':>8} {'N_match':>8} "
                f"{'|<d>| [cMpc/h]':>15} "
                f"{'l [deg]':>12} {'b [deg]':>12}\n")
        f.write("-" * 80 + "\n")

        for step_result in all_results:
            step = step_result['step']
            if radius in step_result['radii_results']:
                r = step_result['radii_results'][radius]
                f.write(f"{step:6d} {r['n_selected']:8d} "
                        f"{r['n_matched']:8d} "
                        f"{r['mean_displacement_magnitude']:15.2f} "
                        f"{r['mean_direction_l']:7.2f}±"
                        f"{r['direction_l_std']:4.2f} "
                        f"{r['mean_direction_b']:7.2f}±"
                        f"{r['direction_b_std']:4.2f}\n")

        f.write("=" * 80 + "\n")

    # Detailed results
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("DETAILED RESULTS\n")
    f.write("=" * 80 + "\n\n")

    for step_result in all_results:
        step = step_result['step']
        f.write(f"\nRealization {step}:\n")
        f.write("-" * 80 + "\n")

        for radius in radii:
            if radius in step_result['radii_results']:
                r = step_result['radii_results'][radius]
                f.write(f"\n  Radius {radius:.1f} cMpc/h:\n")
                f.write(f"    Selected particles: {r['n_selected']:,}\n")
                f.write(f"    Matched particles: {r['n_matched']:,}\n")
                f.write(f"    Displacement magnitude: "
                        f"{r['mean_displacement_magnitude']:.2f} cMpc/h\n")
                f.write(f"    Displacement vector: "
                        f"[{r['mean_displacement_vector'][0]:.2f}, "
                        f"{r['mean_displacement_vector'][1]:.2f}, "
                        f"{r['mean_displacement_vector'][2]:.2f}] cMpc/h\n")
                f.write(f"    Displacement direction (Galactic): "
                        f"l={r['mean_direction_l']:.2f}±"
                        f"{r['direction_l_std']:.2f}°, "
                        f"b={r['mean_direction_b']:.2f}±"
                        f"{r['direction_b_std']:.2f}°\n")

print(f"\nResults saved to: {txt_file}")

# Save numpy data
npz_file = os.path.join(
    cache_dir,
    f"displacement_snap{snap_initial}_to_{snap_final}.npz"
)

save_data = {
    'steps': np.array([r['step'] for r in all_results]),
    'snap_initial': snap_initial,
    'snap_final': snap_final,
    'radii': np.array(radii),
    'n_bootstrap': n_bootstrap,
}

# Add data for each radius
for radius in radii:
    r_str = f"r{int(radius)}"

    # Collect data for this radius across all realizations
    n_sel = []
    n_match = []
    disp_mag = []
    disp_vec = []
    dir_cart = []
    dir_l = []
    dir_b = []
    dir_l_std = []
    dir_b_std = []

    for step_result in all_results:
        if radius in step_result['radii_results']:
            r = step_result['radii_results'][radius]
            n_sel.append(r['n_selected'])
            n_match.append(r['n_matched'])
            disp_mag.append(r['mean_displacement_magnitude'])
            disp_vec.append(r['mean_displacement_vector'])
            dir_cart.append(r['mean_direction_cartesian'])
            dir_l.append(r['mean_direction_l'])
            dir_b.append(r['mean_direction_b'])
            dir_l_std.append(r['direction_l_std'])
            dir_b_std.append(r['direction_b_std'])

    save_data[f'{r_str}_n_selected'] = np.array(n_sel)
    save_data[f'{r_str}_n_matched'] = np.array(n_match)
    save_data[f'{r_str}_displacement_magnitude'] = np.array(disp_mag)
    save_data[f'{r_str}_displacement_vector'] = np.array(disp_vec)
    save_data[f'{r_str}_direction_cartesian'] = np.array(dir_cart)
    save_data[f'{r_str}_direction_l'] = np.array(dir_l)
    save_data[f'{r_str}_direction_b'] = np.array(dir_b)
    save_data[f'{r_str}_direction_l_std'] = np.array(dir_l_std)
    save_data[f'{r_str}_direction_b_std'] = np.array(dir_b_std)

np.savez(npz_file, **save_data)
print(f"Numpy data saved to: {npz_file}")

# Create plots
print("\nCreating plots...")

if particle_directions_by_real is not None:
    with plt.style.context('science'):
        # Create histogram plot: one row per radius, three columns (l, b, mag)
        fig, axes = plt.subplots(len(radii), 3, figsize=(13, 2.5 * len(radii)))

        # Ensure axes is 2D even for single radius
        if len(radii) == 1:
            axes = axes.reshape(1, -1)

        # Plot histograms for each radius
        for i, radius in enumerate(radii):
            ax_mag = axes[i, 0]
            ax_l = axes[i, 1]
            ax_b = axes[i, 2]

            # Collect all data for this radius
            all_l = []
            all_b = []
            all_mag = []

            for step in steps:
                if step in particle_directions_by_real[radius]['mag']:
                    all_mag.extend(
                        particle_directions_by_real[radius]['mag'][step])
                if step in particle_directions_by_real[radius]['l']:
                    all_l.extend(
                        particle_directions_by_real[radius]['l'][step])
                    all_b.extend(
                        particle_directions_by_real[radius]['b'][step])

            # Compute bins using 'auto' method
            l_bins = np.histogram_bin_edges(
                all_l, bins='auto', range=(0, 360))
            b_bins = np.histogram_bin_edges(
                all_b, bins='auto', range=(-90, 90))
            mag_bins = np.histogram_bin_edges(all_mag, bins='auto')

            # Total particle count
            total_particles = len(all_mag)

            hist_kwargs = {"histtype": 'step'}

            # Plot single histogram combining all realizations
            if all_mag:
                ax_mag.hist(all_mag, bins=mag_bins, **hist_kwargs)

            if all_l:
                ax_l.hist(all_l, bins=l_bins, **hist_kwargs)

            if all_b:
                ax_b.hist(all_b, bins=b_bins, **hist_kwargs)

            # Add expected random distribution for b (proportional to cos(b))
            b_centers = 0.5 * (b_bins[:-1] + b_bins[1:])
            # For uniform distribution on sphere: dN/db ∝ cos(b)
            expected_random = np.cos(np.radians(b_centers))
            # Normalize to match total counts
            expected_random *= total_particles / np.sum(expected_random)
            ax_b.plot(b_centers, expected_random, 'k--', linewidth=2,
                      label='Random', alpha=0.8)

            # Formatting
            ax_l.set_ylabel('Bins per count')
            ax_l.set_xlim(0, 360)

            ax_b.set_ylabel('Bins per count')
            ax_b.legend(loc='best')
            ax_b.set_xlim(-90, 90)

            ax_mag.set_ylabel('Bins per count')

            # Only set x-label on bottom row
            if i == len(radii) - 1:
                ax_mag.set_xlabel(r"$|\mathbf{d}|~[h^{-1}\mathrm{cMpc}]$")
                ax_l.set_xlabel(r"$\ell~[\mathrm{deg}]$")
                ax_b.set_xlabel(r"$b~[\mathrm{deg}]$")

        plt.tight_layout()

        # Adjust vertical spacing between rows to make room for titles
        plt.subplots_adjust(hspace=0.35)

        # Add titles above each row after tight_layout
        for i, radius in enumerate(radii):
            # Count particles per realization for this radius
            particles_per_real = [
                len(particle_directions_by_real[radius]['mag'][step])
                for step in steps
                if step in particle_directions_by_real[radius]['mag']]

            # Calculate mean and std
            mean_particles = np.mean(particles_per_real)
            std_particles = np.std(particles_per_real)

            title_y = axes[i, 0].get_position().y1 + 0.02
            title_x = (axes[i, 0].get_position().x0 +
                       axes[i, 2].get_position().x1) / 2
            fig.text(title_x, title_y,
                     f'$r = {radius:.1f}~h^{{-1}}\mathrm{{cMpc}}$, '
                     f'$N = {mean_particles:.0f} \pm {std_particles:.0f}$',
                     ha='center', fontweight='bold')

        # Save figure
        plot_file = os.path.join(
            results_dir,
            f"displacement_direction_snap{snap_initial}_to_{snap_final}.png"
        )
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
        plt.close()
else:
    print("Warning: No particle direction data available for plotting")

print("\nDone!")
