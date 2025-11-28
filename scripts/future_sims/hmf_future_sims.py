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
Generate halo mass functions for future simulation suite and plot percentile
bands.
"""
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa
import csiborgtools
import pickle
import os
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description='Generate halo mass functions for future simulations')
parser.add_argument('--clear-cache', action='store_true',
                    help='Clear the cache and exit')
parser.add_argument('--no-cache', action='store_true',
                    help='Do not use cache (force recomputation)')
args = parser.parse_args()

# Setup
bin_edges = 10**np.arange(13.4, 15.5, 0.2)
volume = 681**3

# Scale factors for each group
scale_factors = [1, 2, 5, 10, 50, 100]
iskip = len(scale_factors) - 1
groups = [1, 2, 3, 4, 5, 6]
num_sim = 10

# Base path pattern
base_path = ("/mnt/home/rstiskalek/ceph/CSiBORG/"
             "2MPP_MULTIBIN_N256_DES_V2/N256_far_future")

# Setup cache directory
script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "cache")
os.makedirs(cache_dir, exist_ok=True)

# Cache settings
USE_CACHE = not args.no_cache  # Can be disabled via --no-cache flag
cache_file = os.path.join(cache_dir, "hmf_cache.pkl")

# Handle cache clearing
if args.clear_cache:
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Cache cleared: {cache_file}")
    else:
        print(f"No cache file found: {cache_file}")
    # Continue running to regenerate cache
    USE_CACHE = False

# Initialize paths object (we'll override the file path)
paths = csiborgtools.read.Paths(**csiborgtools.paths_rusty)

# Try to load from cache
if USE_CACHE and os.path.exists(cache_file):
    print(f"Loading HMF data from cache: {cache_file}")
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
        data = cache['data']
        x_data = cache['x_data']
        bin_edges = cache['bin_edges']
    print(f"  Loaded data for {len(data)} groups")
else:
    # Storage for results: data[group_idx][step] = (y, ey)
    data = {group: [] for group in groups}

    print("Computing halo mass functions...")
    # Loop over all steps (realisations)
    for step in range(num_sim):
        print(f"Processing step {step}/{num_sim - 1}...")

        # Loop over groups (redshifts)
        for group in groups:
            # Construct file path with zero-padded group numbers
            group_str = f"{group:03d}"
            fname = (f"{base_path}/step_{step}/output/groups_{group_str}/"
                     f"fof_subhalo_tab_{group_str}.hdf5")

            try:
                # Read catalogue and compute HMF
                reader = csiborgtools.read.CSiBORG3Catalogue(
                    0, 0, paths, fpath_override=fname, verbose=False)
                x_data, y, ey = reader.halo_mass_function(
                    bin_edges,
                    mass_key="Group_M_Crit200",
                    volume=volume
                )

                data[group].append((y, ey))
            except Exception as e:
                print(f"  Warning: Failed for step {step}, "
                      f"group {group}: {e}")
                data[group].append((None, None))

    # Save to cache
    print(f"Saving HMF data to cache: {cache_file}")
    cache = {
        'data': data,
        'x_data': x_data,
        'bin_edges': bin_edges
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    print("  Cache saved successfully")

print("Computing percentiles...")
# Compute 16-84 percentile bands for each group
percentile_data = {}
for i, group in enumerate(groups):
    # Collect all valid y values for this group
    y_values = []
    for y, ey in data[group]:
        if y is not None:
            y_values.append(y)

    if len(y_values) > 0:
        y_values = np.array(y_values)
        p16 = np.percentile(y_values, 16, axis=0)
        p50 = np.percentile(y_values, 50, axis=0)
        p84 = np.percentile(y_values, 84, axis=0)
        percentile_data[i] = (p16, p50, p84)
        print(f"  Group {group} (a={scale_factors[i]}): "
              f"{len(y_values)} valid realisations")

# Plotting
print("Creating plot...")
with plt.style.context('science'):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5),
                             gridspec_kw={'height_ratios': [2, 1],
                                          'hspace': 0.05})

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(groups)))

    # Get reference HMF (a=10, which is index 3)
    if iskip not in percentile_data:
        raise ValueError(f"Reference a={scale_factors[iskip]} data not found!")
    p16_ref, p50_ref, p84_ref = percentile_data[iskip]

    # Top panel: HMF
    ax_top = axes[0]
    for i, group in enumerate(groups):
        if i in percentile_data:
            p16, p50, p84 = percentile_data[i]
            ax_top.fill_between(x_data, p16, p84, color=colors[i],
                                alpha=0.5,
                                label=fr"$a = {scale_factors[i]}$")

    ax_top.legend(loc='best')
    ax_top.set_yscale("log")
    ax_top.set_ylabel(r"$\mathrm{HMF} ~ [(h^{-1}\,\mathrm{Mpc})^{-3} \, "
                      r"\mathrm{dex}^{-1}]$", fontsize=12)
    ax_top.tick_params(axis='x', labelbottom=False)

    # Bottom panel: Ratio to a=10
    ax_bottom = axes[1]
    for i, group in enumerate(groups):
        if i in percentile_data and i != iskip:
            p16, p50, p84 = percentile_data[i]

            # Compute ratio
            ratio_p50 = p50 / p50_ref
            ratio_p16 = p16 / p50_ref
            ratio_p84 = p84 / p50_ref

            ax_bottom.plot(x_data, ratio_p50, color=colors[i], lw=2)
            ax_bottom.fill_between(x_data, ratio_p16, ratio_p84,
                                   color=colors[i], alpha=0.3)

    ax_bottom.axhline(1, color='black', ls='--', lw=1, alpha=0.5)
    ax_bottom.set_xlabel(r"$\log M_\mathrm{200c} \, "
                         r"[\mathrm{M}_\odot/h]$", fontsize=12)
    ax_bottom.set_ylabel(fr"Ratio to $a={scale_factors[iskip]}$", fontsize=12)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(cache_dir, "hmf_future_sims.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_file}")

    plt.close()
