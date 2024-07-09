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
NOTE: The script is far from finished or written well.
"""
from os.path import join
import numpy as np

import csiborgtools
from h5py import File



from sklearn.neighbors import NearestNeighbors
from numba import jit
from scipy.stats import binned_statistic


###

def find_indxs(rtree, r, radius):
    """
    Find the indices of points that are within a given radius of a given
    point `r`.
    """
    if isinstance(r, (int, float)):
        r = np.array(r)

    return rtree.radius_neighbors(
        r.reshape(-1, 1), radius=radius, return_distance=False, )[0]

@jit(nopython=True)
def dot_product_norm(x1_norm, x2_norm):
    """Dot product of two normalised 1D vectors."""
    return x1_norm[0] * x2_norm[0] + x1_norm[1] * x2_norm[1] + x1_norm[2] * x2_norm[2]  # noqa



@jit(nopython=True)
def get_angdist_vrad_product(i_comb, j_comb, pos_norm, vrad, r, rbin_i, rbin_j):
    # TODO: Add itself?
    len_i = len(i_comb)
    len_j = len(j_comb)

    cos_angdist_values = np.full(len_i * len_j, np.nan)
    vrad_product = np.full(len_i * len_j, np.nan)
    weights = np.full(len_i * len_j, np.nan)

    k = 0
    for i in range(len_i):
        pos_norm_i = pos_norm[i_comb[i]]
        vrad_i = vrad[i_comb[i]]
        w_i = (rbin_i / r[i_comb[i]])**2
        for j in range(len_j):
            cos_angdist_values[k] = dot_product_norm(pos_norm_i, pos_norm[j_comb[j]])

            # Product of the peculiar velocities
            vrad_product[k] = vrad_i * vrad[j_comb[j]]
            # Weight the product
            w = w_i * (rbin_j / r[j_comb[j]])**2
            vrad_product[k] *= w

            weights[k] = w

            k += 1

    return cos_angdist_values, vrad_product, weights


def main(out_summed_product, out_weights, ri, rj, angular_bins, pos, vel, observer, rmax):
    # Centre the positions at the observer
    pos = np.copy(pos) - observer
    r = np.linalg.norm(pos, axis=1)
    mask = r < rmax

    # Select only the haloes within the radial range
    pos = pos[mask]
    vel = vel[mask]
    r = r[mask]

    # Create a KDTree for the radial positions
    rtree = NearestNeighbors().fit(r.reshape(-1, 1))

    # Calculate the radial velocity and the normalised position vector
    pos_norm = pos / r[:, None]
    vrad = np.sum(vel * pos_norm, axis=1)

    # TODO: eventually here loop over the radii
    # for ....
    dr = 2.5
    i_indxs = find_indxs(rtree, ri, radius=dr)
    j_indxs = find_indxs(rtree, rj, radius=dr)

    # Calculate the cosine of the angular distance and the product of the
    # radial velocities for each pair of points.
    cos_angdist, vrad_product, weights = get_angdist_vrad_product(
        i_indxs, j_indxs, pos_norm, vrad, r, ri, rj)

    out_summed_product += binned_statistic(
        cos_angdist, vrad_product, bins=angular_bins, statistic="sum")[0]

    out_weights += binned_statistic(
        cos_angdist, weights, bins=angular_bins, statistic="sum")[0]

    return out_summed_product, out_weights


if __name__ == "__main__":
    fdir = "/mnt/extraspace/rstiskalek/BBF/Quijote_C_ij"

    rmax = 150
    nradial = 20
    nangular = 40
    ncatalogue = 1

    # NOTE check this
    radial_bins = np.linspace(0, rmax, nradial + 1)
    angular_bins = np.linspace(-1, 1, nangular + 1)

    summed_product = np.zeros(nangular)
    weights = np.zeros(nangular)

    fiducial_observers = csiborgtools.read.fiducial_observers(1000, rmax)

    ri = 100
    rj = 120

    # for i in trange(ncatalogue, desc="Catalogues"):
    for i in [30]:
        cat = csiborgtools.read.QuijoteCatalogue(i)

        for j in range(len(fiducial_observers)):
            # Loop over all the fiducial observers in this simulation
            observer = fiducial_observers[j]
            summed_product, weights = main(
                summed_product, weights, ri, rj, angular_bins,
                cat["cartesian_pos"], cat["cartesian_vel"], observer, rmax)


    # TODO: Save
    fname = join(fdir, "test.h5")
    print(f"Saving to `{fname}`.")
    with File(fname, 'w') as f:
        f.create_dataset("summed_product", data=summed_product)
        f.create_dataset("weights", data=weights)
        f.create_dataset("angular_bins", data=angular_bins)
