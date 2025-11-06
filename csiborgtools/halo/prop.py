# Copyright (C) 2023 Richard Stiskalek
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

"""Halo properties and utilities.

This module provides a Halo dataclass to represent dark matter halos
and functions to compute their properties, such as the center of mass
using the shrinking sphere method.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..utils import (periodic_distance, euclidean_distance,
                     center_of_mass_euclidean)


def center_of_mass_periodic(points, mass, boxsize):
    """
    Compute the center of mass for a set of points with periodic boundary
    conditions.
    """
    sin_inv_points = np.sin(2 * np.pi * points / boxsize)
    cos_inv_points = np.cos(2 * np.pi * points / boxsize)

    cm_real = np.sum(mass[:, np.newaxis] * cos_inv_points, axis=0)
    cm_imag = np.sum(mass[:, np.newaxis] * sin_inv_points, axis=0)

    cm = np.arctan2(cm_imag, cm_real) * boxsize / (2 * np.pi)
    cm[cm < 0] += boxsize

    return cm


@dataclass
class Halo:
    """Represents a dark matter halo.

    Attributes:
        pos (np.ndarray): Particle positions.
        mass (np.ndarray): Particle masses.
        vel (Optional[np.ndarray]): Particle velocities (optional).
    """
    pos: np.ndarray
    mass: np.ndarray
    vel: Optional[np.ndarray] = None

    def compute_center(self, boxsize, npart_min=50, shrink_factor=0.95,
                       periodic=True):
        """
        Compute the center of the halo using the shrinking sphere method.

        This method iteratively refines the center of mass by considering
        only particles within a shrinking sphere around the current center.
        """
        if periodic:
            distance_func = periodic_distance
            cm_func = center_of_mass_periodic
            initial_center = cm_func(self.pos, self.mass, boxsize)
            dist = distance_func(self.pos, initial_center, boxsize)
        else:
            distance_func = euclidean_distance
            cm_func = center_of_mass_euclidean
            initial_center = cm_func(self.pos, self.mass)
            dist = distance_func(self.pos, initial_center)

        center = initial_center
        rad = np.percentile(dist, 90)

        while True:
            if periodic:
                mask = distance_func(self.pos, center, boxsize) <= rad
            else:
                mask = distance_func(self.pos, center) <= rad

            if np.sum(mask) < npart_min:
                break

            if periodic:
                center = cm_func(self.pos[mask], self.mass[mask], boxsize)
            else:
                center = cm_func(self.pos[mask], self.mass[mask])
            rad *= shrink_factor

        return center

    def spherical_overdensity_mass(self, center, rho_target, boxsize=None,
                                   periodic=False):
        """
        Compute mass and radius at spherical overdensity threshold.

        Parameters
        ----------
        center : np.ndarray
            Center position (3D).
        rho_target : float
            Target overdensity [Msun / Mpc^3].
        boxsize : float, optional
            Box size for periodic boundary conditions.
        periodic : bool, optional
            Whether to use periodic boundary conditions.

        Returns
        -------
        mass : float
            Mass within the computed radius.
        radius : float
            Radius enclosing the target overdensity.
        """
        if periodic:
            dist = periodic_distance(self.pos, center, boxsize)
        else:
            dist = euclidean_distance(self.pos, center)

        sort_idx = np.argsort(dist)
        dist_sorted = dist[sort_idx]
        mass_sorted = self.mass[sort_idx]

        cumulative_mass = np.cumsum(mass_sorted)
        rho = cumulative_mass / (4/3 * np.pi * dist_sorted**3)

        rho_ratio = rho / rho_target

        if rho_ratio[1] > 1:
            idx = np.where(rho_ratio < 1)[0]
        else:
            idx = np.where(rho_ratio > 1)[0]

        if len(idx) == 0:
            return np.nan, np.nan

        j = idx[0]
        i = j - 1

        if not ((rho_ratio[i] > 1 and rho_ratio[j] < 1)
                or (rho_ratio[i] < 1 and rho_ratio[j] > 1)):
            return np.nan, np.nan

        # Linear interpolation
        r = (dist_sorted[j] - dist_sorted[i]) * (1 - rho_ratio[i])
        r /= (rho_ratio[j] - rho_ratio[i])
        r += dist_sorted[i]

        mass = 4/3 * np.pi * rho_target * r**3

        if mass > cumulative_mass[-1]:
            return np.nan, np.nan

        return mass, r

    def compute_r200c(self, center, h, boxsize=None, periodic=False):
        """
        Compute M200c and R200c for the halo.

        M200c is the mass within R200c, where R200c is the radius within
        which the mean density is 200 times the critical density of the
        universe at z=0.

        Parameters
        ----------
        center : np.ndarray
            Center position (3D).
        h : float
            Hubble parameter in units of 100 km/s/Mpc.
        boxsize : float, optional
            Box size for periodic boundary conditions.
        periodic : bool, optional
            Whether to use periodic boundary conditions.

        Returns
        -------
        m200c : float
            Mass within R200c [Msun/h].
        r200c : float
            R200c radius [Mpc/h].
        """
        rho_crit0 = 2.77536627e11 * h**2  # Msun / Mpc^3
        rho_200c = 200 * rho_crit0

        return self.spherical_overdensity_mass(center, rho_200c, boxsize,
                                               periodic)
