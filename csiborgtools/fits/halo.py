# Copyright (C) 2022 Richard Stiskalek, Deaglan Bartlett
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
"""A halo object."""
from abc import ABC

import numpy
from numba import jit
from scipy.optimize import minimize


class BaseStructure(ABC):
    """
    Basic structure object for handling operations on its particles.
    """

    _particles = None
    _box = None

    @property
    def particles(self):
        """
        Particle array.

        Returns
        -------
        particles : structured array
        """
        return self._particles

    @particles.setter
    def particles(self, particles):
        assert particles.ndim == 2 and particles.shape[1] == 7
        self._particles = particles

    @property
    def box(self):
        """
        Box object handling unit conversion.

        Returns
        -------
        box : Object derived from :py:class:`csiborgtools.units.BaseBox`
        """
        return self._box

    @box.setter
    def box(self, box):
        try:
            assert box._name == "box_units"
            self._box = box
        except AttributeError as err:
            raise TypeError from err

    @property
    def pos(self):
        """
        Cartesian particle coordinates in the box coordinate system.

        Returns
        -------
        pos : 2-dimensional array of shape `(n_particles, 3)`.
        """
        return numpy.vstack([self[p] for p in ('x', 'y', 'z')]).T

    @property
    def vel(self):
        """
        Cartesian particle velocity components.

        Returns
        -------
        vel : 2-dimensional array of shape (`n_particles, 3`)
        """
        return numpy.vstack([self[p] for p in ("vx", "vy", "vz")]).T

    def spherical_overdensity_mass(self, delta_mult, kind="crit", tol=1e-8,
                                   maxiter=20):
        r"""
        Calculate spherical overdensity mass and radius via the iterative
        shrinking sphere method.

        Parameters
        ----------
        delta_mult : int or float
            Overdensity multiple.
        kind : str, optional
            Either `crit` or `matter`, for critical or matter overdensity
        tol : float, optional
            Tolerance for the change in the center of mass or radius.
        maxiter : int, optional
            Maximum number of iterations.
        npart_min : int, optional
            Minimum number of enclosed particles for a radius to be
            considered trustworthy.

        Returns
        -------
        mass :  float
            The requested spherical overdensity mass.
        rad : float
            The radius of the sphere enclosing the requested overdensity.
        cm : 1-dimensional array of shape `(3, )`
            The center of mass of the sphere enclosing the requested
            overdensity.
        """
        assert kind in ["crit", "matter"]
        rho = delta_mult * self.box.box_rhoc
        if kind == "matter":
            rho *= self.box.box_Om
        pos = self.pos
        mass = self["M"]

        # First assign the initial guesses
        cm = center_of_mass(pos, mass, boxsize=1)
        rad = mass_to_radius(numpy.sum(mass), rho)

        success = False
        for i in range(maxiter):
            # Calculate the distance of each particle from the current guess.
            dist = periodic_distance(pos, cm, boxsize=1)
            within_rad = dist <= rad
            # Calculate the enclosed mass for the current CM and radius.
            enclosed_mass = numpy.sum(mass[within_rad])

            # Calculate the new CM and radius from this mass.
            new_rad = mass_to_radius(enclosed_mass, rho)
            new_cm = center_of_mass(pos[within_rad], mass[within_rad],
                                    boxsize=1)

            # Update the CM and radius
            prev_cm, cm = cm, new_cm
            prev_rad, rad = rad, new_rad

            # Check if the change in CM and radius is small enough.
            dcm = numpy.linalg.norm(cm - prev_cm)
            drad = abs(rad - prev_rad)
            if dcm < tol or drad < tol:
                success = True
                break

        if not success:
            return numpy.nan, numpy.nan

        return enclosed_mass, rad, cm

    def angular_momentum(self, ref, rad):
        """
        Calculate angular momentum around a reference point using all particles
        within a radius. The angular momentum is returned in box units.

        Parameters
        ----------
        ref : 1-dimensional array of shape `(3, )`
            Reference point.
        rad : float
            Radius around the reference point.

        Returns
        -------
        angmom : 1-dimensional array or shape `(3, )`
        """
        pos = self.pos
        mask = periodic_distance(pos, ref, boxsize=1) < rad

        mass = self["M"][mask]
        pos = pos[mask]
        vel = self.vel[mask]
        # Velocitities in the object CM frame
        vel -= numpy.average(vel, axis=0, weights=mass)
        return numpy.sum(mass[:, numpy.newaxis] * numpy.cross(pos, vel),
                         axis=0)

    def lambda_bullock(self, ref, rad):
        r"""
        Bullock spin, see Eq. 5 in [1], in a given radius around a reference
        point.

        Parameters
        ----------
        ref : 1-dimensional array of shape `(3, )`
            Reference point.
        rad : float
            Radius around the reference point.

        Returns
        -------
        lambda_bullock : float

        References
        ----------
        [1] A Universal Angular Momentum Profile for Galactic Halos; 2001;
        Bullock, J. S.;  Dekel, A.;  Kolatt, T. S.;  Kravtsov, A. V.;
        Klypin, A. A.;  Porciani, C.;  Primack, J. R.
        """
        pos = self.pos
        mask = periodic_distance(pos, ref, boxsize=1) < rad
        mass = numpy.sum(self["M"][mask])
        circvel = numpy.sqrt(self.box.box_G * mass / rad)
        angmom_norm = numpy.linalg.norm(self.angular_momentum(ref, rad))
        return angmom_norm / (numpy.sqrt(2) * mass * circvel * rad)

    def nfw_concentration(self, ref, rad):
        """
        Calculate the NFW concentration parameter in a given radius around a
        reference point.

        Parameters
        ----------
        ref : 1-dimensional array of shape `(3, )`
            Reference point.
        rad : float
            Radius around the reference point.

        Returns
        -------
        conc : float
        """
        pos = self.pos
        dist = periodic_distance(pos, ref, boxsize=1)
        mask = dist < rad

        dist = dist[mask]
        weight = self["M"][mask] / numpy.mean(self["M"][mask])

        def negll_nfw_concentration(c, xs, weight):
            ll = xs / (1 + c * xs)**2 * c**2
            ll *= (1 + c) / ((1 + c) * numpy.log(1 + c) - c)
            ll = numpy.sum(numpy.log(weight * ll))
            return -ll

        res = minimize(negll_nfw_concentration, x0=5,
                       args=(dist / rad, weight, ), method='Nelder-Mead',
                       bounds=[(0, 100)])

        if not res.success:
            return numpy.nan

        return res["x"][0]

    def __getitem__(self, key):
        keys = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'M']
        if key not in keys:
            raise RuntimeError(f"Invalid key `{key}`!")
        return self.particles[:, keys.index(key)]

    def __len__(self):
        return self.particles.shape[0]


class Halo(BaseStructure):
    """
    Halo object to handle operations on its particles.

    Parameters
    ----------
    particles : structured array
        Particle array. Must contain `['x', 'y', 'z', 'vx', 'vy', 'vz', 'M']`.
    info : structured array
        Array containing information from the halo finder.
    box : :py:class:`csiborgtools.read.CSiBORGBox`
        Box units object.
    """

    def __init__(self, particles, box):
        self.particles = particles
        # self.info = info
        self.box = box


###############################################################################
#                       Other, supplementary functions                        #
###############################################################################


def center_of_mass(points, mass, boxsize):
    """
    Calculate the center of mass of a halo, while assuming for periodic
    boundary conditions of a cubical box. Assuming that particle positions are
    in `[0, boxsize)` range.

    Parameters
    ----------
    points : 2-dimensional array of shape (n_particles, 3)
        Particle position array.
    mass : 1-dimensional array of shape `(n_particles, )`
        Particle mass array.
    boxsize : float
        Box size in the same units as `parts` coordinates.

    Returns
    -------
    cm : 1-dimensional array of shape `(3, )`
    """
    # Convert positions to unit circle coordinates in the complex plane
    pos = numpy.exp(2j * numpy.pi * points / boxsize)
    # Compute weighted average of these coordinates, convert it back to
    # box coordinates and fix any negative positions due to angle calculations.
    cm = numpy.angle(numpy.average(pos, axis=0, weights=mass))
    cm *= boxsize / (2 * numpy.pi)
    cm[cm < 0] += boxsize
    return cm


def periodic_distance(points, reference, boxsize):
    """
    Compute the periodic distance between multiple points and a reference
    point.

    Parameters
    ----------
    points : 2-dimensional array of shape `(n_points, 3)`
        Points to calculate the distance from the reference point.
    reference : 1-dimensional array of shape `(3, )`
        Reference point.
    boxsize : float
        Box size.

    Returns
    -------
    dist : 1-dimensional array of shape `(n_points, )`
    """
    delta = numpy.abs(points - reference)
    delta = numpy.where(delta > boxsize / 2, boxsize - delta, delta)
    return numpy.linalg.norm(delta, axis=1)


def shift_to_center_of_box(points, cm, boxsize, set_cm_to_zero=False):
    """
    Shift the positions such that the CM is at the center of the box, while
    accounting for periodic boundary conditions.

    Parameters
    ----------
    points : 2-dimensional array of shape `(n_points, 3)`
        Points to shift.
    cm : 1-dimensional array of shape `(3, )`
        Center of mass.
    boxsize : float
        Box size.
    set_cm_to_zero : bool, optional
        If `True`, set the CM to zero.

    Returns
    -------
    shifted_positions : 2-dimensional array of shape `(n_points, 3)`
    """
    pos = (points + (boxsize / 2 - cm)) % boxsize
    if set_cm_to_zero:
        pos -= boxsize / 2
    return pos


def mass_to_radius(mass, rho):
    """
    Compute the radius of a sphere with a given mass and density.

    Parameters
    ----------
    mass : float
        Mass of the sphere.
    rho : float
        Density of the sphere.

    Returns
    -------
    rad : float
        Radius of the sphere.
    """
    return ((3 * mass) / (4 * numpy.pi * rho))**(1./3)




@jit(nopython=True)
def delta2ncells(delta):
    """
    Calculate the number of cells in `delta` that are non-zero.

    Parameters
    ----------
    delta : 3-dimensional array
        Halo density field.

    Returns
    -------
    ncells : int
        Number of non-zero cells.
    """
    tot = 0
    imax, jmax, kmax = delta.shape
    for i in range(imax):
        for j in range(jmax):
            for k in range(kmax):
                if delta[i, j, k] > 0:
                    tot += 1
    return tot


@jit(nopython=True)
def number_counts(x, bin_edges):
    """
    Calculate counts of samples in bins.

    Parameters
    ----------
    x : 1-dimensional array
        Samples' values.
    bin_edges : 1-dimensional array
        Bin edges.

    Returns
    -------
    counts : 1-dimensional array
        Bin counts.
    """
    out = numpy.full(bin_edges.size - 1, numpy.nan, dtype=numpy.float32)
    for i in range(bin_edges.size - 1):
        out[i] = numpy.sum((x >= bin_edges[i]) & (x < bin_edges[i + 1]))
    return out
