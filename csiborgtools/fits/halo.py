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

GRAV = 4.300917270069976e-09  # G in (Msun / h)^-1 (Mpc / h) (km / s)^2


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

    def center_of_mass(self, npart_min=30, shrink_factor=0.98):
        r"""
        Calculate the center of mass of a halo via the shrinking sphere
        procedure. Iteratively reduces initial radius and calculates the CM of
        enclosed particles while the number of enclosed particles is greater
        than a set minimum.


        Parameters
        ----------
        npart_min : int, optional
            Minimum number of enclosed particles above which to continue
            shrinking the sphere.
        shrink_factor : float, optional
            Factor by which to shrink the sphere radius at each iteration.

        Returns
        -------
        cm : 1-dimensional array of shape `(3, )`
            Center of mass in box units.
        dist : 1-dimensional array of shape `(n_particles, )`
            Distance of each particle from the center of mass in box units.
        """
        pos, mass = self.pos, self["M"]

        cm = center_of_mass(pos, mass, boxsize=1)
        rad = None

        while True:
            dist = periodic_distance(pos, cm, boxsize=1)

            if rad is None:
                rad = numpy.max(dist)

            within_rad = dist <= rad

            cm = center_of_mass(pos[within_rad], mass[within_rad], boxsize=1)

            if numpy.sum(within_rad) < npart_min:
                return cm, periodic_distance(pos, cm, boxsize=1)

            rad *= shrink_factor

    def spherical_overdensity_mass(self, dist, delta_mult, kind="crit"):
        r"""
        Calculate spherical overdensity mass and radius around a CM, defined as
        the inner-most radius where the density falls below a given threshold.
        The exact radius is found via linear interpolation between the two
        particles enclosing the threshold.

        Parameters
        ----------
        dist : 1-dimensional array of shape `(n_particles, )`
            Distance of each particle from the centre of mass in box units.
        delta_mult : int or float
            Overdensity multiple.
        kind : str, optional
            Either `crit` or `matter`, for critical or matter overdensity

        Returns
        -------
        mass :  float
            Overdensity mass in (Msun / h).
        rad : float
            Overdensity radius in box units.
        """
        if kind not in ["crit", "matter"]:
            raise ValueError("kind must be either `crit` or `matter`.")

        rho = delta_mult * self.box.rho_crit0
        rho *= self.box.Om if kind == "matter" else 1.

        argsort = numpy.argsort(dist)
        dist = self.box.box2mpc(dist[argsort])

        norm_density = numpy.cumsum(self['M'][argsort])
        totmass = norm_density[-1]
        norm_density /= (4. / 3. * numpy.pi * dist**3)
        norm_density /= rho

        # This ensures that the j - 1 index is also just above 1, therefore the
        # expression below strictly interpolates.
        j = find_first_below_threshold(norm_density, 1.)

        if j is None:
            return numpy.nan, numpy.nan

        i = j - 1

        rad = (dist[j] - dist[i])
        rad *= (1. - norm_density[i]) / (norm_density[j] - norm_density[i])
        rad += dist[i]

        mass = radius_to_mass(rad, rho)
        rad = self.box.mpc2box(rad)

        if mass > totmass:
            return numpy.nan, numpy.nan

        return mass, rad

    def angular_momentum(self, dist, cm, rad, npart_min=10):
        r"""
        Calculate angular momentum around a centre of mass using all particles
        within a radius. Accounts for periodicity of the box and units are
        (Msun / h) * (Mpc / h) * (km / s).

        Parameters
        ----------
        dist : 1-dimensional array of shape `(n_particles, )`
            Distance of each particle from center of mass in box units.
        cm : 1-dimensional array of shape `(3, )`
            Reference point in box units.
        rad : float
            Radius around the reference point in box units.
        npart_min : int, optional
            Minimum number of enclosed particles for a radius to be
            considered trustworthy.

        Returns
        -------
        angmom : 1-dimensional array or shape `(3, )`
        """
        mask = dist < rad

        if numpy.sum(mask) < npart_min:
            return numpy.full(3, numpy.nan, numpy.float32)

        mass, pos, vel = self["M"][mask], self.pos[mask], self.vel[mask]

        pos = shift_to_center_of_box(pos, cm, 1.0, set_cm_to_zero=True)
        pos = self.box.box2mpc(pos)
        vel -= numpy.average(vel, axis=0, weights=mass)

        return numpy.sum(mass[:, numpy.newaxis] * numpy.cross(pos, vel),
                         axis=0)

    def lambda_bullock(self, angmom, mass, rad):
        """
        Calculate the Bullock spin, see Eq. 5 in [1].

        Parameters
        ----------
        angmom : 1-dimensional array of shape `(3, )`
            Angular momentum in (Msun / h) * (Mpc / h) * (km / s).
        ref : 1-dimensional array of shape `(3, )`
            Reference point in box units.
        rad : float
            Radius around the reference point in box units.

        Returns
        -------
        lambda_bullock : float

        References
        ----------
        [1] A Universal Angular Momentum Profile for Galactic Halos; 2001;
        Bullock, J. S.; Dekel, A.;  Kolatt, T. S.; Kravtsov, A. V.;
        Klypin, A. A.; Porciani, C.; Primack, J. R.
        """
        out = numpy.linalg.norm(angmom)
        return out / numpy.sqrt(2 * GRAV * mass**3 * self.box.box2mpc(rad))

    def nfw_concentration(self, dist, rad, conc_min=1e-3, npart_min=10):
        """
        Calculate the NFW concentration parameter in a given radius around a
        reference point.

        Parameters
        ----------
        dist : 1-dimensional array of shape `(n_particles, )`
            Distance of each particle from center of mass in box units.
        rad : float
            Radius around the reference point in box units.
        conc_min : float
            Minimum concentration limit.
        npart_min : int, optional
            Minimum number of enclosed particles to calculate the
            concentration.

        Returns
        -------
        conc : float
        """
        mask = dist < rad
        if numpy.sum(mask) < npart_min:
            return numpy.nan

        dist, weight = dist[mask], self["M"][mask]
        weight /= weight[0]

        res = minimize(negll_nfw_concentration, x0=1.,
                       args=(dist / rad, weight, ), method='Nelder-Mead',
                       bounds=((numpy.log10(conc_min), 5),))

        if not res.success:
            return numpy.nan

        conc = 10**res["x"][0]
        if conc < conc_min or numpy.isclose(conc, conc_min):
            return numpy.nan

        return conc

    def __getitem__(self, key):
        key_to_index = {'x': 0, 'y': 1, 'z': 2,
                        'vx': 3, 'vy': 4, 'vz': 5, 'M': 6}
        if key not in key_to_index:
            raise RuntimeError(f"Invalid key `{key}`!")
        return self.particles[:, key_to_index[key]]

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
        self.box = box


###############################################################################
#                       Other, supplementary functions                        #
###############################################################################


@jit(nopython=True, fastmath=True, boundscheck=False)
def center_of_mass(points, mass, boxsize):
    """
    Calculate the center of mass of a halo while assuming periodic boundary
    conditions of a cubical box. Assuming that particle positions are in
    `[0, boxsize)` range. This is a JIT implementation.

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
    cm = numpy.zeros(3, dtype=points.dtype)
    totmass = sum(mass)

    # Convert positions to unit circle coordinates in the complex plane,
    # calculate the weighted average and convert it back to box coordinates.
    for i in range(3):
        cm_i = sum(mass * numpy.exp(2j * numpy.pi * points[:, i] / boxsize))
        cm_i /= totmass

        cm_i = numpy.arctan2(cm_i.imag, cm_i.real) * boxsize / (2 * numpy.pi)

        if cm_i < 0:
            cm_i += boxsize
        cm[i] = cm_i

    return cm


@jit(nopython=True)
def periodic_distance(points, reference, boxsize):
    """
    Compute the 3D distance between multiple points and a reference point using
    periodic boundary conditions. This is an optimized JIT implementation.

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
    npoints = len(points)
    half_box = boxsize / 2

    dist = numpy.zeros(npoints, dtype=points.dtype)
    for i in range(npoints):
        for j in range(3):
            dist_1d = abs(points[i, j] - reference[j])

            if dist_1d > (half_box):
                dist_1d = boxsize - dist_1d

            dist[i] += dist_1d**2

        dist[i] = dist[i]**0.5

    return dist


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


@jit(nopython=True, fastmath=True, boundscheck=False)
def radius_to_mass(radius, rho):
    """
    Compute the mass of a sphere with a given radius and density.

    Parameters
    ----------
    radius : float
        Radius of the sphere.
    rho : float
        Density of the sphere.

    Returns
    -------
    mass : float
    """
    return ((4 * numpy.pi * rho) / 3) * radius**3


@jit(nopython=True, fastmath=True, boundscheck=False)
def find_first_below_threshold(x, threshold):
    """
    Find index of first element in `x` that is below `threshold`. The index
    must be greater than 0. If no such element is found, return `None`.

    Parameters
    ----------
    x : 1-dimensional array
        Array to search in.
    threshold : float
        Threshold value.

    Returns
    -------
    index : int or None
    """
    for i in range(1, len(x)):
        if 1 < x[i - 1] and x[i] < threshold:
            return i
    return None


@jit(nopython=True, fastmath=True, boundscheck=False)
def negll_nfw_concentration(log_c, xs, w):
    """
    Negative log-likelihood of the NFW concentration parameter.

    Parameters
    ----------
    log_c : float
        Logarithm of the concentration parameter.
    xs : 1-dimensional array
        Normalised radii.
    w : 1-dimensional array
        Weights.

    Returns
    ------
    negll : float
    """
    c = 10**log_c
    ll = xs / (1 + c * xs)**2 * c**2
    ll *= (1 + c) / ((1 + c) * numpy.log(1 + c) - c)
    ll = numpy.sum(numpy.log(w * ll))
    return -ll


@jit(nopython=True, fastmath=True, boundscheck=False)
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


@jit(nopython=True, fastmath=True, boundscheck=False)
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
