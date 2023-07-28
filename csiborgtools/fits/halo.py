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


GRAV = 6.6743e-11               # m^3 kg^-1 s^-2
MSUN = 1.988409870698051e+30    # kg
MPC2M = 3.0856775814671916e+22  # 1 Mpc is this many meters


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

    def spherical_overdensity_mass(self, delta_mult, kind="crit", rtol=1e-8,
                                   maxiter=100, npart_min=10):
        r"""
        Calculate spherical overdensity mass and radius via the iterative
        shrinking sphere method.

        Parameters
        ----------
        delta_mult : int or float
            Overdensity multiple.
        kind : str, optional
            Either `crit` or `matter`, for critical or matter overdensity
        rtol : float, optional
            Tolerance for the change in the center of mass or radius.
        maxiter : int, optional
            Maximum number of iterations.
        npart_min : int, optional
            Minimum number of enclosed particles to reset the iterator.

        Returns
        -------
        mass :  float
            The requested spherical overdensity mass in :math:`M_\odot / h`.
        rad : float
            The radius of the sphere enclosing the requested overdensity in box
            units.
        cm : 1-dimensional array of shape `(3, )`
            The center of mass of the sphere enclosing the requested
            overdensity in box units.
        """
        assert kind in ["crit", "matter"]

        # Calculate density based on the provided kind
        rho = delta_mult * self.box.rho_crit0
        if kind == "matter":
            rho *= self.box.Om

        pos, mass = self.pos, self["M"]

        # Initial estimates for center of mass and radius
        init_cm = center_of_mass(pos, mass, boxsize=1)
        init_rad = self.box.mpc2box(mass_to_radius(numpy.sum(mass), rho) * 1.5)

        rad, cm = init_rad, numpy.copy(init_cm)

        for _ in range(maxiter):
            dist = periodic_distance(pos, cm, boxsize=1)
            within_rad = dist <= rad

            # Heuristic reset if too few enclosed particles
            if numpy.sum(within_rad) < npart_min:
                js = numpy.random.choice(len(self), len(self), replace=True)
                cm = center_of_mass(pos[js], mass[js], boxsize=1)
                rad = init_rad * (0.5 + numpy.random.rand())
                dist = periodic_distance(pos, cm, boxsize=1)
                within_rad = dist <= rad

            enclosed_mass = numpy.sum(mass[within_rad])
            new_rad = self.box.mpc2box(mass_to_radius(enclosed_mass, rho))
            new_cm = center_of_mass(pos[within_rad], mass[within_rad],
                                    boxsize=1)

            # Check convergence based on center of mass and radius
            cm_conv = numpy.linalg.norm(cm - new_cm) < rtol
            rad_conv = abs(rad - new_rad) < rtol

            if cm_conv or rad_conv:
                return enclosed_mass, rad, cm

            cm, rad = new_cm, new_rad

        # Return NaN values if no convergence after max iterations
        return numpy.nan, numpy.nan, numpy.full(3, numpy.nan, numpy.float32)

    def angular_momentum(self, ref, rad, npart_min=10):
        r"""
        Calculate angular momentum around a reference point using all particles
        within a radius. Units are
        :math:`(M_\odot / h) (\mathrm{Mpc} / h) \mathrm{km} / \mathrm{s}`.

        Parameters
        ----------
        ref : 1-dimensional array of shape `(3, )`
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
        # Calculate the distance of each particle from the reference point.
        distances = periodic_distance(self.pos, ref, boxsize=1)

        # Filter particles within the provided radius.
        mask = distances < rad
        if numpy.sum(mask) < npart_min:
            return numpy.full(3, numpy.nan, numpy.float32)

        mass, pos, vel = self["M"][mask], self.pos[mask], self.vel[mask]

        # Convert positions to Mpc / h and center around the reference point.
        pos = self.box.box2mpc(pos) - ref
        # Adjust velocities to be in the CM frame.
        vel -= numpy.average(vel, axis=0, weights=mass)
        # Calculate angular momentum.
        return numpy.sum(mass[:, numpy.newaxis] * numpy.cross(pos, vel),
                         axis=0)

    def lambda_bullock(self, ref, rad):
        r"""
        Bullock spin, see Eq. 5 in [1], in a given radius around a reference
        point.

        Parameters
        ----------
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
        # Filter particles within the provided radius
        mask = periodic_distance(self.pos, ref, boxsize=1) < rad
        # Calculate the total mass of the enclosed particles
        enclosed_mass = numpy.sum(self["M"][mask])
        # Convert the radius from box units to Mpc/h
        rad_mpc = self.box.box2mpc(rad)
        # Circular velocity in km/s
        circvel = (GRAV * enclosed_mass * MSUN / (rad_mpc * MPC2M))**0.5 * 1e-3
        # Magnitude of the angular momentum
        l_norm = numpy.linalg.norm(self.angular_momentum(ref, rad))
        # Compute and return the Bullock spin parameter
        return l_norm / (numpy.sqrt(2) * enclosed_mass * circvel * rad_mpc)

    def nfw_concentration(self, ref, rad, conc_min=1e-3, npart_min=10):
        """
        Calculate the NFW concentration parameter in a given radius around a
        reference point.

        Parameters
        ----------
        ref : 1-dimensional array of shape `(3, )`
            Reference point in box units.
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
        dist = periodic_distance(self.pos, ref, boxsize=1)
        mask = dist < rad

        if numpy.sum(mask) < npart_min:
            return numpy.nan

        dist, weight = dist[mask], self["M"][mask]
        weight /= numpy.mean(weight)

        # Objective function for minimization
        def negll_nfw_concentration(log_c, xs, w):
            c = 10**log_c
            ll = xs / (1 + c * xs)**2 * c**2
            ll *= (1 + c) / ((1 + c) * numpy.log(1 + c) - c)
            ll = numpy.sum(numpy.log(w * ll))
            return -ll

        initial_guess = 1.5
        res = minimize(negll_nfw_concentration, x0=initial_guess,
                       args=(dist / rad, weight, ), method='Nelder-Mead',
                       bounds=((numpy.log10(conc_min), 5),))

        if not res.success:
            return numpy.nan

        conc_value = 10**res["x"][0]
        if conc_value < conc_min or numpy.isclose(conc_value, conc_min):
            return numpy.nan

        return conc_value

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
