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
"""A clump object."""
from abc import ABC

import numpy


class BaseStructure(ABC):
    """
    Basic structure object for handling operations on its particles.
    """

    _particles = None
    _info = None
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
    def info(self):
        """
        Array containing information from the clump finder.

        Returns
        -------
        info : structured array
        """
        return self._info

    @info.setter
    def info(self, info):
        # TODO turn this into a structured array and add some checks
        self._info = info

    @property
    def box(self):
        """
        CSiBORG box object handling unit conversion.

        Returns
        -------
        box : :py:class:`csiborgtools.units.BoxUnits`
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
        Cartesian particle coordinates centered at the object.

        Returns
        -------
        pos : 2-dimensional array of shape `(n_particles, 3)`.
        """
        ps = ("x", "y", "z")
        return numpy.vstack([self[p] - self.info[p] for p in ps]).T

    @property
    def vel(self):
        """
        Cartesian particle velocity components.

        Returns
        -------
        vel : 2-dimensional array of shape (`n_particles, 3`)
        """
        return numpy.vstack([self[p] for p in ("vx", "vy", "vz")]).T

    def r(self):
        """
        Calculate the radial separation of the particles from the centre of the
        object.

        Returns
        -------
        r : 1-dimensional array of shape `(n_particles, )`.
        """
        return numpy.linalg.norm(self.pos, axis=1)

    def cmass(self, rmax, rmin):
        """
        Calculate Cartesian position components of the object's centre of mass.
        Note that this is already in a frame centered at the clump's potential
        minimum, so its distance from origin indicates the separation of the
        centre of mass and potential minimum.

        Parameters
        ----------
        rmax : float
            Maximum radius for particles to be included in the calculation.
        rmin : float
            Minimum radius for particles to be included in the calculation.

        Returns
        -------
        cm : 1-dimensional array of shape `(3, )`
        """
        r = self.r()
        mask = (r >= rmin) & (r <= rmax)
        return numpy.average(self.pos[mask], axis=0, weights=self["M"][mask])

    def angular_momentum(self, rmax, rmin=0):
        """
        Calculate angular momentum in the box coordinates.

        Parameters
        ----------
        rmax : float
            Maximum radius for particles to be included in the calculation.
        rmin : float
            Minimum radius for particles to be included in the calculation.

        Returns
        -------
        J : 1-dimensional array or shape `(3, )`
        """
        r = self.r()
        mask = (r >= rmin) & (r <= rmax)
        pos = self.pos[mask] - self.cmass(rmax, rmin)
        # Velocitities in the object CM frame
        vel = self.vel[mask]
        vel -= numpy.average(self.vel[mask], axis=0, weights=self["M"][mask])
        return numpy.einsum("i,ij->j", self["M"][mask], numpy.cross(pos, vel))

    def enclosed_mass(self, rmax, rmin=0):
        """
        Sum of particle masses between two radii.

        Parameters
        ----------
        rmax : float
            Maximum radial distance.
        rmin : float, optional
            Minimum radial distance.

        Returns
        -------
        enclosed_mass : float
        """
        r = self.r()
        return numpy.sum(self["M"][(r >= rmin) & (r <= rmax)])

    def lambda_bullock(self, radius, npart_min=10):
        r"""
        Bullock spin, see Eq. 5 in [1], in a radius of `radius`, which should
        define to some overdensity radius.

        Parameters
        ----------
        radius : float
            Radius in which to calculate the spin.
        npart_min : int
            Minimum number of enclosed particles for a radius to be
            considered trustworthy.

        Returns
        -------
        lambda_bullock : float

        References
        ----------
        [1] A Universal Angular Momentum Profile for Galactic Halos; 2001;
        Bullock, J. S.;  Dekel, A.;  Kolatt, T. S.;  Kravtsov, A. V.;
        Klypin, A. A.;  Porciani, C.;  Primack, J. R.
        """
        mask = self.r() <= radius
        if numpy.sum(mask) < npart_min:
            return numpy.nan
        mass = self.enclosed_mass(radius)
        V = numpy.sqrt(self.box.box_G * mass / radius)
        out = numpy.linalg.norm(self.angular_momentum(radius))
        out /= numpy.sqrt(2) * mass * V * radius
        return out

    def spherical_overdensity_mass(self, delta_mult, npart_min=10,
                                   kind="crit"):
        r"""
        Calculate spherical overdensity mass and radius. The mass is defined as
        the enclosed mass within an outermost radius where the mean enclosed
        spherical density reaches a multiple of the critical density `delta`
        (times the matter density if `kind` is `matter`).

        Parameters
        ----------
        delta_mult : list of int or float
            Overdensity multiple.
        npart_min : int
            Minimum number of enclosed particles for a radius to be
            considered trustworthy.
        kind : str
            Either `crit` or `matter`, for critical or matter overdensity

        Returns
        -------
        rx : float
            Radius where the enclosed density reaches required value.
        mx :  float
            Corresponding spherical enclosed mass.
        """
        # Quick check of inputs
        assert kind in ["crit", "matter"]

        # We first sort the particles in an increasing separation
        rs = self.r()
        order = numpy.argsort(rs)
        rs = rs[order]
        cmass = numpy.cumsum(self["M"][order])  # Cumulative mass
        # We calculate the enclosed volume and indices where it is above target
        vol = 4 * numpy.pi / 3 * (rs**3 - rs[0] ** 3)

        target_density = delta_mult * self.box.box_rhoc
        if kind == "matter":
            target_density *= self.box.cosmo.Om0
        with numpy.errstate(divide="ignore"):
            ks = numpy.where(cmass / vol > target_density)[0]
        if ks.size == 0:  # Never above the threshold?
            return numpy.nan, numpy.nan
        k = numpy.max(ks)
        if k < npart_min:  # Too few particles?
            return numpy.nan, numpy.nan
        return rs[k], cmass[k]

    def __getitem__(self, key):
        keys = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'M']
        if key not in self.keys:
            raise RuntimeError(f"Invalid key `{key}`!")
        return self.particles[:, keys.index(key)]

    def __len__(self):
        return self.particles.shape[0]


class Clump(BaseStructure):
    """
    Clump object to handle operations on its particles.

    Parameters
    ----------
    particles : structured array
        Particle array. Must contain `['x', 'y', 'z', 'vx', 'vy', 'vz', 'M']`.
    info : structured array
        Array containing information from the clump finder.
    box : :py:class:`csiborgtools.read.BoxUnits`
        Box units object.
    """

    def __init__(self, particles, info, box):
        self.particles = particles
        self.info = info
        self.box = box


class Halo(BaseStructure):
    """
    Ultimate halo object to handle operations on its particles, i.e. the summed
    particles halo.

    Parameters
    ----------
    particles : structured array
        Particle array. Must contain `['x', 'y', 'z', 'vx', 'vy', 'vz', 'M']`.
    info : structured array
        Array containing information from the clump finder.
    box : :py:class:`csiborgtools.read.BoxUnits`
        Box units object.
    """

    def __init__(self, particles, info, box):
        self.particles = particles
        self.info = info
        self.box = box
