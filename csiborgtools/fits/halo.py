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
from ..units import BoxUnits


class BaseStructure(ABC):
    r"""
    A clump object handling operations with its particles.
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
        pars = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'M']
        assert all(p in particles.dtype.names for p in pars)
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
        assert isinstance(box, BoxUnits)
        self._box = box

    @property
    def pos(self):
        """
        Cartesian particle coordinates centered at the object.

        Returns
        -------
        pos : 2-dimensional array of shape `(n_particles, 3)`.
        """
        ps = ('x', 'y', 'z')
        return numpy.vstack([self[p] - self.info[p] for p in ps]).T

    @property
    def r(self):
        """
        Radial separation of the particles from the centre of the object.

        Returns
        -------
        r : 1-dimensional array of shape `(n_particles, )`.
        """
        return numpy.linalg.norm(self.pos, axis=1)

    @property
    def vel(self):
        """
        Cartesian particle velocity components.

        Returns
        -------
        vel : 2-dimensional array of shape (`n_particles, 3`)
        """
        return numpy.vstack([self[p] for p in ("vx", "vy", "vz")]).T

    @property
    def cmass(self):
        """
        Cartesian position components of the object's centre of mass. Note that
        this is already in a frame centered at the clump's potential minimum,
        so its distance from origin indicates the separation of the centre of
        mass and potential minimum.

        Returns
        -------
        cm : 1-dimensional array of shape `(3, )`
        """
        return numpy.average(self.pos, axis=0, weights=self['M'])

    @property
    def angular_momentum(self):
        """
        Angular momentum in the box coordinates.

        NOTE: here also change velocities to the CM and appropriately edit the
        docs.

        Returns
        -------
        J : 1-dimensional array or shape `(3, )`
        """
        J = numpy.cross(self.pos - self.cmass, self.vel)
        return numpy.einsum("i,ij->j", self.m, J)

    def lambda_bullock(self, ):
        r"""
        Bullock spin, see Eq. 5 in [1], in a radius of :math:`R_{\rm x}`.

        TODO: docs and correct this function. Watch out up to where calculating

        Parameters
        ----------
        delta : int or float
            Overdensity multiple...
        n_particles_min : int
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
        J = self.angular_momentum
        R, M = self.spherical_overdensity_mass(200)
        V = numpy.sqrt(self.G * M / R)
        return numpy.linalg.norm(J) / (numpy.sqrt(2) * M * V * R)

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
        r = self.r
        return numpy.sum(self['M'][(r >= rmin) & (r <= rmax)])

    def spherical_overdensity_mass(self, delta, npart_min=10):
        r"""
        TODO: docs

        Spherical overdensity mass and radius. The mass is defined as the
        enclosed mass within a radius of where the mean enclosed spherical
        density reaches a multiple of the critical radius at a given redshift
        `self.rho_c`.

        Starts from the furthest particle, working its way inside the halo
        through an ordered list of particles. The corresponding values is the
        radial distance of the first particle whose addition sufficiently
        increases the mean density.


        Parameters
        ----------
        delta : list of int or float
            The :math:`\delta_{\rm x}` parameters where :math:`\mathrm{x}` is
            the overdensity multiple.
        n_particles_min : int
            Minimum number of enclosed particles for a radius to be
            considered trustworthy.

        Returns
        -------
        rx : float
            Radius where the enclosed density reaches required value.
        mx :  float
            Corresponding spherical enclosed mass.
        """
        # We first sort the particles in an increasing separation
        rs = self.r
        order = numpy.argsort(rs)
        rs = rs[order]
        cmass = numpy.cumsum(self['M'])  # Cumulative mass
        # We calculate the enclosed volume and indices where it is above target
        vol = 4 * numpy.pi / 3 * (rs**3 - rs[0]**3)
        ks = numpy.where([cmass / vol > delta * self.box.rhoc])[0]
        if ks.size == 0:  # Never above the threshold?
            return numpy.nan, numpy.nan
        k = numpy.maximum(ks)
        if k < npart_min:  # Too few particles?
            return numpy.nan, numpy.nan
        return rs[k], cmass[k]

    @property
    def keys(self):
        """
        Particle array keys.

        Returns
        -------
        key : list of str
        """
        return self.data.dtype.names

    def __getitem__(self, key):
        if key not in self.keys:
            raise RuntimeError("Invalid key `{}`!".format(key))
        return self.particles[key]

    def __len__(self):
        return self.particles.size


class Clump(BaseStructure):

    def __init__(self, particles, info, box):
        self.particles = particles
        self.info = info
        self.box = box

