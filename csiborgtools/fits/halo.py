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
import numpy


class Clump:
    r"""
    A clump object handling operations with its particles.

    Parameters
    ----------
    x : 1-dimensional array
        Particle coordinates along the x-axis.
    y : 1-dimensional array
        Particle coordinates along the y-axis.
    z : 1-dimensional array
        Particle coordinates along the z-axis.
    m : 1-dimensional array
        Particle masses.
    x0 : float
        Clump center coordinate along the x-axis.
    y0 : float
        Clump center coordinate along the y-axis.
    z0 : float
        Clump center coordinate along the z-axis.
    clump_mass : float, optional
        Mass of the clump. By default not set.
    vx : 1-dimensional array, optional
        Particle velocity along the x-axis. By default not set.
    vy : 1-dimensional array, optional
        Particle velocity along the y-axis. By default not set.
    vz : 1-dimensional array, optional
        Particle velocity along the z-axis. By default not set.
    index : int, optional
        The halo finder index of this clump. By default not set.
    rhoc : float, optional
        The critical density :math:`\rho_c` at this snapshot in box units. By
        default not set.
    G : float, optional
        The gravitational constant :math:`G` in box units. By default not set.
    """
    _pos = None
    _clump_pos = None
    _clump_mass = None
    _vel = None
    _index = None
    _rhoc = None
    _G = None

    def __init__(self, x, y, z, m, x0, y0, z0, clump_mass=None,
                 vx=None, vy=None, vz=None, index=None, rhoc=None, G=None):
        self._pos = numpy.vstack([x - x0, y - y0, z - z0]).T
        self._clump_pos = numpy.asarray((x0, y0, z0))
        assert clump_mass is None or isinstance(clump_mass, float)
        self._clump_mass = clump_mass
        if all(v is not None for v in (vx, vy, vz)):
            self._vel = numpy.vstack([vx, vy, vz]).T
            assert self._vel.shape == self.pos.shape
        assert m.ndim == 1 and m.size == self.Npart
        self._m = m
        assert index is None or (isinstance(index, (int, numpy.int64)) and index >= 0)  # noqa
        self._index = index
        assert rhoc is None or rhoc > 0
        self._rhoc = rhoc
        assert G is None or G > 0
        self._G = G

    @property
    def pos(self):
        """
        Cartesian particle coordinates centered at the clump.

        Returns
        -------
        pos : 2-dimensional array of shape `(n_particles, 3)`.
        """
        return self._pos

    @property
    def Npart(self):
        """
        Number of particles associated with this clump.

        Returns
        -------
        Npart : int
        """
        return self.pos.shape[0]

    @property
    def r(self):
        """
        Radial distance of the particles from the clump peak.

        Returns
        -------
        r : 1-dimensional array of shape `(n_particles, )`.
        """
        return numpy.sum(self.pos**2, axis=1)**0.5

    @property
    def rmin(self):
        """
        The minimum radial distance of a particle.

        Returns
        -------
        rmin : float
        """
        return numpy.min(self.r)

    @property
    def rmax(self):
        """
        The maximum radial distance of a particle.

        Returns
        -------
        rmin : float
        """
        return numpy.max(self.r)

    @property
    def clump_pos(self):
        """
        Cartesian position components of the clump.

        Returns
        -------
        pos : 1-dimensional array of shape `(3, )`
        """
        return self._clump_pos

    @property
    def clump_mass(self):
        """
        Clump mass.

        Returns
        -------
        mass : float
        """
        if self._clump_mass is None:
            raise ValueError("Clump mass `clump_mass` has not been set.")
        return self._clump_mass

    @property
    def vel(self):
        """
        Cartesian velocity components of the clump.

        Returns
        -------
        vel : 2-dimensional array of shape (`n_particles, 3`)
        """
        if self._vel is None:
            raise ValueError("Velocities `vel` have not been set.")
        return self._vel

    @property
    def m(self):
        """
        Particle masses.

        Returns
        -------
        m : 1-dimensional array of shape `(n_particles, )`
        """
        return self._m

    @property
    def center_mass(self):
        """
        Cartesian position components of the clump centre of mass. Note that
        this is already in a frame centered at the clump's potential minimum.

        Returns
        -------
        cm : 1-dimensional array of shape `(3, )`
        """
        return numpy.average(self.pos, axis=0, weights=self.m)

    @property
    def angular_momentum(self):
        """
        Clump angular momentum in the box coordinates.

        Returns
        -------
        J : 1-dimensional array or shape `(3, )`
        """
        J = numpy.cross(self.pos - self.center_mass, self.vel)
        return numpy.einsum("i,ij->j", self.m, J)

    @property
    def lambda200c(self):
        r"""
        Clump Bullock spin, see Eq. 5 in [1], in a radius of
        :math:`R_{\rm 200c}`.

        References
        ----------
        [1] A Universal Angular Momentum Profile for Galactic Halos; 2001;
        Bullock, J. S.;  Dekel, A.;  Kolatt, T. S.;  Kravtsov, A. V.;
        Klypin, A. A.;  Porciani, C.;  Primack, J. R.

        Returns
        -------
        lambda200c : float
        """
        J = self.angular_momentum
        R, M = self.spherical_overdensity_mass(200)
        V = numpy.sqrt(self.G * M / R)
        return numpy.linalg.norm(J) / (numpy.sqrt(2) * M * V * R)

    @property
    def index(self):
        """
        Halo finder clump index.

        Returns
        -------
        hindex : int
        """
        if self._index is None:
            raise ValueError("Halo index `hindex` has not been set.")
        return self._index

    @property
    def rhoc(self):
        r"""
        Critical density :math:`\rho_c` at this snapshot in box units.

        Returns
        -------
        rhoc : float
        """
        if self._rhoc is None:
            raise ValueError("The critical density `rhoc` has not been set.")
        return self._rhoc

    @property
    def G(self):
        r"""
        Gravitational constant :math:`G` in box units.

        Returns
        -------
        G : float
        """
        if self._G is None:
            raise ValueError("The grav. constant `G` has not been set.")
        return self._G

    @property
    def total_particle_mass(self):
        """
        Total mass of all particles.

        Returns
        -------
        tot_mass : float
        """
        return numpy.sum(self.m)

    @property
    def mean_particle_pos(self):
        """
        Mean Cartesian particle coordinate. Not centered at the halo!

        Returns
        -------
        pos : 1-dimensional array of shape `(3, )`
        """
        return numpy.mean(self.pos + self.clump_pos, axis=0)

    def enclosed_spherical_mass(self, rmax, rmin=0):
        """
        Enclosed spherical mass between two radii in box units.

        Parameters
        ----------
        rmax : float
            The maximum radial distance.
        rmin : float, optional
            The minimum radial distance. By default 0.

        Returns
        -------
        M_enclosed : float
            The enclosed mass.
        """
        return numpy.sum(self.m[(self.r >= rmin) & (self.r <= rmax)])

    def enclosed_spherical_volume(self, rmax, rmin=0):
        """
        Enclosed spherical volume within two radii in box units.

        Parameters
        ----------
        rmax : float
            Maximum radial distance.
        rmin : float, optional
            Minimum radial distance. By default 0.

        Returns
        -------
        vol : float
        """
        return 4 * numpy.pi / 3 * (rmax**3 - rmin**3)

    def spherical_overdensity_mass(self, delta, n_particles_min=10):
        r"""
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
        # If single `delta` turn to list
        delta = [delta] if isinstance(delta, (float, int)) else delta
        # If given a list or tuple turn to array
        _istlist = isinstance(delta, (list, tuple))
        delta = numpy.asarray(delta, dtype=float) if _istlist else delta

        # Ordering of deltas
        order_delta = numpy.argsort(delta)
        # Sort the particles
        order_particles = numpy.argsort(self.r)[::-1]
        # Density to aim for
        n_delta = delta.size
        target_density = delta * self.rhoc

        # The sum of particle masses, starting from the outside
        # Adds the furtherst particle ensure that the 0th index is tot mass
        cummass_ordered = (self.total_particle_mass
                           + self.m[order_particles][0]
                           - numpy.cumsum(self.m[order_particles]))
        # Enclosed volumes at particle radii
        volumes = self.enclosed_spherical_volume(self.r[order_particles])
        densities = cummass_ordered / volumes

        # Pre-allocate arrays
        rfound = numpy.full_like(delta, numpy.nan)
        mfound = numpy.full_like(rfound, numpy.nan)

        for n in order_delta:
            overdense_mask = densities > target_density[n]

            # Enforce that we have at least several particles enclosed
            if numpy.sum(overdense_mask) < n_particles_min:
                continue
            # The outermost particle radius where the overdensity is achieved
            k = numpy.where(overdense_mask)[0][0]
            rfound[n] = self.r[order_particles][k]
            mfound[n] = cummass_ordered[k]

        # If only one delta return simply numbers
        if n_delta == 1:
            rfound = rfound[0]
            mfound = mfound[0]

        return rfound, mfound

    @classmethod
    def from_arrays(cls, particles, clump, rhoc=None, G=None):
        r"""
        Initialises `Clump` from `particles` containing the relevant particle
        information and its `clump` information.

        Paramaters
        ----------
        particles : structured array
            Array of particles belonging to this clump. Must contain
            `["x", "y", "z", "M"]` and optionally also `["vx", "vy", "vz"]`.
        clump : array
            A slice of a `clumps` array corresponding to this clump. Must
            contain `["peak_x", "peak_y", "peak_z", "mass_cl"]`.
        rhoc : float, optional
            The critical density :math:`\rho_c` at this snapshot in box units.
            By default not set.
        G : float, optional
            The gravitational constant :math:`G` in box units. By default not
            set.

        Returns
        -------
        clump : `Clump`
        """
        x, y, z, m = (particles[p] for p in ["x", "y", "z", "M"])
        x0, y0, z0, cl_mass, hindex = (
            clump[p] for p in ["peak_x", "peak_y", "peak_z", "mass_cl",
                               "index"])
        try:
            vx, vy, vz = (particles[p] for p in ["vx", "vy", "vz"])
        except ValueError:
            vx, vy, vz = None, None, None
        return cls(x, y, z, m, x0, y0, z0, cl_mass,
                   vx, vy, vz, hindex, rhoc, G)
