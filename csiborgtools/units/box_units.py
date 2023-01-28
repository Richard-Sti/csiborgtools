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
"""
Simulation box unit transformations.
"""

import numpy
from scipy.interpolate import interp1d
from astropy.cosmology import LambdaCDM
from astropy import (constants, units)
from ..read import ParticleReader


# Map of unit conversions
CONV_NAME = {
    "length": ["peak_x", "peak_y", "peak_z", "Rs", "rmin", "rmax", "r200",
               "r500", "x0", "y0", "z0", "patch_size"],
    "mass": ["mass_cl", "totpartmass", "m200", "m500", "mass_mmain"],
    "density": ["rho0"]
    }


class BoxUnits:
    r"""
    Box units class for converting between box and physical units.

    Paramaters
    ----------
    paths : py:class`csiborgtools.read.CSiBORGPaths`
        CSiBORG paths-handling object with set `n_sim` and `n_snap`.
    """
    _cosmo = None

    def __init__(self, paths):
        """
        Read in the snapshot info file and set the units from it.
        """
        partreader = ParticleReader(paths)
        info = partreader.read_info()
        pars = ["boxlen", "time", "aexp", "H0",
                "omega_m", "omega_l", "omega_k", "omega_b",
                "unit_l", "unit_d", "unit_t"]
        for par in pars:
            setattr(self, "_" + par, float(info[par]))

        self._cosmo = LambdaCDM(H0=self._H0, Om0=self._omega_m,
                                Ode0=self._omega_l, Tcmb0=2.725 * units.K,
                                Ob0=self._omega_b)
        self._Msuncgs = constants.M_sun.cgs.value  # Solar mass in grams

    @property
    def cosmo(self):
        """
        The  box cosmology.

        Returns
        -------
        cosmo : `astropy.cosmology.LambdaCDM`
            The CSiBORG cosmology.
        """
        return self._cosmo

    @property
    def H0(self):
        r"""
        The Hubble parameter at the time of the snapshot
        in :math:`\mathrm{Mpc} / \mathrm{km} / \mathrm{s}`.

        Returns
        -------
        H0 : float
            Hubble constant.
        """
        return self._H0

    @property
    def h(self):
        r"""
        The little 'h` parameter at the time of the snapshot.

        Returns
        -------
        h : float
            The little h
        """
        return self._H0 / 100

    @property
    def box_G(self):
        """
        Gravitational constant :math:`G` in box units. Given everything else
        it looks like `self.unit_t` is in seconds.

        Returns
        -------
        G : float
            The gravitational constant.
        """
        return constants.G.cgs.value * (self._unit_d * self._unit_t ** 2)

    @property
    def box_H0(self):
        """
        Present time Hubble constant :math:`H_0` in box units.

        Returns
        -------
        H0 : float
            The Hubble constant.
        """
        return self.H0 * 1e5 / units.Mpc.to(units.cm) * self._unit_t

    @property
    def box_c(self):
        """
        Speed of light in box units.

        Returns
        -------
        c : float
            The speed of light.
        """
        return constants.c.cgs.value * self._unit_t / self._unit_l

    @property
    def box_rhoc(self):
        """
        Critical density in box units.

        Returns
        -------
        rhoc : float
            The critical density.
        """

        return 3 * self.box_H0 ** 2 / (8 * numpy.pi * self.box_G)

    def box2kpc(self, length):
        r"""
        Convert length from box units to :math:`\mathrm{ckpc}` (with
        :math:`h=0.705`). It appears that `self.unit_l` must be in
        :math:`\mathrm{cm}`.

        Parameters
        ----------
        length : float
            Length in box units.

        Returns
        -------
        length : foat
            Length in :math:`\mathrm{ckpc}`
        """
        return length * (self._unit_l / units.kpc.to(units.cm) / self._aexp)

    def kpc2box(self, length):
        r"""
        Convert length from :math:`\mathrm{ckpc}` (with :math:`h=0.705`) to
        box units.

        Parameters
        ----------
        length : float
            Length in :math:`\mathrm{ckpc}`

        Returns
        -------
        length : foat
            Length in box units.
        """
        return length / (self._unit_l / units.kpc.to(units.cm) / self._aexp)

    def box2mpc(self, length):
        r"""
        Convert length from box units to :math:`\mathrm{cMpc}` (with
        :math:`h=0.705`).

        Parameters
        ----------
        length : float
            Length in box units.

        Returns
        -------
        length : foat
            Length in :math:`\mathrm{ckpc}`
        """
        return self.box2kpc(length) * 1e-3

    def box2vel(self, vel):
        r"""
        Convert velocity from box units to :math:`\mathrm{m} / \mathrm{s}^2`.

        Parameters
        ----------
        vel : float
            Velocity in box units.

        Returns
        -------
        vel : float
            Velocity in :math:`\mathrm{m} / \mathrm{s}^2`
        """
        return vel * (1e-2 * self._unit_l / self._unit_t / self._aexp)

    def box2cosmoredshift(self, dist):
        r"""
        Convert the box comoving distance to cosmological redshift.

        NOTE: this likely is already the observed redshift.

        Parameters
        ----------
        dist : float
            Dist in box units.

        Returns
        -------
        cosmo_redshift : foat
            The cosmological redshift.
        """
        x = numpy.linspace(0., 1., 5001)
        y = self.cosmo.comoving_distance(x)
        return interp1d(y, x)(self.box2mpc(dist))

    def box2pecredshift(self, vx, vy, vz, px, py, pz, p0x=0, p0y=0, p0z=0):
        """
        Convert the box phase-space information to a peculiar redshift.

        NOTE: there is some confusion about this.

        Parameters
        ----------
        vx, vy, vz : 1-dimensional arrays
            The Cartesian velocity components.
        px, py, pz : 1-dimensional arrays
            The Cartesian position vectors components.
        p0x, p0y, p0z : floats
            The centre of the box. By default 0, in which it is assumed that
            the coordinates are already centred.

        Returns
        -------
        pec_redshift : 1-dimensional array
            The peculiar redshift.
        """
        # Peculiar velocity along the radial distance
        r = numpy.vstack([px - p0x, py - p0y, pz - p0z]).T
        norm = numpy.sum(r**2, axis=1)**0.5
        v = numpy.vstack([vx, vy, vz]).T
        vpec = numpy.sum(r * v, axis=1) / norm
        # Ratio between the peculiar velocity and speed of light
        x = self.box2vel(vpec) / constants.c.value
        # Doppler shift
        return numpy.sqrt((1 + x) / (1 - x)) - 1

    def box2obsredshift(self, vx, vy, vz, px, py, pz, p0x=0, p0y=0, p0z=0):
        """
        Convert the box phase-space information to an 'observed' redshift.

        NOTE: there is some confusion about this.

        Parameters
        ----------
        vx, vy, vz : 1-dimensional arrays
            The Cartesian velocity components.
        px, py, pz : 1-dimensional arrays
            The Cartesian position vectors components.
        p0x, p0y, p0z : floats
            The centre of the box. By default 0, in which it is assumed that
            the coordinates are already centred.

        Returns
        -------
        obs_redshift : 1-dimensional array
            The observed redshift.
        """
        r = numpy.vstack([px - p0x, py - p0y, pz - p0z]).T
        zcosmo = self.box2cosmoredshift(numpy.sum(r**2, axis=1)**0.5)
        zpec = self.box2pecredshift(vx, vy, vz, px, py, pz, p0x, p0y, p0z)
        return (1 + zpec) * (1 + zcosmo) - 1

    def solarmass2box(self, mass):
        r"""
        Convert mass from :math:`M_\odot` (with :math:`h=0.705`) to box units.

        Parameters
        ----------
        mass : float
            Mass in :math:`M_\odot`.

        Returns
        -------
        mass : float
            Mass in box units.
        """
        return mass / (self._unit_d * self._unit_l**3) * self._Msuncgs

    def box2solarmass(self, mass):
        r"""
        Convert mass from box units to :math:`M_\odot` (with :math:`h=0.705`).
        It appears that `self.unit_d` is density in units of
        :math:`\mathrm{g}/\mathrm{cm}^3`.

        Parameters
        ----------
        mass : float
            Mass in box units.

        Returns
        -------
        mass : float
            Mass in :math:`M_\odot`.
        """
        return mass * (self._unit_d * self._unit_l**3) / self._Msuncgs

    def box2dens(self, density):
        r"""
        Convert density from box units to :math:`M_\odot / \mathrm{Mpc}^3`
        (with :math:`h=0.705`).

        Parameters
        ----------
        density : float
            Density in box units.

        Returns
        -------
        density : float
            Density in :math:`M_\odot / \mathrm{pc}^3`.
        """
        return (density * self._unit_d / self._Msuncgs
                * (units.Mpc.to(units.cm))**3)

    def dens2box(self, density):
        r"""
        Convert density from :math:`M_\odot / \mathrm{Mpc}^3`
        (with :math:`h=0.705`) to box units.

        Parameters
        ----------
        density : float
            Density in :math:`M_\odot / \mathrm{pc}^3`.

        Returns
        -------
        density : float
            Density in box units.
        """
        return (density / self._unit_d * self._Msuncgs
                / (units.Mpc.to(units.cm))**3)

    def convert_from_boxunits(self, data, names):
        r"""
        Convert columns named `names` in array `data` from box units to
        physical units, such that
            - length -> :math:`Mpc`,
            - mass -> :math:`M_\odot`,
            - density -> :math:`M_\odot / \mathrm{Mpc}^3`.

        Any other conversions are currently not implemented. Note that the
        array is passed by reference and directly modified, even though it is
        also explicitly returned. Additionally centres the box coordinates on
        the observer, if they are being transformed.

        Parameters
        ----------
        data : structured array
            Input array.
        names : list of str
            Columns to be converted.

        Returns
        -------
        data : structured array
            Input array with converted columns.
        """
        names = [names] if isinstance(names, str) else names

        # Shortcut for the transform functions
        transforms = {
            "length": self.box2mpc,
            "mass": self.box2solarmass,
            "density": self.box2dens
            }

        for name in names:
            # Check that the name is even in the array
            if name not in data.dtype.names:
                raise ValueError("Name `{}` not in `data` array.".format(name))

            # Convert
            found = False
            for unittype, suppnames in CONV_NAME.items():
                if name in suppnames:
                    data[name] = transforms[unittype](data[name])
                    found = True
                    continue
            # If nothing found
            if not found:
                raise NotImplementedError(
                    "Conversion of `{}` is not defined.".format(name))

            # Center at the observer
            if name in ["peak_x", "peak_y", "peak_z", "x0", "y0", "z0"]:
                data[name] -= transforms["length"](0.5)

        return data
