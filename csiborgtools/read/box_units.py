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
from abc import ABC, abstractmethod, abstractproperty

import numpy
from astropy import constants, units
from astropy.cosmology import LambdaCDM

from .readsim import CSiBORG1Reader

###############################################################################
#                              Base box                                       #
###############################################################################


class BaseBox(ABC):
    _name = "box_units"
    _cosmo = None

    @property
    def cosmo(self):
        if self._cosmo is None:
            raise ValueError("Cosmology not set.")
        return self._cosmo

    @property
    def H0(self):
        r"""Present Hubble parameter in :math:`\mathrm{km} \mathrm{s}^{-1}`"""
        return self.cosmo.H0.value

    @property
    def rho_crit0(self):
        """Present-day critical density in M_sun h^2 / cMpc^3."""
        rho_crit0 = self.cosmo.critical_density0
        return rho_crit0.to_value(units.solMass / units.Mpc**3)

    @property
    def h(self):
        """The little 'h' parameter at the time of the snapshot."""
        return self._h

    @property
    def Om0(self):
        """The present time matter density parameter."""
        return self.cosmo.Om0

    @abstractproperty
    def boxsize(self):
        """Box size in cMpc."""
        pass

    @abstractmethod
    def mpc2box(self, length):
        r"""
        Convert length from :math:`\mathrm{cMpc} / h` to box units.

        Parameters
        ----------
        length : float
            Length in :math:`\mathrm{cMpc}`

        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def box2mpc(self, length):
        r"""
        Convert length from box units to :math:`\mathrm{cMpc} / h`.

        Parameters
        ----------
        length : float
            Length in box units.

        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def solarmass2box(self, mass):
        r"""
        Convert mass from :math:`M_\odot / h` to box units.

        Parameters
        ----------
        mass : float
            Mass in :math:`M_\odot / h`.

        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def box2solarmass(self, mass):
        r"""
        Convert mass from box units to :math:`M_\odot / h`.

        Parameters
        ----------
        mass : float
            Mass in box units.

        Returns
        -------
        float
        """
        pass

    @abstractmethod
    def m200c_to_r200c(self, m200c):
        """
        Convert M200c to R200c in units of cMpc / h.

        Parameters
        ----------
        m200c : float
            M200c in units of M_sun / h.

        Returns
        -------
        float
        """
        pass


###############################################################################
#                              CSiBORG box                                    #
###############################################################################


class CSiBORG1Box(BaseBox):
    r"""
    CSiBORG1 box units class for converting between box and physical units.

    Paramaters
    ----------
    nsnap : int
        Snapshot index.
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.Paths`
        CSiBORG paths object.
    """

    def __init__(self, nsnap, nsim, paths):
        """
        Read in the snapshot info file and set the units from it.
        """
        partreader = CSiBORG1Reader(paths)
        info = partreader.read_info(nsnap, nsim)
        pars = ["boxlen", "time", "aexp", "H0", "omega_m", "omega_l",
                "omega_k", "omega_b", "unit_l", "unit_d", "unit_t"]
        for par in pars:
            setattr(self, "_" + par, info[par])
        self._h = self._H0 / 100
        self._cosmo = LambdaCDM(H0=100, Om0=self._omega_m,
                                Ode0=self._omega_l, Tcmb0=2.725 * units.K,
                                Ob0=self._omega_b)
        self._Msuncgs = constants.M_sun.cgs.value  # Solar mass in grams

    def mpc2box(self, length):
        conv = (self._unit_l / units.kpc.to(units.cm) / self._aexp) * 1e-3
        conv *= self._h
        return length / conv

    def box2mpc(self, length):
        conv = (self._unit_l / units.kpc.to(units.cm) / self._aexp) * 1e-3
        conv *= self._h
        return length * conv

    def solarmass2box(self, mass):
        conv = (self._unit_d * self._unit_l**3) / self._Msuncgs
        conv *= self.h
        return mass / conv

    def box2solarmass(self, mass):
        conv = (self._unit_d * self._unit_l**3) / self._Msuncgs
        conv *= self.h
        return mass * conv

    def box2vel(self, vel):
        r"""
        Convert velocity from box units to :math:`\mathrm{km} \mathrm{s}^{-1}`.

        Parameters
        ----------
        vel : float
            Velocity in box units.

        Returns
        -------
        vel : float
            Velocity in :math:`\mathrm{km} \mathrm{s}^{-1}`.
        """
        return vel * (1e-2 * self._unit_l / self._unit_t / self._aexp) * 1e-3

    @property
    def boxsize(self):
        return self.box2mpc(1.)

    def m200c_to_r200c(self, m200c):
        rho_crit = self.cosmo.critical_density(1 / self._aexp - 1)
        rho_crit = rho_crit.to_value(units.solMass / units.Mpc**3)
        r200c = (3 * m200c / (4 * numpy.pi * 200 * rho_crit))**(1 / 3)
        return r200c / self._aexp
