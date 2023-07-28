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

from .readsim import CSiBORGReader, QuijoteReader


###############################################################################
#                              Base box                                       #
###############################################################################


class BaseBox(ABC):
    """
    Base class for box units.
    """
    _name = "box_units"
    _cosmo = None

    @property
    def cosmo(self):
        """
        The  box cosmology.

        Returns
        -------
        cosmo : `astropy.cosmology.LambdaCDM`
        """
        if self._cosmo is None:
            raise ValueError("Cosmology not set.")
        return self._cosmo

    @property
    def H0(self):
        r"""
        The Hubble parameter at the time of the snapshot in units of
        :math:`\mathrm{km} \mathrm{s}^{-1} \mathrm{Mpc}^{-1}`.

        Returns
        -------
        H0 : float
        """
        return self.cosmo.H0.value

    @property
    def h(self):
        r"""
        The little 'h' parameter at the time of the snapshot.

        Returns
        -------
        h : float
        """
        return self._h

    @property
    def Om0(self):
        r"""
        The matter density parameter.

        Returns
        -------
        Om0 : float
        """
        return self.cosmo.Om0

    @abstractproperty
    def boxsize(self):
        """
        Box size in cMpc.

        Returns
        -------
        boxsize : float
        """
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
        length : float
            Length in box units.
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
        length : float
            Length in :math:`\mathrm{cMpc} / h`
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
        mass : float
            Mass in box units.
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
        mass : float
            Mass in :math:`M_\odot / h`.
        """
        pass


###############################################################################
#                              CSiBORG box                                    #
###############################################################################


class CSiBORGBox(BaseBox):
    r"""
    CSiBORG box units class for converting between box and physical units.

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
        partreader = CSiBORGReader(paths)
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

    @property
    def box_G(self):
        """
        Gravitational constant :math:`G` in box units. Given everything else
        it looks like `self.unit_t` is in seconds.

        Returns
        -------
        G : float
        """
        return constants.G.cgs.value * (self._unit_d * self._unit_t**2)

    @property
    def box_H0(self):
        """
        Present time Hubble constant :math:`H_0` in box units.

        Returns
        -------
        H0 : float
        """
        return self.H0 * 1e5 / units.Mpc.to(units.cm) * self._unit_t

    @property
    def box_c(self):
        """
        Speed of light in box units.

        Returns
        -------
        c : float
        """
        return constants.c.cgs.value * self._unit_t / self._unit_l

    @property
    def box_rhoc(self):
        """
        Critical density in box units.

        Returns
        -------
        rhoc : float
        """
        return 3 * self.box_H0**2 / (8 * numpy.pi * self.box_G)

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


###############################################################################
#                      Quijote fiducial cosmology box                         #
###############################################################################


class QuijoteBox(BaseBox):
    """
    Quijote fiducial cosmology box.

    Parameters
    ----------
    nsnap : int
        Snapshot number.
    **kwargs : dict
        Empty keyword arguments. For backwards compatibility.
    """

    def __init__(self, nsnap, nsim, paths):
        zdict = {4: 0.0, 3: 0.5, 2: 1.0, 1: 2.0, 0: 3.0}
        assert nsnap in zdict.keys(), f"`nsnap` must be in {zdict.keys()}."
        info = QuijoteReader(paths).read_info(nsnap, nsim)
        self._aexp = 1 / (1 + zdict[nsnap])
        self._h = info["h"]
        self._cosmo = LambdaCDM(H0=100, Om0=info["Omega_m"],
                                Ode0=info["Omega_l"], Tcmb0=2.725 * units.K)
        self._info = info

    @property
    def boxsize(self):
        return self._info["BoxSize"]

    def box2mpc(self, length):
        return length * self.boxsize

    def mpc2box(self, length):
        return length / self.boxsize

    def solarmass2box(self, mass):
        r"""
        Convert mass from :math:`M_\odot / h` to box units.

        Parameters
        ----------
        mass : float
            Mass in :math:`M_\odot`.

        Returns
        -------
        mass : float
            Mass in box units.
        """
        return mass / self._info["TotMass"]

    def box2solarmass(self, mass):
        r"""
        Convert mass from box units to :math:`M_\odot / h`.

        Parameters
        ----------
        mass : float
            Mass in box units.

        Returns
        -------
        mass : float
            Mass in :math:`M_\odot / h`.
        """
        return mass * self._info["TotMass"]
