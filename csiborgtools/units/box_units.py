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
from astropy.cosmology import LambdaCDM
from astropy import (constants, units)
from ..io import read_info


# Conversion factors
MSUNCGS = constants.M_sun.cgs.value
KPC_TO_CM = 3.0856775814913673e+21


class BoxUnits:
    r"""
    Box units class for converting between box and physical units.

    TODO: check factors of :math:`a` in mass and density transformations

    Paramaters
    ----------
    Nsnap : int
        Snapshot index.
    simpath : str
        Path to the simulation where its snapshot index folders are stored.
    """
    _info = {}
    _cosmo = None

    def __init__(self, Nsnap, simpath):
        """
        Read in the snapshot info file and set the units from it.
        """
        info = read_info(Nsnap, simpath)
        names_extract = ["boxlen", "time", "aexp", "H0",
                         "omega_m", "omega_l", "omega_k", "omega_b",
                         "unit_l", "unit_d", "unit_t"]
        for name in names_extract:
            self._info.update({name: float(info[name])})
        self._cosmo = LambdaCDM(H0=self.H0, Om0=self.omega_m,
                                Ode0=self.omega_l, Tcmb0=2.725 * units.K,
                                Ob0=self.omega_b)

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
    def box_G(self):
        """
        Gravitational constant :math:`G` in box units.

        Returns
        -------
        G : float
            The gravitational constant.
        """
        return constants.G.cgs.value * (self.unit_d * self.unit_t ** 2)

    @property
    def box_H0(self):
        """
        Present time Hubble constant :math:`H_0` in box units.

        Returns
        -------
        H0 : float
            The Hubble constant.
        """
        return self.H0 * 1e5 / (1e3 * KPC_TO_CM) * self.unit_t

    @property
    def box_c(self):
        """
        Speed of light in box units.

        Returns
        -------
        c : float
            The speed of light.
        """
        return constants.c.cgs.value * self.unit_t / self.unit_l

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
        return length * self.unit_l / KPC_TO_CM / self.aexp

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
        return length / self.unit_l * KPC_TO_CM * self.aexp

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
        return mass / self.unit_d / (self.unit_l**3 / MSUNCGS)

    def box2solarmass(self, mass):
        r"""
        Convert mass from box units to :math:`M_\odot` (with :math:`h=0.705`).

        Parameters
        ----------
        mass : float
            Mass in box units.

        Returns
        -------
        mass : float
            Mass in :math:`M_\odot`.
        """
        return mass * self.unit_d * self.unit_l**3 / MSUNCGS

    def box2dens(self, density):
        r"""
        Convert density from box units to :math:`M_\odot / \mathrm{pc}^3`
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
        return density * self.unit_d / MSUNCGS * (KPC_TO_CM * 1e-3)**3

    def dens2box(self, density):
        r"""
        Convert density from :math:`M_\odot / \mathrm{pc}^3`
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
        return density / self.unit_d * MSUNCGS / (KPC_TO_CM * 1e-3)**3

    def __getattr__(self, attr):
        return self._info[attr]
