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


from astropy.cosmology import LambdaCDM
from astropy import (constants, units)
from ..io import read_info


# Conversion factors
MSUNCGS = constants.M_sun.cgs.value
KPC_TO_CM = 3.08567758149137e21
PI = 3.1415926535897932384626433


class BoxUnits:
    """
    Box units class for converting between box and physical units.

    Paramaters
    ----------
    Nsnap : int
        Snapshot index.
    simpath : str
        Path to the simulation where its snapshot index folders are stored.
    """

    def __init__(self, Nsnap, simpath):
        """
        Read in the snapshot info file and set the units from it.
        """
        info = read_info(Nsnap, simpath)
        pars = ["boxlen", "time", "aexp", "H0",
                "omega_m", "omega_l", "omega_k", "omega_b",
                "unit_l", "unit_d", "unit_t"]
        for par in pars:
            setattr(self, par, float(info[par]))

        self.h = self.H0 / 100
        self.cosmo = LambdaCDM(H0=self.H0, Om0=self.omega_m, Ode0=self.omega_l,
                               Tcmb0=2.725 * units.K, Ob0=self.omega_b)
        # Constants in box units
        self.G = constants.G.cgs.value * (self.unit_d * self.unit_t ** 2)
        self.H0 = self.H0 * 1e5 / (1e3 * KPC_TO_CM) * self.unit_t
        self.c = constants.c.cgs.value * self.unit_t / self.unit_l
        self.rho_crit = 3 * self.H0 ** 2 / (8 * PI * self.G)

    def box2kpc(self, length):
        r"""
        Convert length from box units to :math:`\mathrm{kpc}`.

        Parameters
        ----------
        length : float
            Length in box units.

        Returns
        -------
        length : foat
            Length in :math:`\mathrm{kpc}`
        """
        return length * self.unit_l / KPC_TO_CM

    def kpc2box(self, length):
        r"""
        Convert length from :math:`\mathrm{kpc}` to box units.

        Parameters
        ----------
        length : float
            Length in :math:`\mathrm{kpc}`

        Returns
        -------
        length : foat
            Length in box units.
        """
        return length / self.unit_l * KPC_TO_CM

    def solarmass2box(self, mass):
        r"""
        Convert mass from :math:`M_\odot` to box units.

        Parameters
        ----------
        mass : float
            Mass in :math:`M_\odot`.

        Returns
        -------
        mass : float
            Mass in box units.
        """
        m = mass * MSUNCGS   # In cgs
        unit_m = self.unit_d * self.unit_l ** 3
        return m / unit_m

    def box2solarmass(self, mass):
        r"""
        Convert mass from box units to :math:`M_\odot`.

        TODO: check this.

        Parameters
        ----------
        mass : float
            Mass in box units.

        Returns
        -------
        mass : float
            Mass in :math:`M_\odot`.
        """
        unit_m = self.unit_d * self.unit_l**3
        m = mass * unit_m  # In cgs
        m = m / MSUNCGS
        return m

    def box2dens(self, density):
        r"""
        Convert density from box units to :math:`M_\odot / \mathrm{pc}^3`.

        TODO: check this.

        Parameters
        ----------
        density : float
            Density in box units.
        box : `BoxConstants`
            Simulation box class with units.

        Returns
        -------
        density : float
            Density in :math:`M_\odot / \mathrm{pc}^3`.
        """
        rho = density * self.unit_d  # In cgs
        rho = rho * (KPC_TO_CM * 1e-3)**3  # In g/pc^3
        rho = rho / MSUNCGS
        return rho

    def dens2box(self, density):
        r"""
        Convert density from M_sun / pc^3

        TODO: check this and write documentation.
        """
        rho = density * MSUNCGS
        rho = rho / (KPC_TO_CM * 1e-3)**3  # In g/cm^3
        rho = rho / self.unit_d
        return rho
