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
from abc import ABC, abstractproperty

import numpy
from astropy import constants, units
from astropy.cosmology import LambdaCDM

from .readsim import ParticleReader

# Map of CSiBORG unit conversions
CONV_NAME = {
    "length": ["x", "y", "z", "peak_x", "peak_y", "peak_z", "Rs", "rmin",
               "rmax", "r200c", "r500c", "r200m", "x0", "y0", "z0",
               "lagpatch_size"],
    "velocity": ["vx", "vy", "vz"],
    "mass": ["mass_cl", "totpartmass", "m200c", "m500c", "mass_mmain", "M",
             "m200m"],
    "density": ["rho0"]}


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
        The little 'h` parameter at the time of the snapshot.

        Returns
        -------
        h : float
        """
        return self.H0 / 100

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
        partreader = ParticleReader(paths)
        info = partreader.read_info(nsnap, nsim)
        pars = ["boxlen", "time", "aexp", "H0", "omega_m", "omega_l",
                "omega_k", "omega_b", "unit_l", "unit_d", "unit_t"]
        for par in pars:
            setattr(self, "_" + par, float(info[par]))

        self._cosmo = LambdaCDM(H0=self._H0, Om0=self._omega_m,
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

    def mpc2box(self, length):
        r"""
        Convert length from :math:`\mathrm{cMpc}` (with :math:`h=0.705`) to
        box units.

        Parameters
        ----------
        length : float
            Length in :math:`\mathrm{cMpc}`

        Returns
        -------
        length : foat
            Length in box units.
        """
        return self.kpc2box(length * 1e3)

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
        return (density * self._unit_d
                / self._Msuncgs * (units.Mpc.to(units.cm)) ** 3)

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
                / (units.Mpc.to(units.cm)) ** 3)

    def convert_from_box(self, data, names):
        r"""
        Convert columns named `names` in array `data` from box units to
        physical units, such that
            - length -> :math:`Mpc`,
            - mass -> :math:`M_\odot`,
            - velocity -> :math:`\mathrm{km} / \mathrm{s}`,
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
        transforms = {"length": self.box2mpc,
                      "mass": self.box2solarmass,
                      "velocity": self.box2vel,
                      "density": self.box2dens}

        for name in names:
            # Check that the name is even in the array
            if name not in data.dtype.names:
                raise ValueError(f"Name `{name}` not in `data` array.")

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
                    f"Conversion of `{name}` is not defined.")

            # Center at the observer
            if name in ["peak_x", "peak_y", "peak_z", "x0", "y0", "z0"]:
                data[name] -= transforms["length"](0.5)

        return data

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

    def __init__(self, nsnap, **kwargs):
        zdict = {4: 0.0, 3: 0.5, 2: 1.0, 1: 2.0, 0: 3.0}
        assert nsnap in zdict.keys(), f"`nsnap` must be in {zdict.keys()}."
        self._aexp = 1 / (1 + zdict[nsnap])

        self._cosmo = LambdaCDM(H0=67.11, Om0=0.3175, Ode0=0.6825,
                                Tcmb0=2.725 * units.K, Ob0=0.049)

    @property
    def boxsize(self):
        return 1000. / (self._cosmo.H0.value / 100)
