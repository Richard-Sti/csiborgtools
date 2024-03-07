# Copyright (C) 2023 Richard Stiskalek
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
Various user parameters for csiborgtools.
"""


def simname2boxsize(simname):
    """
    Return boxsize in `Mpc/h` for a given simname.

    Parameters
    ----------
    simname : str
        Simulation name.

    Returns
    -------
    boxsize : float
    """
    d = {"csiborg1": 677.7,
         "csiborg2_main": 676.6,
         "csiborg2_varysmall": 676.6,
         "csiborg2_random": 676.6,
         "borg1": 677.7,
         "borg2": 676.6,
         "quijote": 1000.,
         "TNG300-1": 205.,
         "Carrick2015": 400.,
         }

    boxsize = d.get(simname, None)

    if boxsize is None:
        raise ValueError("Unknown simname: {}".format(simname))

    return boxsize


def simname2Omega_m(simname):
    """
    Return Omega_m for a given simname.

    Parameters
    ----------
    simname : str
        Simulation name.

    Returns
    -------
    Omega_m: float
    """
    d = {"csiborg1": 0.307,
         "borg1": 0.307,
         "Carrick2015": 0.3,
         }

    omega_m = d.get(simname, None)

    if omega_m is None:
        raise ValueError("Unknown simname: {}".format(simname))

    return omega_m


paths_glamdring = {
    "csiborg1_srcdir": "/mnt/extraspace/rstiskalek/csiborg1",
    "csiborg2_main_srcdir": "/mnt/extraspace/rstiskalek/csiborg2_main",
    "csiborg2_varysmall_srcdir": "/mnt/extraspace/rstiskalek/csiborg2_varysmall",   # noqa
    "csiborg2_random_srcdir": "/mnt/extraspace/rstiskalek/csiborg2_random",         # noqa
    "postdir": "/mnt/extraspace/rstiskalek/csiborg_postprocessing/",
    "quijote_dir": "/mnt/extraspace/rstiskalek/quijote",
    "borg1_dir": "/mnt/users/hdesmond/BORG_final",
    "borg2_dir": "/mnt/extraspace/rstiskalek/BORG_STOPYRA_2023",
    "tng300_1_dir": "/mnt/extraspace/rstiskalek/TNG300-1/",
    }


# neighbour_kwargs = {"rmax_radial": 155 / 0.705,
#                     "nbins_radial": 50,
#                     "rmax_neighbour": 100.,
#                     "nbins_neighbour": 150,
#                     "paths_kind": paths_glamdring}
