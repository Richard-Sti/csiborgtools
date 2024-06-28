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
Various user parameters for CSiBORGTools.
"""

SPEED_OF_LIGHT = 299792.458  # km / s
CB2_REDSHIFT = [69.0000210000063, 40.250007218751264, 28.24050991940438,
                21.6470609550175, 17.480001404480106, 14.608109099433955,
                12.508772664512199, 10.90721705951751, 9.64516173673259,
                8.625000360937513, 7.7832702592057235, 7.0769233254437935,
                6.475728365821477, 5.95783150553419, 5.50704240932355,
                5.111111246913583, 4.760598622974984, 4.448113312911626,
                4.1677853285437605, 3.914893700679041, 3.685598452365574,
                3.476744253718227, 3.285714346938776, 3.1103203402819117,
                2.9487179993425383, 2.7993421515051513, 2.6608558268213116,
                2.5321101306287352, 2.4121122957547967, 2.3000000330000008,
                2.1950207773798662, 2.096514773533915, 2.003901196522936,
                1.9166666909722223, 1.8343558508261513, 1.7565632668759008,
                1.6829268488994646, 1.613122190273029, 1.5468577900064306,
                1.4838709837669097, 1.4239244641145379, 1.366803292753544,
                1.3123123255056859, 1.2602739849878026, 1.210526327423823,
                1.162921359250726, 1.117323566656109, 1.0736086272735772,
                1.0316622782422846, 0.9913793189283591, 0.9526627299814432,
                0.9154228931957131, 0.8795768989699038, 0.8450479301016136,
                0.8117647122768166, 0.7796610229819017, 0.7486752517178681,
                0.7187500053710938, 0.6898317534223188, 0.6618705083794834,
                0.6348195374209455, 0.6086351017498701, 0.5832762206018658,
                0.5587044572276223, 0.5348837244997295, 0.5117801080759505,
                0.48936170529651424, 0.46759847820604516, 0.4464621192761633,
                0.42592592856652933, 0.4059647012034677, 0.3865546241790834,
                0.3676731815824261, 0.34929906746973005, 0.3314121056648591,
                0.31399317585528075, 0.2970241454144613, 0.28048780643961924,
                0.2643678175452504, 0.2486486499985392, 0.23331553782343795,
                0.21835443153641232, 0.20375195520916023, 0.18949536658248856,
                0.17557251998135315, 0.1619718318042056, 0.14868224838055033,
                0.13569321600925854, 0.122994653006949, 0.11057692361085425,
                0.09843081359419292, 0.08654750746436402, 0.0749185671253807,
                0.06353591189600438, 0.05239179978414388, 0.04147880992632613,
                0.03078982610853953, 0.020318021291547472,
                0.010056843069963017, 0.0]


def snap2redshift(snapnum, simname):
    """Convert a snapshot number to redshift."""
    if "csiborg2_" in simname:
        try:
            return CB2_REDSHIFT[snapnum]
        except KeyError:
            raise ValueError(f"Unknown snapshot: `{snapnum}`.")
    else:
        raise ValueError(f"Unsupported simulation: `{simname}`.")


def simname2boxsize(simname):
    """Return boxsize in `Mpc/h` for a given simulation."""
    d = {"csiborg1": 677.7,
         "csiborg2_main": 676.6,
         "csiborg2_varysmall": 676.6,
         "csiborg2_random": 676.6,
         "csiborg2X": 681.1,
         "borg1": 677.7,
         "borg2": 676.6,
         "borg2_all": 676.6,
         "quijote": 1000.,
         "TNG300-1": 205.,
         "Carrick2015": 400.,
         "CF4": 1000.,  # These need to be checked with Helene Courtois.
         "CF4gp": 1000.,
         "Lilow2024": 400.,
         }
    boxsize = d.get(simname, None)

    if boxsize is None:
        raise ValueError(f"Unknown simulation: `{simname}`.")

    return boxsize


def simname2Omega_m(simname):
    """Return `Omega_m` for a given simulation"""
    d = {"csiborg1": 0.307,
         "csiborg2_main": 0.3111,
         "csiborg2_random": 0.3111,
         "csiborg2_varysmall": 0.3111,
         "csiborg2X": 0.306,
         "borg1": 0.307,
         "borg2": 0.3111,
         "borg2_all": 0.3111,
         "Carrick2015": 0.3,
         "CF4": 0.3,
         "CF4gp": 0.3,
         "Lilow2024": 0.3175,
         }

    omega_m = d.get(simname, None)

    if omega_m is None:
        raise ValueError(f"Unknown simulation: `{simname}`.")

    return omega_m


paths_glamdring = {
    "csiborg1_srcdir": "/mnt/extraspace/rstiskalek/csiborg1",
    "csiborg2_main_srcdir": "/mnt/extraspace/rstiskalek/csiborg2_main",
    "csiborg2_varysmall_srcdir": "/mnt/extraspace/rstiskalek/csiborg2_varysmall",   # noqa
    "csiborg2_random_srcdir": "/mnt/extraspace/rstiskalek/csiborg2_random",
    "postdir": "/mnt/extraspace/rstiskalek/csiborg_postprocessing/",
    "quijote_dir": "/mnt/extraspace/rstiskalek/quijote",
    "borg1_dir": "/mnt/users/hdesmond/BORG_final",
    "borg2_dir": "/mnt/extraspace/rstiskalek/BORG_STOPYRA_2023",
    "tng300_1_dir": "/mnt/extraspace/rstiskalek/TNG300-1/",
    "aux_cat_dir": "/mnt/extraspace/rstiskalek/catalogs",
    }
