# Copyright (C) 2024 Richard Stiskalek
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
Database of a few nearby observed clusters. Can be augmented with the list
compiled in https://arxiv.org/abs/2402.01834 or some eROSITA clusters?
"""
from csiborgtools.read import ObservedCluster
from .utils import hms_to_degrees, dms_to_degrees


# https://arxiv.org/abs/astro-ph/0702510
# https://arxiv.org/abs/2002.12820
# https://en.wikipedia.org/wiki/Virgo_Cluster
VIRGO = ObservedCluster(
    RA=hms_to_degrees(12, 27), dec=dms_to_degrees(12, 43), dist=16.5 * 0.73,
    mass=6.3e14 * 0.73, name="Virgo")

# https://arxiv.org/abs/astro-ph/0702320
# https://en.wikipedia.org/wiki/Fornax_Cluster
FORNAX = ObservedCluster(
    RA=hms_to_degrees(3, 35), dec=-35.7, dist=19.3 * 0.7,
    mass=7e13 * 0.73, name="Fornax")

# https://en.wikipedia.org/wiki/Coma_Cluster
# https://arxiv.org/abs/2311.08603
COMA = ObservedCluster(
    RA=hms_to_degrees(12, 59, 48.7), dec=dms_to_degrees(27, 58, 50),
    dist=102.975 * 0.705, mass=1.2e15 * 0.73, name="Coma")

# https://en.wikipedia.org/wiki/Perseus_Cluster
# https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.1681A/abstract
PERSEUS = ObservedCluster(
    RA=hms_to_degrees(3, 18), dec=dms_to_degrees(41, 30),
    dist=73.6 * 0.705, mass=1.2e15 * 0.7, name="Perseus")

# https://en.wikipedia.org/wiki/Centaurus_Cluster
# Not sure about the mass, couldn't find a good estimate. Some paper claimed
# 3e13 Msun, but that seems a little low?
CENTAURUS = ObservedCluster(
    RA=hms_to_degrees(12, 48, 51.8), dec=dms_to_degrees(-41, 18, 21),
    dist=52.4 * 0.705, mass=2e14 * 0.7, name="Centaurus")

# https://en.wikipedia.org/wiki/Shapley_Supercluster
# https://arxiv.org/abs/0805.0596
SHAPLEY = ObservedCluster(
    RA=hms_to_degrees(13, 25), dec=dms_to_degrees(-30),
    dist=136, mass=1e16 * 0.7, name="Shapley")

# https://en.wikipedia.org/wiki/Norma_Cluster
# https://arxiv.org/abs/0706.2227
NORMA = ObservedCluster(
    RA=hms_to_degrees(16, 15, 32.8), dec=dms_to_degrees(-60, 53, 30),
    dist=67.8 * 0.705, mass=1e15 * 0.7, name="Norma")

# Wikipedia seems to give the wrong distance.
# https://en.wikipedia.org/wiki/Leo_Cluster
# https://arxiv.org/abs/astro-ph/0406367
LEO = ObservedCluster(
    RA=hms_to_degrees(11, 44, 36.5), dec=dms_to_degrees(19, 43, 32),
    dist=91.3 * 0.705, mass=7e14 * 0.7, name="Leo")

# https://en.wikipedia.org/wiki/Hydra_Cluster
HYDRA = ObservedCluster(
    RA=hms_to_degrees(9, 18), dec=dms_to_degrees(-12, 5),
    dist=58.3 * 0.705, mass=4e15 * 0.7, name="Hydra")

# I think this is Pisces? Not very sure about its mass.
# https://en.wikipedia.org/wiki/Abell_262
# https://arxiv.org/abs/0911.1774
PISCES = ObservedCluster(
    RA=hms_to_degrees(1, 52, 50.4), dec=dms_to_degrees(36, 8, 46),
    dist=68.8 * 0.705, mass=2e14 * 0.7, name="Pisces")

# This one is in the ZOA
# https://en.wikipedia.org/wiki/Ophiuchus_Supercluster
# https://arxiv.org/abs/1509.00986
OPICHIUS = ObservedCluster(
    RA=hms_to_degrees(17, 10, 0), dec=dms_to_degrees(-22),
    dist=83.4, mass=1e15 * 0.7, name="Ophiuchus")


clusters = {"Virgo": VIRGO,
            "Fornax": FORNAX,
            "Coma": COMA,
            "Perseus": PERSEUS,
            "Centaurus": CENTAURUS,
            "Shapley": SHAPLEY,
            "Norma": NORMA,
            "Leo": LEO,
            "Hydra": HYDRA,
            "Pisces": PISCES,
            "Opichius": OPICHIUS,
            }
