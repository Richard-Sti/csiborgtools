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
2PCF calculation.

NOTE: This is an old script that needs to be updated.
"""
import numpy
from Corrfunc.mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf
from .utils import (rvs_on_sphere, wrapRA)


def sphere_angular_tpcf(bins, RA1, DEC1, RA2=None, DEC2=None, nthreads=1,
                        Nmult=5, seed1=42, seed2=666):
    """
    Calculate the angular two-point correlation function. The coordinates must
    be provided in degrees. With the right ascension and degrees being
    in range of :math:`[-180, 180]` and :math:`[-90, 90]` degrees.
    If `RA2` and `DEC2` are provided cross-correlates the first data set with
    the second. Creates a uniformly sampled randoms on the surface of a sphere
    of size `Nmult` times the corresponding number of data points. Uses the
    Landy-Szalay estimator.

    Parameters
    ----------
    bins : 1-dimensional array
        Angular bins to calculate the angular twop-point correlation function.
    RA1 : 1-dimensional array
        Right ascension of the 1st data set, in degrees.
    DEC1 : 1-dimensional array
        Declination of the 1st data set, in degrees.
    RA2 : 1-dimensional array, optional
        Right ascension of the 2nd data set, in degrees.
    DEC2 : 1-dimensional array, optional
        Declination of the 2nd data set, in degrees.
    nthreads : int, optional
        Number of threads, by default 1.
    Nmult : int, optional
        Relative randoms size with respect to the data set. By default 5.
    seed1 : int, optional
        Seed to generate the first set of randoms.
    seed2 : int, optional
        Seed to generate the second set of randoms.

    Returns
    -------
    cf : 1-dimensional array
        The angular 2-point correlation function.
    """
    # If not provided calculate autocorrelation
    if RA2 is None:
        RA2 = RA1
        DEC2 = DEC1
    # Get the array sizes
    ND1 = RA1.size
    ND2 = RA2.size
    NR1 = ND1 * Nmult
    NR2 = ND2 * Nmult
    # Generate randoms. Note that these are over the sphere!
    randRA1, randDEC1 = rvs_on_sphere(NR1, indeg=True, random_state=seed1)
    randRA2, randDEC2 = rvs_on_sphere(NR2, indeg=True, random_state=seed2)
    # Wrap RA
    RA1 = wrapRA(numpy.copy(RA1), indeg=True)
    RA2 = wrapRA(numpy.copy(RA2), indeg=True)
    # Calculate pairs
    D1D2 = DDtheta_mocks(0, nthreads, bins, RA1, DEC1, RA2=RA2, DEC2=DEC2)
    D1R2 = DDtheta_mocks(0, nthreads, bins, RA1, DEC1,
                         RA2=randRA2, DEC2=randDEC2)
    D2R1 = DDtheta_mocks(0, nthreads, bins, RA2, DEC2,
                         RA2=randRA1, DEC2=randDEC1)
    R1R2 = DDtheta_mocks(0, nthreads, bins, randRA1, randDEC1,
                         RA2=randRA2, DEC2=randDEC2)
    # Convert to the CF
    return convert_3d_counts_to_cf(ND1, ND2, NR1, NR2, D1D2, D1R2, D2R1, R1R2)