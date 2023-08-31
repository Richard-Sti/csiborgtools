# Copyright (C) 2022 Richard Stiskalek
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
Utility functions for the field module.
"""
import healpy
import numpy
import smoothing_library as SL


def force_single_precision(x):
    """
    Attempt to convert an array `x` to float 32.
    """
    if x.dtype != numpy.float32:
        x = x.astype(numpy.float32)
    return x


def smoothen_field(field, smooth_scale, boxsize, threads=1, make_copy=False):
    """
    Smooth a field with a Gaussian filter.
    """
    W_k = SL.FT_filter(boxsize, smooth_scale, field.shape[0], "Gaussian",
                       threads)

    if make_copy:
        field = numpy.copy(field)

    return SL.field_smoothing(field, W_k, threads)


def nside2radec(nside):
    """
    Generate RA [0, 360] deg. and declination [-90, 90] deg. for HEALPix pixel
    centres at a given nside.
    """
    pixs = numpy.arange(healpy.nside2npix(nside))
    theta, phi = healpy.pix2ang(nside, pixs)
    theta -= numpy.pi / 2
    return 180 / numpy.pi * numpy.vstack([phi, theta]).T
