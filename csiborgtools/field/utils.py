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
from warnings import warn

import numpy
import smoothing_library as SL


def force_single_precision(x, name):
    """
    Convert `x` to float32 if it is not already.

    Parameters
    ----------
    x : array
        Array to convert.
    name : str
        Name of the array.

    Returns
    -------
    x : array
        Converted array.
    """
    if x.dtype != numpy.float32:
        warn(f"Converting `{name}` to float32.", UserWarning, stacklevel=1)
        x = x.astype(numpy.float32)
    return x


def smoothen_field(field, smooth_scale, boxsize, threads=1):
    """
    Smooth a field with a Gaussian filter.

    Parameters
    ----------
    field : 3-dimensional array of shape `(grid, grid, grid)`
        Field to be smoothed.
    smooth_scale : float, optional
        Gaussian kernal scale to smoothen the density field, in box units.
    boxsize : float
        Size of the box.
    threads : int, optional
        Number of threads. By default 1.

    Returns
    -------
    smoothed_field : 3-dimensional array of shape `(grid, grid, grid)`
    """
    W_k = SL.FT_filter(boxsize, smooth_scale, field.shape[0], "Gaussian",
                       threads)
    return SL.field_smoothing(field, W_k, threads)
