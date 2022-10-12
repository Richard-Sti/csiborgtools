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

"""Utilility functions for manipulation structured arrays."""

import numpy


def add_columns(arr, X, cols):
    """
    Add new columns to a record array `arr`. Creates a new array.

    Parameters
    ----------
    arr : record array
        The record array to add columns to.
    X : (list of) 1-dimensional array(s) or 2-dimensional array
        Columns to be added.
    cols : str or list of str
        Column names to be added.

    Returns
    -------
    out : record array
        The new record array with added values.
    """
    # Make sure cols is a list of str and X a 2D array
    cols = [cols] if isinstance(cols, str) else cols
    if isinstance(X, numpy.ndarray) and X.ndim == 1:
        X = X.reshape(-1, 1)
    if isinstance(X, list) and all(x.ndim == 1 for x in X):
        X = numpy.vstack([X]).T
    if len(cols) != X.shape[1]:
        raise ValueError("Number of columns of `X` does not match `cols`.")
    if arr.size != X.shape[0]:
        raise ValueError("Number of rows of `X` does not match size of `arr`.")

    # Get the new data types
    dtype = arr.dtype.descr
    for i, col in enumerate(cols):
        dtype.append((col, X[i, :].dtype.descr[0][1]))

    # Fill in the old array
    out = numpy.full(arr.size, numpy.nan, dtype=dtype)
    for col in arr.dtype.names:
        out[col] = arr[col]
    for i, col in enumerate(cols):
        out[col] = X[:, i]

    return out

def rm_columns(arr, cols):
    """
    Remove columns `cols` from a record array `arr`. Creates a new array.

    Parameters
    ----------
    arr : record array
        The record array to remove columns from.
    cols : str or list of str
        Column names to be removed.

    Returns
    -------
    out : record array
        Record array with removed columns.
    """
    # Check columns we wish to delete are in the array
    cols = [cols] if isinstance(cols, str) else cols
    for col in cols:
        if col not in arr.dtype.names:
            raise ValueError("Column `{}` not in `arr`.".format(col))

    # Get a new dtype without the cols to be deleted
    new_dtype = []
    for dtype, name in zip(arr.dtype.descr, arr.dtype.names):
        if name not in cols:
            new_dtype.append(dtype)

    # Allocate a new array and fill it in.
    out = numpy.full(arr.size, numpy.nan, new_dtype)
    for name in out.dtype.names:
        out[name] = arr[name]

    return out
