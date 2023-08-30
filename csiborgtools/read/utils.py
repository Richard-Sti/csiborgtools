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
from os.path import isfile

import numpy
from h5py import File


###############################################################################
#                          Array manipulation                                 #
###############################################################################


def cols_to_structured(N, cols):
    """
    Allocate a structured array from `cols`, a list of (name, dtype) tuples.
    """
    if not (isinstance(cols, list)
            and all(isinstance(c, tuple) and len(c) == 2 for c in cols)):
        raise TypeError("`cols` must be a list of (name, dtype) tuples.")

    names, formats = zip(*cols)
    dtype = {"names": names, "formats": formats}

    return numpy.full(N, numpy.nan, dtype=dtype)


def add_columns(arr, X, cols):
    """
    Add new columns `X` to a record array `arr`. Creates a new array.
    """
    cols = [cols] if isinstance(cols, str) else cols

    # Convert X to a list of 1D arrays for consistency
    if isinstance(X, numpy.ndarray) and X.ndim == 1:
        X = [X]
    elif isinstance(X, numpy.ndarray):
        raise ValueError("`X` should be a 1D array or a list of 1D arrays.")

    if len(X) != len(cols):
        raise ValueError("Mismatch between `X` and `cols` lengths.")

    if not all(isinstance(x, numpy.ndarray) and x.ndim == 1 for x in X):
        raise ValueError("All elements of `X` should be 1D arrays.")

    if not all(x.size == arr.size for x in X):
        raise ValueError("All arrays in `X` must have the same size as `arr`.")

    # Define new dtype
    dtype = list(arr.dtype.descr) + [(col, x.dtype) for col, x in zip(cols, X)]

    # Create a new array and fill in values
    out = numpy.empty(arr.size, dtype=dtype)
    for col in arr.dtype.names:
        out[col] = arr[col]
    for col, x in zip(cols, X):
        out[col] = x

    return out


def rm_columns(arr, cols):
    """
    Remove columns `cols` from a structured array `arr`. Allocates a new array.
    """
    # Ensure cols is a list
    cols = [cols] if isinstance(cols, str) else cols

    # Check columns we wish to delete are in the array
    missing_cols = [col for col in cols if col not in arr.dtype.names]
    if missing_cols:
        raise ValueError(f"Columns `{missing_cols}` not in `arr`.")

    # Define new dtype without the cols to be deleted
    new_dtype = [(n, dt) for n, dt in arr.dtype.descr if n not in cols]

    # Allocate a new array and fill in values
    out = numpy.empty(arr.size, dtype=new_dtype)
    for name in out.dtype.names:
        out[name] = arr[name]

    return out


def flip_cols(arr, col1, col2):
    """
    Flip values in columns `col1` and `col2` of a structured array `arr`.
    """
    if col1 not in arr.dtype.names or col2 not in arr.dtype.names:
        raise ValueError(f"Both `{col1}` and `{col2}` must exist in `arr`.")

    arr[col1], arr[col2] = numpy.copy(arr[col2]), numpy.copy(arr[col1])


###############################################################################
#                           h5py functions                                    #
###############################################################################


def read_h5(path):
    """
    Return and return and open `h5py.File` object.

    Parameters
    ----------
    path : str
        Path to the file.

    Returns
    -------
    file : `h5py.File`
    """
    if not isfile(path):
        raise IOError(f"File `{path}` does not exist!")
    return File(path, "r")
