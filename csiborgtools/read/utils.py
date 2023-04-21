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
Various coordinate transformations.
"""
import numpy

###############################################################################
#                          Coordinate transforms                              #
###############################################################################


def cartesian_to_radec(X, indeg=True):
    """
    Calculate the radial distance, RA, dec from Cartesian coordinates. Note,
    RA is in range [0, 360) degrees and dec in range [-90, 90] degrees.

    Parameters
    ----------
    X : 2-dimensional array `(nsamples, 3)`
        Cartesian coordinates.
    indeg : bool, optional
        Whether to return RA and DEC in degrees.

    Returns
    -------
    out : 2-dimensional array `(nsamples, 3)`
        Radial distance, RA and dec.
    """
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    dist = numpy.linalg.norm(X, axis=1)
    dec = numpy.arcsin(z / dist)
    ra = numpy.arctan2(y, x)
    ra[ra < 0] += 2 * numpy.pi  # Wrap RA to [0, 2pi)
    if indeg:
        ra = numpy.rad2deg(ra)
        dec = numpy.rad2deg(dec)
    return numpy.vstack([dist, ra, dec]).T


def radec_to_cartesian(X, isdeg=True):
    """
    Calculate Cartesian coordinates from radial distance, RA, dec. Note, RA is
    expected in range [0, 360) degrees and dec in range [-90, 90] degrees.

    Parameters
    ----------
    X : 2-dimensional array `(nsamples, 3)`
        Radial distance, RA and dec.
    isdeg : bool, optional
        Whether to return RA and DEC in degrees.

    Returns
    -------
    out : 2-dimensional array `(nsamples, 3)`
        Cartesian coordinates.
    """
    dist, ra, dec = X[:, 0], X[:, 1], X[:, 2]
    if isdeg:
        ra = numpy.deg2rad(ra)
        dec = numpy.deg2rad(dec)
    x = numpy.cos(dec) * numpy.cos(ra)
    y = numpy.cos(dec) * numpy.sin(ra)
    z = numpy.sin(dec)
    return dist * numpy.vstack([x, y, z]).T


###############################################################################
#                          Array manipulation                                 #
###############################################################################


def cols_to_structured(N, cols):
    """
    Allocate a structured array from `cols`.

    Parameters
    ----------
    N : int
        Structured array size.
    cols: list of tuples
        Column names and dtypes. Each tuple must written as `(name, dtype)`.

    Returns
    -------
    out : structured array
        Initialised structured array.
    """
    if not isinstance(cols, list) and all(isinstance(c, tuple) for c in cols):
        raise TypeError("`cols` must be a list of tuples.")

    dtype = {"names": [col[0] for col in cols],
             "formats": [col[1] for col in cols]}
    return numpy.full(N, numpy.nan, dtype=dtype)


def add_columns(arr, X, cols):
    """
    Add new columns to a record array `arr`. Creates a new array.

    Parameters
    ----------
    arr : record array
        Record array to add columns to.
    X : (list of) 1-dimensional array(s) or 2-dimensional array
        Columns to be added.
    cols : str or list of str
        Column names to be added.

    Returns
    -------
    out : record array
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
        Record array to remove columns from.
    cols : str or list of str
        Column names to be removed.

    Returns
    -------
    out : record array
    """
    # Check columns we wish to delete are in the array
    cols = [cols] if isinstance(cols, str) else cols
    for col in cols:
        if col not in arr.dtype.names:
            raise ValueError("Column `{}` not in `arr`.".format(col))

    # Get a new dtype without the cols to be deleted
    new_dtype = []
    for dtype, name in zip(arr.dtype.descr, arr.dtype.names, strict=True):
        if name not in cols:
            new_dtype.append(dtype)

    # Allocate a new array and fill it in.
    out = numpy.full(arr.size, numpy.nan, new_dtype)
    for name in out.dtype.names:
        out[name] = arr[name]

    return out


def list_to_ndarray(arrs, cols):
    """
    Convert a list of structured arrays of CSiBORG simulation catalogues to
    an 3-dimensional array.

    Parameters
    ----------
    arrs : list of structured arrays
        List of CSiBORG catalogues.
    cols : str or list of str
        Columns to be extracted from the CSiBORG catalogues.

    Returns
    -------
    out : 3-dimensional array
        Catalogue array of shape `(n_realisations, n_samples, n_cols)`, where
        `n_samples` is the maximum number of samples over the CSiBORG
        catalogues.
    """
    if not isinstance(arrs, list):
        raise TypeError("`arrs` must be a list of structured arrays.")
    cols = [cols] if isinstance(cols, str) else cols

    Narr = len(arrs)
    Nobj_max = max([arr.size for arr in arrs])
    Ncol = len(cols)
    # Preallocate the array and fill it
    out = numpy.full((Narr, Nobj_max, Ncol), numpy.nan)
    for i in range(Narr):
        Nobj = arrs[i].size
        for j in range(Ncol):
            out[i, :Nobj, j] = arrs[i][cols[j]]
    return out


def array_to_structured(arr, cols):
    """
    Create a structured array from a 2-dimensional array.

    Parameters
    ----------
    arr : 2-dimensional array
        Original array of shape `(n_samples, n_cols)`.
    cols : list of str
        Columns of the structured array

    Returns
    -------
    out : structured array
        Output structured array.
    """
    cols = [cols] if isinstance(cols, str) else cols
    if arr.ndim != 2 and arr.shape[1] != len(cols):
        raise TypeError("`arr` must be a 2D array `(n_samples, n_cols)`.")

    dtype = {"names": cols, "formats": [arr.dtype] * len(cols)}
    out = numpy.full(arr.shape[0], numpy.nan, dtype=dtype)
    for i, col in enumerate(cols):
        out[col] = arr[:, i]

    return out


def flip_cols(arr, col1, col2):
    """
    Flip values in columns `col1` and `col2`. `arr` is passed by reference and
    is not explicitly returned back.

    Parameters
    ----------
    arr : structured array
        Array whose columns are to be converted.
    col1 : str
        First column name.
    col2 : str
        Second column name.

    Returns
    -------
    None
    """
    dum = numpy.copy(arr[col1])
    arr[col1] = arr[col2]
    arr[col2] = dum


def extract_from_structured(arr, cols):
    """
    Extract columns `cols` from a structured array. The  array dtype is set
    to be that of the first column in `cols`.

    Parameters
    ----------
    arr : structured array
        Array from which to extract columns.
    cols : list of str or str
        Column to extract.

    Returns
    -------
    out : 2- or 1-dimensional array
        Array with shape `(n_particles, len(cols))`. If `len(cols)` is 1
        flattens the array.
    """
    cols = [cols] if isinstance(cols, str) else cols
    for col in cols:
        if col not in arr.dtype.names:
            raise ValueError("Invalid column `{}`!".format(col))
    # Preallocate an array and populate it
    out = numpy.zeros((arr.size, len(cols)), dtype=arr[cols[0]].dtype)
    for i, col in enumerate(cols):
        out[:, i] = arr[col]
    # Optionally flatten
    if len(cols) == 1:
        return out.reshape(
            -1,
        )
    return out
