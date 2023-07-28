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
from os.path import isfile

import numpy
from h5py import File

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
    x = dist * numpy.cos(dec) * numpy.cos(ra)
    y = dist * numpy.cos(dec) * numpy.sin(ra)
    z = dist * numpy.sin(dec)
    return numpy.vstack([x, y, z]).T


def real2redshift(pos, vel, origin, box, in_box_units, periodic_wrap=True,
                  make_copy=True):
    r"""
    Convert real-space position to redshift space position.

    Parameters
    ----------
    pos : 2-dimensional array `(nsamples, 3)`
        Real-space Cartesian position components.
    vel : 2-dimensional array `(nsamples, 3)`
        Cartesian velocity components.
    origin : 1-dimensional array `(3,)`
        Origin of the coordinate system in the `pos` reference frame.
    box : py:class:`csiborg.read.CSiBORGBox`
        Box units.
    in_box_units: bool
        Whether `pos` and `vel` are in box units. If not, position is assumed
        to be in :math:`\mathrm{Mpc}`, velocity in
        :math:`\mathrm{km} \mathrm{s}^{-1}` and math:`h=0.705`, or otherwise
        matching the box.
    periodic_wrap : bool, optional
        Whether to wrap around the box, particles may be outside the default
        bounds once RSD is applied.
    make_copy : bool, optional
        Whether to make a copy of `pos` before modifying it.

    Returns
    -------
    pos : 2-dimensional array `(nsamples, 3)`
        Redshift-space Cartesian position components, with an observer assumed
        at the `origin`.
    """
    a = box._aexp
    H0 = box.box_H0 if in_box_units else box.H0

    if make_copy:
        pos = numpy.copy(pos)
    for i in range(3):
        pos[:, i] -= origin[i]

    norm2 = numpy.sum(pos**2, axis=1)
    dot = numpy.einsum("ij,ij->i", pos, vel)
    pos *= (1 + a / H0 * dot / norm2).reshape(-1, 1)

    for i in range(3):
        pos[:, i] += origin[i]

    if periodic_wrap:
        boxsize = 1. if in_box_units else box.box2mpc(1.)
        # Wrap around the box: x > 1 -> x - 1, x < 0 -> x + 1
        pos[pos > boxsize] -= boxsize
        pos[pos < 0] += boxsize
    return pos


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
        Column names and dtypes. Each tuple must be written as `(name, dtype)`.

    Returns
    -------
    out : structured array
        Initialized structured array.
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

    Parameters
    ----------
    arr : structured array
        Structured array to add columns to.
    X : (list of) 1-dimensional array(s)
        Columns to be added.
    cols : str or list of str
        Column names to be added.

    Returns
    -------
    out : structured array
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

    Parameters
    ----------
    arr : structured array
        Structured array to remove columns from.
    cols : str or list of str
        Column names to be removed.

    Returns
    -------
    out : structured array
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
    Flip values in columns `col1` and `col2`. `arr` is modified in place.

    Parameters
    ----------
    arr : structured array
        Array whose columns are to be flipped.
    col1 : str
        First column name.
    col2 : str
        Second column name.
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
