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
Tools for interpolating 3D fields at arbitrary positions.
"""
import MAS_library as MASL
import numpy
from tqdm import trange

from ..read.utils import radec_to_cartesian
from .utils import force_single_precision


def evaluate_cartesian(*fields, pos):
    """
    Evaluate a scalar field at Cartesian coordinates using CIC
    interpolation.

    Parameters
    ----------
    field : (list of) 3-dimensional array of shape `(grid, grid, grid)`
        Fields to be interpolated.
    pos : 2-dimensional array of shape `(n_samples, 3)`
        Positions to evaluate the density field. Assumed to be in box
        units.

    Returns
    -------
    interp_fields : (list of) 1-dimensional array of shape `(n_samples,).
    """
    boxsize = 1.
    pos = force_single_precision(pos, "pos")

    nsamples = pos.shape[0]
    interp_fields = [numpy.full(nsamples, numpy.nan, dtype=numpy.float32)
                     for __ in range(len(fields))]
    for i, field in enumerate(fields):
        MASL.CIC_interp(field, boxsize, pos, interp_fields[i])

    if len(fields) == 1:
        return interp_fields[0]
    return interp_fields


def evaluate_sky(*fields, pos, box, isdeg=True):
    """
    Evaluate the scalar fields at given distance, right ascension and
    declination. Assumes an observed in the centre of the box, with
    distance being in :math:`Mpc`. Uses CIC interpolation.

    Parameters
    ----------
    fields : (list of) 3-dimensional array of shape `(grid, grid, grid)`
        Field to be interpolated.
    pos : 2-dimensional array of shape `(n_samples, 3)`
        Spherical coordinates to evaluate the field. Columns are distance,
        right ascension, declination, respectively.
    box : :py:class:`csiborgtools.read.BoxUnits`
        The simulation box information and transformations.
    isdeg : bool, optional
        Whether `ra` and `dec` are in degres. By default `True`.

    Returns
    -------
    interp_fields : (list of) 1-dimensional array of shape `(n_samples,).
    """
    pos = force_single_precision(pos, "pos")
    # We first calculate convert the distance to box coordinates and then
    # convert to Cartesian coordinates.
    X = numpy.copy(pos)
    X[:, 0] = box.mpc2box(X[:, 0])
    X = radec_to_cartesian(pos, isdeg)
    # Then we move the origin to match the box coordinates
    X -= 0.5
    return evaluate_cartesian(*fields, pos=X)


def make_sky(field, angpos, dist, verbose=True):
    r"""
    Make a sky map of a scalar field. The observer is in the centre of the
    box the field is evaluated along directions `angpos`. Along each
    direction, the field is evaluated distances `dist_marg` and summed.
    Uses CIC interpolation.

    Parameters
    ----------
    field : 3-dimensional array of shape `(grid, grid, grid)`
        Field to be interpolated
    angpos : 2-dimensional arrays of shape `(ndir, 2)`
        Directions to evaluate the field. Assumed to be RA
        :math:`\in [0, 360]` and dec :math:`\in [-90, 90]` degrees,
        respectively.
    dist : 1-dimensional array
        Radial distances to evaluate the field.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    interp_field : 1-dimensional array of shape `(n_pos, )`.
    """
    assert angpos.ndim == 2 and dist.ndim == 1
    # We loop over the angular directions, at each step evaluating a vector
    # of distances. We pre-allocate arrays for speed.
    dir_loop = numpy.full((dist.size, 3), numpy.nan, dtype=numpy.float32)
    ndir = angpos.shape[0]
    out = numpy.zeros(ndir, numpy.nan, dtype=numpy.float32)
    for i in trange(ndir) if verbose else range(ndir):
        dir_loop[1, :] = angpos[i, 0]
        dir_loop[2, :] = angpos[i, 1]
        out[i] = numpy.sum(evaluate_sky(field, dir_loop, isdeg=True))
    return out
