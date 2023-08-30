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
from numba import jit
from tqdm import trange

from .utils import force_single_precision
from ..utils import periodic_wrap_grid, radec_to_cartesian


def evaluate_cartesian(*fields, pos, interp="CIC"):
    """
    Evaluate a scalar field at Cartesian coordinates.

    Parameters
    ----------
    field : (list of) 3-dimensional array of shape `(grid, grid, grid)`
        Fields to be interpolated.
    pos : 2-dimensional array of shape `(n_samples, 3)`
        Positions to evaluate the density field. Assumed to be in box
        units.
    interp : str, optional
        Interpolation method. Can be either `CIC` or `NGP`.

    Returns
    -------
    interp_fields : (list of) 1-dimensional array of shape `(n_samples,).
    """
    assert interp in ["CIC", "NGP"]
    boxsize = 1.
    pos = force_single_precision(pos, "pos")

    nsamples = pos.shape[0]
    interp_fields = [numpy.full(nsamples, numpy.nan, dtype=numpy.float32)
                     for __ in range(len(fields))]

    if interp == "CIC":
        for i, field in enumerate(fields):
            MASL.CIC_interp(field, boxsize, pos, interp_fields[i])
    else:
        pos = numpy.floor(pos * fields[0].shape[0]).astype(numpy.int32)
        for i, field in enumerate(fields):
            for j in range(nsamples):
                interp_fields[i][j] = field[pos[j, 0], pos[j, 1], pos[j, 2]]

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
    box : :py:class:`csiborgtools.read.CSiBORGBox`
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
    pos[:, 0] = box.mpc2box(pos[:, 0])
    assert isdeg
    X = radec_to_cartesian(pos)
    # Then we move the origin to match the box coordinates
    X += 0.5
    return evaluate_cartesian(*fields, pos=X)


def observer_vobs(velocity_field):
    """
    Calculate the observer velocity from a velocity field. Assumes an observer
    in the centre of the box.

    Parameters
    ----------
    velocity_field : 4-dimensional array of shape `(3, grid, grid, grid)`

    Returns
    -------
    1-dimensional array of shape `(3,)`
    """
    pos = numpy.asanyarray([0.5, 0.5, 0.5]).reshape(1, 3)
    vobs = numpy.full(3, numpy.nan, dtype=numpy.float32)
    for i in range(3):
        vobs[i] = evaluate_cartesian(velocity_field[i, ...], pos=pos)[0]
    return vobs


def make_sky(field, angpos, dist, box, volume_weight=True, verbose=True):
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
        Uniformly spaced radial distances to evaluate the field.
    box : :py:class:`csiborgtools.read.CSiBORGBox`
        The simulation box information and transformations.
    volume_weight : bool, optional
        Whether to weight the field by the volume of the pixel, i.e. a
        :math:`r^2` correction.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    interp_field : 1-dimensional array of shape `(n_pos, )`.
    """
    dx = dist[1] - dist[0]
    assert numpy.allclose(dist[1:] - dist[:-1], dx)
    assert angpos.ndim == 2 and dist.ndim == 1
    # We loop over the angular directions, at each step evaluating a vector
    # of distances. We pre-allocate arrays for speed.
    dir_loop = numpy.full((dist.size, 3), numpy.nan, dtype=numpy.float32)
    boxdist = box.mpc2box(dist)
    ndir = angpos.shape[0]
    out = numpy.full(ndir, numpy.nan, dtype=numpy.float32)
    for i in trange(ndir) if verbose else range(ndir):
        dir_loop[:, 0] = dist
        dir_loop[:, 1] = angpos[i, 0]
        dir_loop[:, 2] = angpos[i, 1]
        if volume_weight:
            out[i] = numpy.sum(
                boxdist**2
                * evaluate_sky(field, pos=dir_loop, box=box, isdeg=True))
        else:
            out[i] = numpy.sum(
                evaluate_sky(field, pos=dir_loop, box=box, isdeg=True))
    out *= dx
    return out


@jit(nopython=True)
def divide_nonzero(field0, field1):
    """
    Perform in-place `field0 /= field1` but only where `field1 != 0`.
    """
    assert field0.shape == field1.shape, "Field shapes must match."

    imax, jmax, kmax = field0.shape
    for i in range(imax):
        for j in range(jmax):
            for k in range(kmax):
                if field1[i, j, k] != 0:
                    field0[i, j, k] /= field1[i, j, k]


@jit(nopython=True)
def make_gridpos(grid_size):
    """Make a regular grid of positions and distances from the center."""
    grid_pos = numpy.full((grid_size**3, 3), numpy.nan, dtype=numpy.float32)
    grid_dist = numpy.full(grid_size**3, numpy.nan, dtype=numpy.float32)

    n = 0
    for i in range(grid_size):
        px = (i - 0.5 * (grid_size - 1)) / grid_size
        px2 = px**2
        for j in range(grid_size):
            py = (j - 0.5 * (grid_size - 1)) / grid_size
            py2 = py**2
            for k in range(grid_size):
                pz = (k - 0.5 * (grid_size - 1)) / grid_size
                pz2 = pz**2

                grid_pos[n, 0] = px
                grid_pos[n, 1] = py
                grid_pos[n, 2] = pz

                grid_dist[n] = (px2 + py2 + pz2)**0.5

                n += 1

    return grid_pos, grid_dist


def field2rsp(field, radvel_field, box, MAS, init_value=0.):
    """
    Forward model a real space scalar field to redshift space.

    Parameters
    ----------
    field : 3-dimensional array of shape `(grid, grid, grid)`
        Real space field to be evolved to redshift space.
    radvel_field : 3-dimensional array of shape `(grid, grid, grid)`
        Radial velocity field in `km / s`. Expected to account for the observer
        velocity.
    box : :py:class:`csiborgtools.read.CSiBORGBox`
        The simulation box information and transformations.
    MAS : str
        Mass assignment. Must be one of `NGP`, `CIC`, `TSC` or `PCS`.
    init_value : float, optional
        Initial value of the RSP field on the grid.

    Returns
    -------
    3-dimensional array of shape `(grid, grid, grid)`
    """
    grid = field.shape[0]
    H0_inv = 1. / 100 / box.box2mpc(1.)

    # Calculate the regular grid positions and distances from the center.
    grid_pos, grid_dist = make_gridpos(grid)
    grid_dist = grid_dist.reshape(-1, 1)

    # Move the grid positions to redshift space.
    grid_pos *= (1 + H0_inv * radvel_field.reshape(-1, 1) / grid_dist)
    grid_pos += 0.5
    grid_pos = periodic_wrap_grid(grid_pos)

    rsp_field = numpy.full(field.shape, init_value, dtype=numpy.float32)
    cell_counts = numpy.zeros(rsp_field.shape, dtype=numpy.float32)

    # Interpolate the field to the grid positions.
    MASL.MA(grid_pos, rsp_field, 1., MAS, W=field.reshape(-1,))
    MASL.MA(grid_pos, cell_counts, 1., MAS)
    divide_nonzero(rsp_field, cell_counts)

    return rsp_field


@jit(nopython=True)
def fill_outside(field, fill_value, rmax, boxsize):
    """
    Fill cells outside of a sphere of radius `rmax` around the box centre with
    `fill_value`.
    """
    imax, jmax, kmax = field.shape
    assert imax == jmax == kmax
    N = imax

    # Squared radial distance from the center of the box in box units.
    rmax_box2 = (N * rmax / boxsize)**2

    for i in range(N):
        idist2 = (i - 0.5 * (N - 1))**2
        for j in range(N):
            jdist2 = (j - 0.5 * (N - 1))**2
            for k in range(N):
                kdist2 = (k - 0.5 * (N - 1))**2
                if idist2 + jdist2 + kdist2 > rmax_box2:
                    field[i, j, k] = fill_value
    return field
