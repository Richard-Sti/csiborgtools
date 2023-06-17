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

from ..read.utils import radec_to_cartesian, real2redshift
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
    X = radec_to_cartesian(pos, isdeg)
    # Then we move the origin to match the box coordinates
    X += 0.5
    return evaluate_cartesian(*fields, pos=X)


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
    Divide two fields where the second one is not zero. If the second field
    is zero, the first one is left unchanged. Operates in-place.

    Parameters
    ----------
    field0 : 3-dimensional array of shape `(grid, grid, grid)`
        Field to be divided.
    field1 : 3-dimensional array of shape `(grid, grid, grid)`
        Field to divide by.

    Returns
    -------
    field0 : 3-dimensional array of shape `(grid, grid, grid)`
        Field divided by the second one.
    """
    assert field0.shape == field1.shape

    imax, jmax, kmax = field0.shape
    for i in range(imax):
        for j in range(jmax):
            for k in range(kmax):
                if field1[i, j, k] != 0:
                    field0[i, j, k] /= field1[i, j, k]


def field2rsp(*fields, parts, box, nbatch=30, flip_partsxz=True, init_value=0.,
              verbose=True):
    """
    Forward model real space scalar fields to redshift space. Attaches field
    values to particles, those are then moved to redshift space and from their
    positions reconstructs back the field on a grid by NGP interpolation.

    Parameters
    ----------
    fields : (list of) 3-dimensional array of shape `(grid, grid, grid)`
        Real space fields to be evolved to redshift space.
    parts : 2-dimensional array of shape `(n_parts, 6)`
        Particle positions and velocities in real space. Must be organised as
        `x, y, z, vx, vy, vz`.
    box : :py:class:`csiborgtools.read.CSiBORGBox`
        The simulation box information and transformations.
    nbatch : int, optional
        Number of batches to use when moving particles to redshift space.
        Particles are assumed to be lazily loaded to memory.
    flip_partsxz : bool, optional
        Whether to flip the x and z coordinates of the particles. This is
        because of a RAMSES bug.
    init_value : float, optional
        Initial value of the RSP field on the grid.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    rsp_fields : (list of) 3-dimensional array of shape `(grid, grid, grid)`
    """
    nfields = len(fields)
    # Check that all fields have the same shape.
    if nfields > 1:
        assert all(fields[0].shape == fields[i].shape
                   for i in range(1, nfields))

    rsp_fields = [numpy.full(field.shape, init_value, dtype=numpy.float32)
                  for field in fields]
    cellcounts = numpy.zeros(rsp_fields[0].shape, dtype=numpy.float32)

    nparts = parts.shape[0]
    batch_size = nparts // nbatch
    start = 0
    for __ in trange(nbatch + 1) if verbose else range(nbatch + 1):
        # We first load the batch of particles into memory and flip x and z.
        end = min(start + batch_size, nparts)
        pos = parts[start:end]
        pos, vel = pos[:, :3], pos[:, 3:6]
        if flip_partsxz:
            pos[:, [0, 2]] = pos[:, [2, 0]]
            vel[:, [0, 2]] = vel[:, [2, 0]]
        # Then move the particles to redshift space.
        rsp_pos = real2redshift(pos, vel, [0.5, 0.5, 0.5], box,
                                in_box_units=True, periodic_wrap=True,
                                make_copy=True)
        # ... and count the number of particles in each grid cell.
        MASL.MA(rsp_pos, cellcounts, 1., "NGP")

        # Now finally we evaluate the field at the particle positions in real
        # space and then assign the values to the grid in redshift space.
        for i in range(nfields):
            values = evaluate_cartesian(fields[i], pos=pos)
            MASL.MA(rsp_pos, rsp_fields[i], 1., "NGP", W=values)
        if end == nparts:
            break
        start = end

    # We divide by the number of particles in each cell.
    for i in range(len(fields)):
        divide_nonzero(rsp_fields[i], cellcounts)

    if len(fields) == 1:
        return rsp_fields[0]
    return rsp_fields


@jit(nopython=True)
def fill_outside(field, fill_value, rmax, boxsize):
    """
    Fill cells outside of a sphere of radius `rmax` with `fill_value`. Centered
    in the middle of the box.

    Parameters
    ----------
    field : 3-dimensional array of shape `(grid, grid, grid)`
        Field to be filled.
    fill_value : float
        Value to fill the field with.
    rmax : float
        Radius outside of which to fill the field..
    boxsize : float
        Size of the box.

    Returns
    -------
    field : 3-dimensional array of shape `(grid, grid, grid)`
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
