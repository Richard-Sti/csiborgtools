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
from scipy.ndimage import gaussian_filter
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


def field2rsp(field, parts, box, nbatch=30, flip_partsxz=True, init_value=0.,
              verbose=True):
    """
    Forward model real space scalar field to redshift space. Attaches field
    values to particles, those are then moved to redshift space and from their
    positions reconstructs back the field on a regular grid by NGP
    interpolation. This by definition produces a discontinuous field.

    Parameters
    ----------
    field : 3-dimensional array of shape `(grid, grid, grid)`
        Real space field to be evolved to redshift space.
    parts_pos : 2-dimensional array of shape `(n_parts, 3)`
        Particle positions in real space.
    parts_vel : 2-dimensional array of shape `(n_parts, 3)`
        Particle velocities in real space.
    box : :py:class:`csiborgtools.read.CSiBORGBox`
        The simulation box information and transformations.
    nbatch : int, optional
        Number of batches to use when moving particles to redshift space.
        Particles are assumed to be lazily loaded to memory.
    flip_partsxz : bool, optional
        Whether to flip the x and z coordinates of the particles. This is
        because of a BORG bug.
    init_value : float, optional
        Initial value of the RSP field on the grid.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    rsp_fields : (list of) 3-dimensional array of shape `(grid, grid, grid)`
    """
    rsp_field = numpy.full(field.shape, init_value, dtype=numpy.float32)
    cellcounts = numpy.zeros(rsp_field.shape, dtype=numpy.float32)
    # We iterate over the fields and in the inner loop over the particles. This
    # is slower than iterating over the particles and in the inner loop over
    # the fields, but it is more memory efficient. Typically we will only have
    # one field.
    nparts = parts.shape[0]
    batch_size = nparts // nbatch
    start = 0
    for k in trange(nbatch + 1) if verbose else range(nbatch + 1):
        end = min(start + batch_size, nparts)
        pos = parts[start:end]
        pos, vel = pos[:, :3], pos[:, 3:6]
        if flip_partsxz:
            pos[:, [0, 2]] = pos[:, [2, 0]]
            vel[:, [0, 2]] = vel[:, [2, 0]]
        # Evaluate the field at the particle positions in real space.
        values = evaluate_cartesian(field, pos=pos)
        # Move particles to redshift space.
        rsp_pos = real2redshift(pos, vel, [0.5, 0.5, 0.5], box,
                                in_box_units=True, periodic_wrap=True,
                                make_copy=True)
        # Assign particles' values to the grid.
        MASL.MA(rsp_pos, rsp_field, 1., "NGP", W=values)
        # Count the number of particles in each grid cell.
        MASL.MA(rsp_pos, cellcounts, 1., "NGP")
        if end == nparts:
            break
        start = end

    # Finally divide by the number of particles in each cell and smooth.
    divide_nonzero(rsp_field, cellcounts)
    return rsp_field
