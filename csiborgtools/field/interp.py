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
import healpy
import MAS_library as MASL
import numpy
import smoothing_library as SL
from numba import jit
from tqdm import tqdm, trange

from ..utils import periodic_wrap_grid, radec_to_cartesian
from .utils import divide_nonzero, force_single_precision


###############################################################################
#                       Cartesian interpolation                               #
###############################################################################


def evaluate_cartesian(*fields, pos, smooth_scales=None, verbose=False):
    """
    Evaluate a scalar field(s) at Cartesian coordinates `pos`.

    Parameters
    ----------
    field : (list of) 3-dimensional array of shape `(grid, grid, grid)`
        Fields to be interpolated.
    pos : 2-dimensional array of shape `(n_samples, 3)`
        Query positions in box units.
    smooth_scales : (list of) float, optional
        Smoothing scales in box units. If `None`, no smoothing is performed.
    verbose : bool, optional
        Smoothing verbosity flag.

    Returns
    -------
    (list of) 1-dimensional array of shape `(n_samples, len(smooth_scales))`
    """
    pos = force_single_precision(pos)

    if isinstance(smooth_scales, (int, float)):
        smooth_scales = [smooth_scales]

    if smooth_scales is None:
        shape = (pos.shape[0],)
    else:
        shape = (pos.shape[0], len(smooth_scales))

    interp_fields = [numpy.full(shape, numpy.nan, dtype=numpy.float32)
                     for __ in range(len(fields))]

    for i, field in enumerate(fields):
        if smooth_scales is None:
            MASL.CIC_interp(field, 1., pos, interp_fields[i])
        else:
            desc = f"Smoothing and interpolating field {i + 1}/{len(fields)}"
            iterator = tqdm(smooth_scales, desc=desc, disable=not verbose)

            for j, scale in enumerate(iterator):
                if not scale > 0:
                    fsmooth = numpy.copy(field)
                else:
                    fsmooth = smoothen_field(field, scale, 1., make_copy=True)
                MASL.CIC_interp(fsmooth, 1., pos, interp_fields[i][:, j])

    if len(fields) == 1:
        return interp_fields[0]

    return interp_fields


def observer_peculiar_velocity(velocity_field, smooth_scales=None,
                               observer=None, verbose=True):
    """
    Calculate the peculiar velocity in the centre of the box.

    Parameters
    ----------
    velocity_field : 4-dimensional array of shape `(3, grid, grid, grid)`
        Velocity field in `km / s`.
    smooth_scales : (list of) float, optional
        Smoothing scales in box units. If `None`, no smoothing is performed.
    observer : 1-dimensional array of shape `(3,)`, optional
        Observer position in box units. If `None`, the observer is assumed to
        be in the centre of the box.
    verbose : bool, optional
        Smoothing verbosity flag.

    Returns
    -------
    vpec : 1-dimensional array of shape `(3,)` or `(len(smooth_scales), 3)`
    """
    if observer is None:
        pos = numpy.asanyarray([0.5, 0.5, 0.5]).reshape(1, 3)
    else:
        pos = numpy.asanyarray(observer).reshape(1, 3)

    vx, vy, vz = evaluate_cartesian(
        *velocity_field, pos=pos, smooth_scales=smooth_scales, verbose=verbose)

    # Reshape since we evaluated only one point
    vx = vx.reshape(-1, )
    vy = vy.reshape(-1, )
    vz = vz.reshape(-1, )

    if smooth_scales is None:
        return numpy.array([vx[0], vy[0], vz[0]])

    return numpy.vstack([vx, vy, vz]).T


###############################################################################
#                              Sky maps                                       #
###############################################################################


def evaluate_sky(*fields, pos, mpc2box, smooth_scales=None, verbose=False):
    """
    Evaluate a scalar field(s) at radial distance `Mpc / h`, right ascensions
    [0, 360) deg and declinations [-90, 90] deg.

    Parameters
    ----------
    fields : (list of) 3-dimensional array of shape `(grid, grid, grid)`
        Field to be interpolated.
    pos : 2-dimensional array of shape `(n_samples, 3)`
        Query spherical coordinates.
    mpc2box : float
        Conversion factor to multiply the radial distance by to get box units.
    smooth_scales : (list of) float, optional
        Smoothing scales in `Mpc / h`. If `None`, no smoothing is performed.
    verbose : bool, optional
        Smoothing verbosity flag.

    Returns
    -------
    (list of) 1-dimensional array of shape `(n_samples, len(smooth_scales))`
    """
    # Make a copy of the positions to avoid modifying the input.
    pos = numpy.copy(pos)

    pos = force_single_precision(pos)
    pos[:, 0] *= mpc2box

    cart_pos = radec_to_cartesian(pos) + 0.5

    if smooth_scales is not None:
        if isinstance(smooth_scales, (int, float)):
            smooth_scales = [smooth_scales]

        if isinstance(smooth_scales, list):
            smooth_scales = numpy.array(smooth_scales, dtype=numpy.float32)

        smooth_scales *= mpc2box

    return evaluate_cartesian(*fields, pos=cart_pos,
                              smooth_scales=smooth_scales, verbose=verbose)


def make_sky(field, angpos, dist, boxsize, verbose=True):
    r"""
    Make a sky map of a scalar field. The observer is in the centre of the
    box the field is evaluated along directions `angpos` (RA [0, 360) deg,
    dec [-90, 90] deg). Along each direction, the field is evaluated distances
    `dist` (`Mpc / h`) and summed. Uses CIC interpolation.

    Parameters
    ----------
    field : 3-dimensional array of shape `(grid, grid, grid)`
        Field to be interpolated
    angpos : 2-dimensional arrays of shape `(ndir, 2)`
        Directions to evaluate the field.
    dist : 1-dimensional array
        Uniformly spaced radial distances to evaluate the field in `Mpc / h`.
    boxsize : float
        Box size in `Mpc / h`.
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

    ndir = angpos.shape[0]
    out = numpy.full(ndir, numpy.nan, dtype=numpy.float32)
    for i in trange(ndir) if verbose else range(ndir):
        dir_loop[:, 0] = dist
        dir_loop[:, 1] = angpos[i, 0]
        dir_loop[:, 2] = angpos[i, 1]
        out[i] = numpy.sum(
            dist**2 * evaluate_sky(field, pos=dir_loop, mpc2box=1 / boxsize))

    out *= dx
    return out


def nside2radec(nside):
    """
    Generate RA [0, 360] deg. and declination [-90, 90] deg. for HEALPix pixel
    centres at a given nside.
    """
    pixs = numpy.arange(healpy.nside2npix(nside))
    theta, phi = healpy.pix2ang(nside, pixs)

    ra = 180 / numpy.pi * phi
    dec = 90 - 180 / numpy.pi * theta

    return numpy.vstack([ra, dec]).T


###############################################################################
#                     Real-to-redshift space field dragging                   #
###############################################################################


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
    box : :py:class:`csiborgtools.read.CSiBORG1Box`
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

###############################################################################
#                          Supplementary function                             #
###############################################################################


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


def smoothen_field(field, smooth_scale, boxsize, threads=1, make_copy=False):
    """
    Smooth a field with a Gaussian filter.
    """
    W_k = SL.FT_filter(boxsize, smooth_scale, field.shape[0], "Gaussian",
                       threads)

    if make_copy:
        field = numpy.copy(field)

    return SL.field_smoothing(field, W_k, threads)
