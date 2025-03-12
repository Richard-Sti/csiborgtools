# Copyright (C) 2024 Richard Stiskalek
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
"""Various cosmography functions for converting between distance indicators."""
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from interpax import Interpolator1D
from jax import numpy as jnp
from scipy.interpolate import CubicSpline

H0 = 100  # km / s / Mpc


class ComovingDistance2Redshift:
    """
    Class to build an interpolator and convert comoving distance in `Mpc / h`
    to redshift.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=0, zmax_interp=0.5,
                 npoints_interp=250):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value

        self._f = Interpolator1D(r_grid, z_grid, extrap=False)

    def __call__(self, r):
        return self._f(r)


class ComovingDistance2Distmod:
    """
    Class to build an interpolator to convert comoving distance in `Mpc / h`
    to distance modulus.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-6, zmax_interp=0.5,
                 npoints_interp=500):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        z_grid = np.logspace(np.log10(zmin_interp), np.log10(zmax_interp),
                             npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value
        mu_grid = cosmo.distmod(z_grid).value

        self._f = Interpolator1D(jnp.log(r_grid), mu_grid, extrap=False)

    def __call__(self, r):
        return self._f(jnp.log(r))


class Distmod2Distance:
    """
    Class to build an interpolator to convert distance modulus to comoving
    distance in `Mpc / h`.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-6, zmax_interp=0.5,
                 npoints_interp=250):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value
        mu_grid = cosmo.distmod(z_grid).value

        self._f = Interpolator1D(mu_grid, jnp.log(r_grid), extrap=False)

    def __call__(self, r, return_log=False):
        if return_log:
            return self._f(r)

        return jnp.exp(self._f(r))


class Distmod2Redshift:
    """
    Class to build an interpolator to convert distance modulus to comoving
    distance in `Mpc / h`.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-6, zmax_interp=0.5,
                 npoints_interp=250):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        mu_grid = cosmo.distmod(z_grid).value

        self._f = Interpolator1D(mu_grid, jnp.log(z_grid), extrap=False)

    def __call__(self, r, return_log=False):
        if return_log:
            return self._f(r)

        return jnp.exp(self._f(r))


###############################################################################
#                           Various gradients                                 #
###############################################################################


class LogGrad_Distmod2ComovingDistance:
    """
    Class to build an interpolator to compute the log gradient of the comoving
    distance in `Mpc / h` with respect to distance modulus.

    The function is: `log (dr / dmu) | mu`.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-6, zmax_interp=0.5,
                 npoints_interp=500):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        z_grid = np.logspace(np.log10(zmin_interp), np.log10(zmax_interp),
                             npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value
        mu_grid = cosmo.distmod(z_grid).value

        spline = CubicSpline(mu_grid, r_grid, extrapolate=False)
        drdmu = spline.derivative()(mu_grid)

        self._f = Interpolator1D(mu_grid, jnp.log(drdmu), extrap=False)

    def __call__(self, mu):
        return self._f(mu)


class LogGrad_ComovingDistance2Distmod:
    """
    Class to build an interpolator to compute the log gradient of the distance
    modulus with respect to comoving distance in `Mpc / h`.

    The function is: `log (dmu / dr) | r`.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-6, zmax_interp=0.5,
                 npoints_interp=1000):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        z_grid = np.logspace(np.log10(zmin_interp), np.log10(zmax_interp),
                             npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value
        mu_grid = cosmo.distmod(z_grid).value

        spline = CubicSpline(r_grid, mu_grid, extrapolate=False)
        dmudr = spline.derivative()(r_grid)

        self._f = Interpolator1D(r_grid, jnp.log(dmudr), extrap=False)

    def __call__(self, mu):
        return self._f(mu)


class Grad_Redshift2ComovingDistance:
    """
    Class to build an interpolator to compute the gradient of the comoving
    distance in `Mpc / h` with respect to redshift.

    Parameters
    ----------
    Om0 : float
        Matter density parameter.
    zmin_interp, zmax_interp : float
        Minimum and maximum redshift for the interpolation grid.
    npoints_interp : int
        Number of points in the interpolation grid.
    """
    def __init__(self, Om0=0.3, zmin_interp=1e-6, zmax_interp=0.5,
                 npoints_interp=500):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
        r_grid = cosmo.comoving_distance(z_grid).value

        spline = CubicSpline(z_grid, r_grid, extrapolate=False)
        drdmu = spline.derivative()(z_grid)

        self._f = Interpolator1D(z_grid, drdmu, extrap=False)

    def __call__(self, mu):
        return self._f(mu)
