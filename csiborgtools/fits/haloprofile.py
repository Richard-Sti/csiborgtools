# Copyright (C) 2022 Richard Stiskalek, Deaglan Bartlett
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
"""Halo profiles functions and posteriors."""
import numpy
from scipy.optimize import minimize_scalar
from scipy.stats import uniform

from .halo import Clump


class NFWProfile:
    r"""
    The Navarro-Frenk-White (NFW) density profile.

    .. math::
        \rho(r) = \frac{\rho_0}{x(1 + x)^2},

    :math:`x = r / R_s` and its free paramaters are :math:`R_s, \rho_0`: scale
    radius and NFW density parameter.
    """

    @staticmethod
    def profile(r, Rs, rho0):
        """
        Evaluate the halo profile at `r`.

        Parameters
        ----------
        r : 1-dimensional array
            Radial distance.
        Rs : float
            Scale radius.
        rho0 : float
            NFW density parameter.

        Returns
        -------
        density : 1-dimensional array
        """
        x = r / Rs
        return rho0 / (x * (1 + x) ** 2)

    @staticmethod
    def _logprofile(r, Rs, rho0):
        """Natural logarithm of `NFWPprofile.profile(...)`."""
        x = r / Rs
        return numpy.log(rho0) - numpy.log(x) - 2 * numpy.log(1 + x)

    @staticmethod
    def mass(r, Rs, rho0):
        r"""
        Calculate the enclosed mass of a NFW profile in radius `r`.

        Parameters
        ----------
        r : 1-dimensional array
            Radial distance.
        Rs : float
            Scale radius.
        rho0 : float
            NFW density parameter.

        Returns
        -------
        M : 1-dimensional array
            The enclosed mass.
        """
        x = r / Rs
        out = numpy.log(1 + x) - x / (1 + x)
        return 4 * numpy.pi * rho0 * Rs**3 * out

    def bounded_mass(self, rmin, rmax, Rs, rho0):
        r"""
        Calculate the enclosed mass between `rmin` and `rmax`.

        Parameters
        ----------
        rmin : float
            Minimum radius.
        rmax : float
            Maximum radius.
        Rs : float
            Scale radius :math:`R_s`.
        rho0 : float
            NFW density parameter.

        Returns
        -------
        M : float
            Enclosed mass within the radial range.
        """
        return self.mass(rmax, Rs, rho0) - self.mass(rmin, Rs, rho0)

    def pdf(self, r, Rs, rmin, rmax):
        r"""
        Calculate the radial PDF of the NFW profile, defined below.

        .. math::
            \frac{4\pi r^2 \rho(r)} {M(r_\min, r_\max)},

        where :math:`M(r_\min, r_\max)` is the enclosed mass between
        :math:`r_\min` and :math:`r_\max'. Note that the dependance on
        :math:`\rho_0` is cancelled and must be accounted for in the
        normalisation term to match the total mass.

        Parameters
        ----------
        r : 1-dimensional array
            Radial distance.
        Rs : float
            Scale radius.
        rmin : float
            Minimum radius to evaluate the PDF (denominator term).
        rmax : float
            Maximum radius to evaluate the PDF (denominator term).

        Returns
        -------
        pdf : 1-dimensional array
        """
        norm = self.bounded_enclosed_mass(rmin, rmax, Rs, 1)
        return 4 * numpy.pi * r**2 * self.profile(r, Rs, 1) / norm

    def rvs(self, rmin, rmax, Rs, size=1):
        """
        Generate random samples from the NFW profile via rejection sampling.

        Parameters
        ----------
        rmin : float
            Minimum radius.
        rmax : float
            Maximum radius.
        Rs : float
            Scale radius.
        size : int, optional
            Number of samples to generate. By default 1.

        Returns
        -------
        samples : float or 1-dimensional array
            Samples following the NFW profile.
        """
        gen = uniform(rmin, rmax - rmin)
        samples = numpy.full(size, numpy.nan)
        for i in range(size):
            while True:
                r = gen.rvs()
                if self.pdf(r, Rs, rmin, rmax) > numpy.random.rand():
                    samples[i] = r
                    break

        if size == 1:
            return samples[0]
        return samples


class NFWPosterior(NFWProfile):
    r"""
    Posterior for fitting the NFW profile in the range specified by the
    closest particle and the :math:`r_{200c}` radius, calculated as below.

    .. math::
        \frac{4\pi r^2 \rho(r)} {M(r_{\min} r_{200c})} \frac{m}{M / N},

    where :math:`M(r_{\min} r_{200c}))` is the NFW enclosed mass between the
    closest particle and the :math:`r_{200c}` radius, :math:`m` is the particle
    mass, :math:`M` is the sum of the particle masses and :math:`N` is the
    number of particles. Calculated only using particles within the
    above-mentioned range.

    Paramaters
    ----------
    clump : `Clump`
        Clump object containing the particles and clump information.
    """

    @property
    def clump(self):
        """
        Clump object containig all particles, i.e. ones beyond :math:`R_{200c}`
        as well.

        Returns
        -------
        clump : `Clump`
        """
        return self._clump

    @clump.setter
    def clump(self, clump):
        assert isinstance(clump, Clump)
        self._clump = clump

    def rho0_from_Rs(self, Rs, rmin, rmax, mass):
        r"""
        Obtain :math:`\rho_0` of the NFW profile from the integral constraint
        on total mass. Calculated as the ratio between the total particle mass
        and the enclosed NFW profile mass.

        Parameters
        ----------
        Rs : float
            Logarithmic scale factor in units matching the coordinates.
        rmin : float
            Minimum radial distance of particles used to fit the profile.
        rmax : float
            Maximum radial distance of particles used to fit the profile.
        mass : float
            Mass enclosed within the radius used to fit the NFW profile.

        Returns
        -------
        rho0: float
        """
        return mass / self.bounded_mass(rmin, rmax, Rs, 1)

    def initlogRs(self, r, rmin, rmax, binsguess=10):
        r"""
        Calculate the most often occuring value of :math:`r` used as initial
        guess of :math:`R_{\rm s}` since :math:`r^2 \rho(r)` peaks at
        :math:`r = R_{\rm s}`.

        Parameters
        ----------
        r : 1-dimensional array
            Radial distance of particles used to fit the profile.
        rmin : float
            Minimum radial distance of particles used to fit the profile.
        rmax : float
            Maximum radial distance of particles used to fit the profile.
        binsguess : int
            Number of bins to initially guess :math:`R_{\rm s}`.

        Returns
        -------
        initlogRs : float
        """
        bins = numpy.linspace(rmin, rmax, binsguess)
        counts, edges = numpy.histogram(r, bins)
        return numpy.log10(edges[numpy.argmax(counts)])

    def logprior(self, logRs, rmin, rmax):
        r"""
        Logarithmic uniform prior on :math:`\log R_{\rm s}`. Unnormalised but
        that does not matter.

        Parameters
        ----------
        logRs : float
            Logarithmic scale factor.
        rmin : float
            Minimum radial distance of particles used to fit the profile.
        rmax : float
            Maximum radial distance of particles used to fit the profile.

        Returns
        -------
        lp : float
        """
        if not rmin < 10**logRs < rmax:
            return -numpy.infty
        return 0.0

    def loglikelihood(self, logRs, r, rmin, rmax, npart):
        """
        Logarithmic likelihood.

        Parameters
        ----------
        r : 1-dimensional array
            Radial distance of particles used to fit the profile.
        logRs : float
            Logarithmic scale factor in units matching the coordinates.
        rmin : float
            Minimum radial distance of particles used to fit the profile.
        rmax : float
            Maximum radial distance of particles used to fit the profile.
        npart : int
            Number of particles used to fit the profile.

        Returns
        -------
        ll : float
        """
        Rs = 10**logRs
        mnfw = self.bounded_mass(rmin, rmax, Rs, 1)
        return numpy.sum(self._logprofile(r, Rs, 1)) - npart * numpy.log(mnfw)

    def __call__(self, logRs, r, rmin, rmax, npart):
        """
        Logarithmic posterior. Sum of the logarithmic prior and likelihood.

        Parameters
        ----------
        logRs : float
            Logarithmic scale factor in units matching the coordinates.
        r : 1-dimensional array
            Radial distance of particles used to fit the profile.
        rmin : float
            Minimum radial distance of particles used to fit the profile.
        rmax : float
            Maximum radial distance of particles used to fit the profile.
        npart : int
            Number of particles used to fit the profile.

        Returns
        -------
        lpost : float
        """
        lp = self.logprior(logRs, rmin, rmax)
        if not numpy.isfinite(lp):
            return -numpy.infty
        return self.loglikelihood(logRs, r, rmin, rmax, npart) + lp

    def fit(self, clump, eps=1e-4):
        r"""
        Fit the NFW profile. If the fit is not converged returns NaNs.

        Checks whether :math:`log r_{\rm max} / R_{\rm s} > \epsilon`,
        to ensure that the scale radius is not too close to the boundary which
        occurs if the fit fails.

        Parameters
        ----------
        clump : :py:class:`csiborgtools.fits.Clump`
            Clump being fitted.
        eps : float
            Tolerance to ensure we are sufficiently far from math:`R_{200c}`.

        Returns
        -------
        Rs: float
            Best fit scale radius.
        rho0: float
            Best fit NFW central density.
        """
        assert isinstance(clump, Clump)
        r = clump.r()
        rmin = numpy.min(r[r > 0])  # First particle that is not at r = 0
        rmax, mtot = clump.spherical_overdensity_mass(200)
        mask = (rmin <= r) & (r <= rmax)
        npart = numpy.sum(mask)
        r = r[mask]

        def loss(logRs):
            return -self(logRs, r, rmin, rmax, npart)

        # Define optimisation boundaries. Check whether they are finite and
        # that rmax > rmin. If not, then return NaN.
        bounds = (numpy.log10(rmin), numpy.log10(rmax))
        if not (numpy.all(numpy.isfinite(bounds)) and bounds[0] < bounds[1]):
            return numpy.nan, numpy.nan

        res = minimize_scalar(loss, bounds=bounds, method="bounded")
        # Check whether the fit converged to radius sufficienly far from `rmax`
        # and that its a success. Otherwise return NaNs.
        if numpy.log10(rmax) - res.x < eps:
            res.success = False
        if not res.success:
            return numpy.nan, numpy.nan
        # Additionally we also wish to calculate the central density from the
        # mass (integral) constraint.
        rho0 = self.rho0_from_Rs(10**res.x, rmin, rmax, mtot)
        return 10**res.x, rho0
