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
"""
Halo profiles functions and posteriors.
"""


import numpy
from scipy.optimize import minimize_scalar
from scipy.stats import uniform
from .halo import Clump


class NFWProfile:
    r"""
    The Navarro-Frenk-White (NFW) density profile defined as

    .. math::
        \rho(r) = \frac{\rho_0}{x(1 + x)^2}

    where :math:`x = r / R_s` with free parameters :math:`R_s, \rho_0`.

    Parameters
    ----------
    Rs : float
        Scale radius :math:`R_s`.
    rho0 : float
        NFW density parameter :math:`\rho_0`.
    """

    def __init__(self):
        pass

    @staticmethod
    def profile(r, Rs, rho0):
        r"""
        Halo profile evaluated at :math:`r`.

        Parameters
        ----------
        r : float or 1-dimensional array
            Radial distance :math:`r`.
        Rs : float
            Scale radius :math:`R_s`.
        rho0 : float
            NFW density parameter :math:`\rho_0`.

        Returns
        -------
        density : float or 1-dimensional array
            Density of the NFW profile at :math:`r`.
        """
        x = r / Rs
        return rho0 / (x * (1 + x)**2)

    @staticmethod
    def logprofile(r, Rs, rho0):
        r"""
        Natural logarithm of the halo profile evaluated at :math:`r`.

        Parameters
        ----------
        r : float or 1-dimensional array
            Radial distance :math:`r`.
        Rs : float
            Scale radius :math:`R_s`.
        rho0 : float
            NFW density parameter :math:`\rho_0`.

        Returns
        -------
        logdensity : float or 1-dimensional array
            Logarithmic density of the NFW profile at :math:`r`.
        """
        x = r / Rs
        return numpy.log(rho0) - numpy.log(x) - 2 * numpy.log(1 + x)

    @staticmethod
    def enclosed_mass(r, Rs, rho0):
        r"""
        Enclosed mass  of a NFW profile in radius :math:`r`.

        Parameters
        ----------
        r : float or 1-dimensional array
            Radial distance :math:`r`.
        Rs : float
            Scale radius :math:`R_s`.
        rho0 : float
            NFW density parameter :math:`\rho_0`.

        Returns
        -------
        M : float or 1-dimensional array
            The enclosed mass.
        """
        x = r / Rs
        out = numpy.log(1 + x) - x / (1 + x)
        return 4 * numpy.pi * rho0 * Rs**3 * out

    def bounded_enclosed_mass(self, rmin, rmax, Rs, rho0):
        """
        Calculate the enclosed mass between :math:`r_min <= r <= r_max`.

        Parameters
        ----------
        rmin : float
            The minimum radius.
        rmax : float
            The maximum radius.
        Rs : float
            Scale radius :math:`R_s`.
        rho0 : float
            NFW density parameter :math:`\rho_0`.

        Returns
        -------
        M : float
            The enclosed mass within the radial range.
        """
        return (self.enclosed_mass(rmax, Rs, rho0)
                - self.enclosed_mass(rmin, Rs, rho0))

    def pdf(self, r, Rs, rmin, rmax):
        r"""
        The radial probability density function of the NFW profile calculated
        as

        .. math::
            \frac{4\pi r^2 \rho(r)} {M(r_\min, r_\max)}

        where :math:`M(r_\min, r_\max)` is the enclosed mass between
        :math:`r_\min` and :math:`r_\max'. Note that the dependance on
        :math:`\rho_0` is cancelled.

        Parameters
        ----------
        r : float or 1-dimensional array
            Radial distance :math:`r`.
        Rs : float
            Scale radius :math:`R_s`.
        rmin : float
            The minimum radius.
        rmax : float
            The maximum radius.

        Returns
        -------
        pdf : float or 1-dimensional array
            Probability density of the NFW profile at :math:`r`.
        """

        norm = self.bounded_enclosed_mass(rmin, rmax, Rs, 1)
        return 4 * numpy.pi * r**2 * self.profile(r, Rs, 1) / norm

    def rvs(self, rmin, rmax, Rs, N=1):
        """
        Generate random samples from the NFW profile via rejection sampling.

        Parameters
        ----------
        rmin : float
            The minimum radius.
        rmax : float
            The maximum radius.
        Rs : float
            Scale radius :math:`R_s`.
        N : int, optional
            Number of samples to generate. By default 1.

        Returns
        -------
        samples : float or 1-dimensional array
            Samples following the NFW profile.
        """
        gen = uniform(rmin, rmax-rmin)
        samples = numpy.full(N, numpy.nan)
        for i in range(N):
            while True:
                r = gen.rvs()
                if self.pdf(r, Rs, rmin, rmax) > numpy.random.rand():
                    samples[i] = r
                    break

        if N == 1:
            return samples[0]
        return samples


class NFWPosterior(NFWProfile):
    r"""
    Posterior of for fitting the NFW profile in the range specified by the
    closest and further particle. The likelihood is calculated as

    .. math::
        \frac{4\pi r^2 \rho(r)} {M(r_\min, r_\max)} \frac{m}{M / N}

    where :math:`M(r_\min, r_\max)` is the enclosed mass between the closest
    and further particle as expected from a NFW profile, :math:`m` is the
    particle mass, :math:`M` is the sum of the particle masses and :math:`N`
    is the number of particles.

    Paramaters
    ----------
    clump : `Clump`
        Clump object containing the particles and clump information.
    """
    _clump = None
    _binsguess = 10

    def __init__(self, clump):
        # Initialise the NFW profile
        super().__init__()
        self.clump = clump

    @property
    def clump(self):
        """
        Clump object.

        Returns
        -------
        clump : `Clump`
            The clump object.
        """
        return self._clump

    @clump.setter
    def clump(self, clump):
        """Sets `clump` and precalculates useful things."""
        if not isinstance(clump, Clump):
            raise TypeError(
                "`clump` must be :py:class:`csiborgtools.fits.Clump` type. "
                "Currently `{}`".format(type(clump)))
        self._clump = clump
        rmin = self.clump.rmin
        if rmin > self.clump.rmin:
            self._logrmin = numpy.log10(self.clump.rmin)
        else:
            r = self.clump.r
            self._logrmin = numpy.log10(numpy.min(r[r > 0]))
        self._logrmax = numpy.log(self.clump.rmax)
        self._logprior_volume = numpy.log(self._logrmax - self._logrmin)
        # Precalculate useful things
        self._logMtot = numpy.log(numpy.sum(self.clump.m))
        N = self.clump.Npart
        gamma = 4 * numpy.pi * self.clump.r**2 * self.clump.m * N
        self._ll0 = numpy.sum(numpy.log(gamma)) - N * self._logMtot

    def rho0_from_logRs(self, logRs):
        r"""
        Obtain :math:`\rho_0` of the NFW profile from the integral constraint
        on total mass. Calculated as the ratio between the total particle mass
        and the enclosed NFW profile mass.

        Parameters
        ----------
        logRs : float
            Logarithmic scale factor in units matching the coordinates.

        Returns
        -------
        rho0: float
            The NFW density parameter.
        """
        Mtot = numpy.exp(self._logMtot)
        Mnfw_norm = self.bounded_enclosed_mass(self.clump.rmin,
                                               self.clump.rmax, 10**logRs, 1)
        return Mtot / Mnfw_norm

    def logprior(self, logRs):
        r"""
        Logarithmic uniform prior on :math:`\log R_{\rm s}`.

        Parameters
        ----------
        logRs : float
            Logarithmic scale factor in units matching the coordinates.

        Returns
        -------
        ll : float
            The logarithmic prior.
        """
        if not self._logrmin < logRs < self._logrmax:
            return - numpy.infty
        return - self._logprior_volume

    def loglikelihood(self, logRs):
        """
        Logarithmic likelihood.

        Parameters
        ----------
        logRs : float
            Logarithmic scale factor in units matching the coordinates.

        Returns
        -------
        ll : float
            The logarithmic likelihood.
        """
        Rs = 10**logRs
        # Expected enclosed mass from a NFW
        Mnfw = self.bounded_enclosed_mass(self.clump.rmin, self.clump.rmax,
                                          Rs, 1)
        ll = self._ll0 + numpy.sum(self.logprofile(self.clump.r, Rs, 1))
        return ll - self.clump.Npart * numpy.log(Mnfw)

    @property
    def initlogRs(self):
        r"""
        The most often occuring value of :math:`r` used as initial guess of
        :math:`R_{\rm s}` since :math:`r^2 \rho(r)` peaks at
        :math:`r = R_{\rm s}`.

        Returns
        -------
        initlogRs : float
            The initial guess of :math:`\log R_{\rm s}`.
        """
        bins = numpy.linspace(self.clump.rmin, self.clump.rmax,
                              self._binsguess)
        counts, edges = numpy.histogram(self.clump.r, bins)
        return numpy.log(edges[numpy.argmax(counts)])

    def __call__(self, logRs):
        """
        Logarithmic posterior. Sum of the logarithmic prior and likelihood.

        Parameters
        ----------
        logRs : float
            Logarithmic scale factor in units matching the coordinates.

        Returns
        -------
        lpost : float
            The logarithmic posterior.
        """
        lp = self.logprior(logRs)
        if not numpy.isfinite(lp):
            return - numpy.infty
        return self.loglikelihood(logRs) + lp

    def maxpost_logRs(self):
        r"""
        Maximum a-posterio estimate of the scale radius :math:`\log R_{\rm s}`.
        Returns the scale radius if the fit converged, otherwise `numpy.nan`.

        Returns
        -------
        res : float
            The scale radius.
        """
        # Loss function to optimize
        def loss(logRs):
            return - self(logRs)
        res = minimize_scalar(loss, bounds=(self._logrmin, self._logrmax),
                              method='bounded')
        return res.x if res.success else numpy.nan
