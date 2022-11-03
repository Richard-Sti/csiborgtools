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
from jax import numpy as jnumpy
from jax import grad
from scipy.optimize import minimize_scalar
from scipy.stats import uniform
from .halo import Clump


def jax_switch(f1, f2, use_jax):
    """
    Return function `f1` if `use_jax` else return `f2`.

    Parameters
    ----------
    f1 : function
        The JAX function.
    f2 : function
        The non-JAX function.
    use_jax : bool
        Whether to use the JAX function.

    Returns
    -------
    chosen_func : function
        The chosen function.
    """
    return f1 if use_jax else f2


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
    def logprofile(r, Rs, rho0, use_jax=False):
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
        use_jax : bool, optional
            Whether to use `JAX` expressions. By default `False`.

        Returns
        -------
        logdensity : float or 1-dimensional array
            Logarithmic density of the NFW profile at :math:`r`.
        """
        log = jax_switch(jnumpy.log, numpy.log, use_jax)
        x = r / Rs
        return log(rho0) - log(x) - 2 * log(1 + x)

    @staticmethod
    def enclosed_mass(r, Rs, rho0, use_jax=False):
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
        use_jax : bool, optional
            Whether to use `JAX` expressions. By default `False`.

        Returns
        -------
        M : float or 1-dimensional array
            The enclosed mass.
        """
        x = r / Rs
        out = jax_switch(jnumpy.log, numpy.log, use_jax)(1 + x) - x / (1 + x)
        return 4 * numpy.pi * rho0 * Rs**3 * out

    def bounded_enclosed_mass(self, rmin, rmax, Rs, rho0, use_jax=False):
        r"""
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
        use_jax : bool, optional
            Whether to use `JAX` expressions. By default `False`.

        Returns
        -------
        M : float
            The enclosed mass within the radial range.
        """
        return (self.enclosed_mass(rmax, Rs, rho0, use_jax)
                - self.enclosed_mass(rmin, Rs, rho0, use_jax))

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
    Posterior for fitting the NFW profile in the range specified by the
    closest particle and the :math:`r_{200c}` radius. The likelihood is
    calculated as

    .. math::
        \frac{4\pi r^2 \rho(r)} {M(r_{\min} r_{200c})} \frac{m}{M / N}

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
    _clump = None
    _binsguess = 10
    _r = None
    _Npart = None
    _m = None
    _rmin = None
    _rmax = None

    def __init__(self, clump):
        # Initialise the NFW profile
        super().__init__()
        self.clump = clump

    @property
    def clump(self):
        """
        Clump object containig all particles, i.e. ones beyond :math:`R_{200c}`
        as well.

        Returns
        -------
        clump : `Clump`
            The clump object.
        """
        return self._clump

    @property
    def r(self):
        r"""
        Radial distance of particles used to fit the NFW profile, i.e. the ones
        whose radial distance is less than :math:`R_{\rm 200c}`.

        Returns
        -------
        r : 1-dimensional array
            Array of shape `(n_particles, )`.
        """
        return self._r

    @property
    def Npart(self):
        r"""
        Number of particles used to fit the NFW profile, i.e. the ones
        whose radial distance is less than :math:`R_{\rm 200c}`.

        Returns
        -------
        Npart : int
            Number of particles.
        """
        return self._Npart

    @property
    def m(self):
        r"""
        Mass of particles used to fit the NFW profile, i.e. the ones
        whose radial distance is less than :math:`R_{\rm 200c}`.

        Returns
        -------
        r : 1-dimensional array
            Array of shape `(n_particles, )`.
        """
        return self._m

    @property
    def rmin(self):
        """
        The minimum radial distance of a particle.

        Returns
        -------
        rmin : float
            The minimum distance.
        """
        return self._rmin

    @property
    def rmax(self):
        r"""
        The maximum radial distance used to fit the profile, here takem to be
        the :math:`R_{\rm 200c}`.

        Returns
        -------
        rmax : float
            The R200c radius.
        """
        return self._rmax

    @clump.setter
    def clump(self, clump):
        """Sets `clump` and precalculates useful things."""
        if not isinstance(clump, Clump):
            raise TypeError(
                "`clump` must be :py:class:`csiborgtools.fits.Clump` type. "
                "Currently `{}`".format(type(clump)))
        self._clump = clump
        # The minimum separation
        rmin = self.clump.rmin
        rmax = self.clump.r200
        # Set the distances
        self._rmin = rmin
        self._rmax = rmax
        # Set particles that will be used to fit the halo
        mask_r200 = (self.clump.r >= rmin) & (self.clump.r <= rmax)
        self._r = self.clump.r[mask_r200]
        self._m = self.clump.m[mask_r200]
        self._Npart = self._r.size
        # Ensure that the minimum separation is > 0 for finite log
        if self.rmin > 0:
            self._logrmin = numpy.log10(self.rmin)
        else:
            self._logrmin = numpy.log10(numpy.min(self.r[self.r > 0]))
        self._logrmax = numpy.log10(self.rmax)
        self._logprior_volume = numpy.log(self._logrmax - self._logrmin)
        # Precalculate useful things
        self._logMtot = numpy.log(numpy.sum(self.m))
        gamma = 4 * numpy.pi * self.r**2 * self.m * self.Npart
        self._ll0 = numpy.sum(numpy.log(gamma)) - self.Npart * self._logMtot

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
        Mnfw_norm = self.bounded_enclosed_mass(self.rmin, self.rmax,
                                               10**logRs, 1)
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

    def loglikelihood(self, logRs, use_jax=False):
        """
        Logarithmic likelihood.

        Parameters
        ----------
        logRs : float
            Logarithmic scale factor in units matching the coordinates.
        use_jax : bool, optional
            Whether to use `JAX` expressions. By default `False`.

        Returns
        -------
        ll : float
            The logarithmic likelihood.
        """
        Rs = 10**logRs
        log = jnumpy.log if use_jax else numpy.log
        # Expected enclosed mass from a NFW
        Mnfw = self.bounded_enclosed_mass(self.rmin, self.rmax,
                                          Rs, 1, use_jax)
        ll = jax_switch(jnumpy.sum, numpy.sum, use_jax)(self.logprofile(
            self.r, Rs, 1, use_jax))
        ll += self._ll0
        return ll - self.Npart * log(Mnfw)

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
        bins = numpy.linspace(self.rmin, self.rmax,
                              self._binsguess)
        counts, edges = numpy.histogram(self.r, bins)
        return numpy.log10(edges[numpy.argmax(counts)])

    def __call__(self, logRs, use_jax=False):
        """
        Logarithmic posterior. Sum of the logarithmic prior and likelihood.

        Parameters
        ----------
        logRs : float
            Logarithmic scale factor in units matching the coordinates.
        use_jax : bool, optional
            Whether to use `JAX` expressions. By default `False`.

        Returns
        -------
        lpost : float
            The logarithmic posterior.
        """
        lp = self.logprior(logRs)
        if not numpy.isfinite(lp):
            return - numpy.infty
        return self.loglikelihood(logRs, use_jax) + lp

    def uncertainty_at_maxpost(self, logRs_max):
        r"""
        Calculate Gaussian approximation of the uncertainty at `logRs_max`, the
        maximum a-posteriori esitmate. This is the square root of the negative
        inverse 2nd derivate of the logarithimic posterior with respect to the
        logarithm of the scale factor. This is only valid `logRs_max` is the
        maximum of the posterior!

        This uses `JAX`. The functions should be compiled but unless there is
        a need for more speed this is fine as it is.

        Parameters
        ----------
        logRs_max : float
            Position :math:`\log R_{\rm s}` to evaluate the uncertainty. Must
            be the maximum.

        Returns
        -------
        uncertainty : float
            The uncertainty calculated as outlined above.
        """
        def f(x):
            return self(x, use_jax=True)

        # Evaluate the second derivative
        h = grad(grad(f))(logRs_max)
        h = float(h)
        if not h < 0:
            return numpy.nan
        return (- 1 / h)**0.5

    def maxpost_logRs(self, calc_err=False):
        r"""
        Maximum a-posterio estimate of the scale radius :math:`\log R_{\rm s}`.
        Returns the scale radius if the fit converged, otherwise `numpy.nan`.

        Parameters
        ----------
        calc_err : bool, optional
            Optional toggle to calculate the uncertainty on the scale radius.
            By default false.

        Returns
        -------
        res : float
            The scale radius.
        uncertainty : float
            The uncertainty on the scale radius. Calculated following
            `self.uncertainty_at_maxpost`.
        """
        # Loss function to optimize
        def loss(logRs):
            return - self(logRs)
        res = minimize_scalar(loss, bounds=(self._logrmin, self._logrmax),
                              method='bounded')
        if not res.success:
            return numpy.nan, numpy.nan
        e_logRs = self.uncertainty_at_maxpost(res.x) if calc_err else numpy.nan
        return res.x, e_logRs
