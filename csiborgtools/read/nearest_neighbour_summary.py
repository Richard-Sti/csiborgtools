# Copyright (C) 2023 Richard Stiskalek
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
Nearest neighbour summary for assessing goodness-of-reconstruction of a halo in
the final snapshot.
"""
from math import floor

import numpy
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde, kstest

from numba import jit
from tqdm import tqdm


class NearestNeighbourReader:
    """
    Shortcut object to read in nearest neighbour data for assessing the
    goodness-of-reconstruction of a halo in the final snapshot.

    Parameters
    ----------
    rmax_radial : float
        Radius of the high-resolution region.
    nbins_radial : int
        Number of radial bins.
    rmax_neighbour : float
        Maximum distance to consider for the nearest neighbour.
    nbins_neighbour : int
        Number of bins for the nearest neighbour.
    paths : py:class`csiborgtools.read.Paths`
        Paths object.
    **kwargs : dict
        Other keyword arguments for backward compatibility. Not used.
    """
    _paths = None
    _rmax_radial = None
    _nbins_radial = None
    _rmax_neighbour = None
    _nbins_neighbour = None

    def __init__(self, rmax_radial, nbins_radial, rmax_neighbour,
                 nbins_neighbour, paths, **kwargs):
        self.paths = paths
        self.rmax_radial = rmax_radial
        self.nbins_radial = nbins_radial
        self.rmax_neighbour = rmax_neighbour
        self.nbins_neighbour = nbins_neighbour

    @property
    def rmax_radial_radial(self):
        """
        Radius of the high-resolution region.

        Parameters
        ----------
        rmax_radial_radial : float
        """
        return self._rmax_radial_radial

    @rmax_radial_radial.setter
    def rmax_radial_radial(self, rmax_radial_radial):
        assert isinstance(rmax_radial_radial, float)
        self._rmax_radial_radial = rmax_radial_radial

    @property
    def paths(self):
        """
        Paths manager.

        Parameters
        ----------
        paths : py:class`csiborgtools.read.Paths`
        """
        return self._paths

    @property
    def nbins_radial(self):
        """
        Number radial of bins.

        Returns
        -------
        nbins_radial : int
        """
        return self._nbins_radial

    @nbins_radial.setter
    def nbins_radial(self, nbins_radial):
        assert isinstance(nbins_radial, int)
        self._nbins_radial = nbins_radial

    @property
    def nbins_neighbour(self):
        """
        Number of neighbour bins.

        Returns
        -------
        nbins_neighbour : int
        """
        return self._nbins_neighbour

    @nbins_neighbour.setter
    def nbins_neighbour(self, nbins_neighbour):
        assert isinstance(nbins_neighbour, int)
        self._nbins_neighbour = nbins_neighbour

    @property
    def rmax_neighbour(self):
        """
        Maximum neighbour distance.

        Returns
        -------
        rmax_neighbour : float
        """
        return self._rmax_neighbour

    @rmax_neighbour.setter
    def rmax_neighbour(self, rmax_neighbour):
        assert isinstance(rmax_neighbour, float)
        self._rmax_neighbour = rmax_neighbour

    @paths.setter
    def paths(self, paths):
        self._paths = paths

    @property
    def radial_bin_edges(self):
        """
        Radial bins.

        Returns
        -------
        radial_bins : 1-dimensional array
        """
        nbins = self.nbins_radial + 1
        return self.rmax_radial * numpy.linspace(0, 1, nbins)**(1./3)

    @property
    def neighbour_bin_edges(self):
        """
        Neighbour bins edges

        Returns
        -------
        neighbour_bins : 1-dimensional array
        """
        nbins = self.nbins_neighbour + 1
        return numpy.linspace(0, self.rmax_neighbour, nbins)

    def bin_centres(self, kind):
        """
        Bin centres. Either for `radial` or `neighbour` bins.

        Parameters
        ----------
        kind : str
            Bin kind. Either `radial` or `neighbour`.

        Returns
        -------
        bin_centres : 1-dimensional array
        """
        assert kind in ["radial", "neighbour"]
        if kind == "radial":
            edges = self.radial_bin_edges
        else:
            edges = self.neighbour_bin_edges
        return 0.5 * (edges[1:] + edges[:-1])

    def read_single(self, simname, run, nsim, nobs=None):
        """
        Read in the nearest neighbour distances for halos from a single
        simulation.

        Parameters
        ----------
        simname : str
            Simulation name. Must be either `csiborg` or `quijote`.
        run : str
            Run name.
        nsim : int
            Simulation index.
        nobs : int, optional
            Fiducial Quijote observer index.

        Returns
        -------
        data : numpy archive
            Archive with keys `ndist`, `rdist`, `mass`, `cross_hindxs``
        """
        assert simname in ["csiborg", "quijote"]
        fpath = self.paths.cross_nearest(simname, run, "dist", nsim, nobs)
        return numpy.load(fpath)

    def count_neighbour(self, out, ndist, rdist):
        """
        Count the number of neighbours for each halo as a function of its
        radial distance.

        Parameters
        ----------
        out : 2-dimensional array of shape `(nbins_radial, nbins_neighbour)`
            Output array to write to. Results are added to this array.
        ndist : 2-dimensional array of shape `(nhalos, ncross_simulations)`
            Distance of each halo to its nearest neighbour from a cross
            simulation.
        rdist : 1-dimensional array of shape `(nhalos, )`
            Distance of each halo to the centre of the high-resolution region.

        Returns
        -------
        out : 2-dimensional array of shape `(nbins_radial, nbins_neighbour)`
        """
        return count_neighbour(out, ndist, rdist, self.radial_bin_edges,
                               self.rmax_neighbour, self.nbins_neighbour)

    def build_dist(self, counts, kind):
        """
        Build the a PDF or a CDF for the nearest neighbour distribution from
        binned counts as a function of radial distance from the centre of the
        high-resolution region.

        Parameters
        ----------
        counts : 2-dimensional array of shape `(nbins_radial, nbins_neighbour)`
            Binned counts of the number of neighbours as a function of
            radial distance.

        Returns
        -------
        dist : 2-dimensional array of shape `(nbins_radial, nbins_neighbour)`
        """
        assert kind in ["pdf", "cdf"]
        if kind == "pdf":
            neighbour_bin_edges = self.neighbour_bin_edges
            dx = neighbour_bin_edges[1] - neighbour_bin_edges[0]
            counts /= numpy.sum(dx * counts, axis=1).reshape(-1, 1)
        else:
            x = self.bin_centres("neighbour")
            counts = cumulative_trapezoid(counts, x, axis=1, initial=0)
            counts /= counts[:, -1].reshape(-1, 1)
        return counts

    def kl_divergence(self, simname, run, nsim, pdf, nobs=None, verbose=True):
        r"""
        Calculate the Kullback-Leibler divergence of the nearest neighbour
        distribution of a reference halo relative to an expected distribution
        from an unconstrained suite of simulations. Approximates reference halo
        neighbour distribution with a Gaussian KDE.

        Parameters
        ----------
        simname : str
            Simulation name. Must be either `csiborg` or `quijote`.
        run : str
            Run name.
        nsim : int
            Simulation index.
        cdf : 2-dimensional array of shape `(nbins_radial, nbins_neighbour)`
            CDF of the nearest neighbour distribution in an unconstrained
            suite of simiulations.
        nobs : int, optional
            Fiducial Quijote observer index.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        kl_divergence: 1-dimensional array of shape `(nhalos,)`
            Information gain from going from the expected distribution to the
            observed distribution in bits.
        """
        assert simname in ["csiborg", "quijote"]
        data = self.read_single(simname, run, nsim, nobs)

        rdist = data["rdist"]
        ndist = data["ndist"]
        rbin_edges = self.radial_bin_edges

        # Create an interpolation function for each radial bin.
        xbin = self.bin_centres("neighbour")
        interp_kwargs = {"kind": "cubic",
                         "bounds_error": False,
                         "fill_value": 1e-16,
                         "assume_sorted": True}
        pdf_interp = [interp1d(xbin, pdf[i, :], **interp_kwargs)
                      for i in range(self.nbins_radial)]

        def KL_density(x, p, q, p_norm, q_norm):
            p = p(x) / p_norm
            q = q(x) / q_norm
            return p * numpy.log2(p / q)

        # We loop over each halo and find its radial bin. Then calculate the
        # KL divergence between the Quijote distribution and the sampled
        # distribution in CSiBORG.
        out = numpy.full(rdist.size, numpy.nan, dtype=numpy.float32)
        cells = numpy.digitize(rdist, rbin_edges) - 1
        for i, radial_cell in enumerate(tqdm(cells) if verbose else cells):
            xmin, xmax = numpy.min(ndist[i, :]), numpy.max(ndist[i, :])
            xrange = numpy.linspace(xmin, xmax, 1000)

            ykde = gaussian_kde(ndist[i, :])(xrange)
            kde = interp1d(xrange, ykde, **interp_kwargs)
            kde_norm = quad(kde, xmin, xmax)[0]

            out[i] = quad(
                KL_density, xmin, xmax,
                args=(kde, pdf_interp[radial_cell], kde_norm, 1.0),
                limit=250, points=(xmin, xmax), epsabs=1e-4, epsrel=1e-4)[0]
        return out

    def ks_significance(self, simname, run, nsim, cdf, nobs=None):
        r"""
        Calculate the p-value significance of the nearest neighbour of a
        reference halo relative to an unconstrained simulation using the
        Kolmogorov–Smirnov test.

        Parameters
        ----------
        simname : str
            Simulation name. Must be either `csiborg` or `quijote`.
        run : str
            Run name.
        nsim : int
            Simulation index.
        cdf : 2-dimensional array of shape `(nbins_radial, nbins_neighbour)`
            CDF of the nearest neighbour distribution in an unconstrained
            suite of simiulations.
        nobs : int, optional
            Fiducial Quijote observer index.

        Returns
        -------
        pval : 1-dimensional array of shape `(nhalos,)`
        """
        assert simname in ["csiborg", "quijote"]
        data = self.read_single(simname, run, nsim, nobs)

        rdist = data["rdist"]
        ndist = data["ndist"]
        rbin_edges = self.radial_bin_edges

        # Create an interpolation function for each radial bin.
        xbin = self.bin_centres("neighbour")
        kwargs = {"bounds_error": False,
                  "fill_value": numpy.nan,
                  "assume_sorted": True}
        N = self.nbins_radial
        cdf_interp = [interp1d(xbin, cdf[i, :], **kwargs) for i in range(N)]

        # We loop over each halo and find its radial bin. Then calculate the
        # p-value under null hypothesis and convert it to a sigma value.
        out = numpy.full(rdist.size, numpy.nan, dtype=numpy.float64)
        for i, radial_cell in enumerate(numpy.digitize(rdist, rbin_edges) - 1):
            # The null hypothesis is that the distances in Quijote are larger
            # or equal to CSiBORG.
            ks = kstest(ndist[i, :], cdf_interp[radial_cell], N=10000,
                        alternative="greater", method="exact")
            out[i] = ks.pvalue
        return out


###############################################################################
#                           Support functions                                 #
###############################################################################


@jit(nopython=True)
def count_neighbour(counts, ndist, rdist, rbin_edges, rmax_neighbour,
                    nbins_neighbour):
    """
    Count the number of neighbour in neighbours bins for each halo as a funtion
    of its radial distance from the centre of the high-resolution region.

    Parameters
    ----------
    counts : 2-dimensional array of shape `(nbins_radial, nbins_neighbour)`
        Array to store the counts.
    ndist : 2-dimensional array of shape `(nhalos, ncross_simulations)`
        Distance of each halo to its nearest neighbour from a cross simulation.
    rdist : 1-dimensional array of shape `(nhalos, )`
        Distance of each halo to the centre of the high-resolution region.
    rbin_edges : 1-dimensional array of shape `(nbins_radial + 1, )`
        Edges of the radial bins.
    rmax_neighbour : float
        Maximum neighbour distance.
    nbins_neighbour : int
        Number of neighbour bins.

    Returns
    -------
    counts : 2-dimensional array of shape `(nbins_radial, nbins_neighbour)`
    """
    ncross = ndist.shape[1]
    # We normalise the neighbour distance by the maximum neighbour distance and
    # multiply by the number of bins. This way, the floor of each distance is
    # the bin number.
    ndist /= rmax_neighbour
    ndist *= nbins_neighbour
    # We loop over each halo, assign it to a radial bin and then assign its
    # neighbours to bins.
    for i, radial_cell in enumerate(numpy.digitize(rdist, rbin_edges) - 1):
        for j in range(ncross):
            neighbour_cell = floor(ndist[i, j])
            if neighbour_cell < nbins_neighbour:
                counts[radial_cell, neighbour_cell] += 1

    return counts
