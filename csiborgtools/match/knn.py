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
kNN-CDF calculation
"""
import numpy
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic
from tqdm import tqdm


class kNN_CDF:
    """
    Object to calculate the kNN-CDF for a set of CSiBORG halo catalogues from
    their kNN objects.
    """
    @staticmethod
    def rvs_in_sphere(nsamples, R, random_state=42, dtype=numpy.float32):
        """
        Generate random samples in a sphere of radius `R` centered at the
        origin.

        Parameters
        ----------
        nsamples : int
            Number of samples to generate.
        R : float
            Radius of the sphere.
        random_state : int, optional
            Random state for the random number generator.
        dtype : numpy dtype, optional
            Data type, by default `numpy.float32`.

        Returns
        -------
        samples : 2-dimensional array of shape `(nsamples, 3)`
        """
        gen = numpy.random.default_rng(random_state)
        # Sample spherical coordinates
        r = gen.uniform(0, 1, nsamples).astype(dtype)**(1/3) * R
        theta = 2 * numpy.arcsin(gen.uniform(0, 1, nsamples).astype(dtype))
        phi = 2 * numpy.pi * gen.uniform(0, 1, nsamples).astype(dtype)
        # Convert to cartesian coordinates
        x = r * numpy.sin(theta) * numpy.cos(phi)
        y = r * numpy.sin(theta) * numpy.sin(phi)
        z = r * numpy.cos(theta)

        return numpy.vstack([x, y, z]).T

    @staticmethod
    def cdf_from_samples(r, rmin=None, rmax=None, neval=None,
                         dtype=numpy.float32):
        """
        Calculate the CDF from samples.

        Parameters
        ----------
        r : 1-dimensional array
            Distance samples.
        rmin : float, optional
            Minimum distance to evaluate the CDF.
        rmax : float, optional
            Maximum distance to evaluate the CDF.
        neval : int, optional
            Number of points to evaluate the CDF. By default equal to `len(x)`.
        dtype : numpy dtype, optional
            Calculation data type. By default `numpy.float32`.

        Returns
        -------
        r : 1-dimensional array
            Distances at which the CDF is evaluated.
        cdf : 1-dimensional array
            CDF evaluated at `r`.
        """
        r = numpy.copy(r)  # Make a copy not to overwrite the original
        # Make cuts on distance
        r = r[r >= rmin] if rmin is not None else r
        r = r[r <= rmax] if rmax is not None else r

        # Calculate the CDF
        r = numpy.sort(r)
        cdf = numpy.arange(r.size) / r.size

        if neval is not None:  # Optinally interpolate at given points
            _r = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), neval,
                                dtype=dtype)
            cdf = interp1d(r, cdf, kind="linear", fill_value=numpy.nan,
                           bounds_error=False)(_r).astype(dtype)
            r = _r

        return r, cdf

    @staticmethod
    def peaked_cdf(cdf, make_copy=True):
        """
        Transform the CDF to a peaked CDF.

        Parameters
        ----------
        cdf : 1- or 2- or 3-dimensional array
            CDF to be transformed along the last axis.
        make_copy : bool, optional
            Whether to make a copy of the CDF before transforming it to avoid
            overwriting it.

        Returns
        -------
        peaked_cdf : 1- or 2- or 3-dimensional array
        """
        cdf = numpy.copy(cdf) if make_copy else cdf
        cdf[cdf > 0.5] = 1 - cdf[cdf > 0.5]
        return cdf

    def brute_cdf(self, knn, nneighbours, Rmax, nsamples, rmin, rmax, neval,
                 random_state=42, dtype=numpy.float32):
        """
        Calculate the CDF for a kNN of CSiBORG halo catalogues without batch
        sizing. This can become memory intense for large numbers of randoms
        and, therefore, is only for testing purposes.

        Parameters
        ----------
        knns : `sklearn.neighbors.NearestNeighbors`
            kNN of CSiBORG halo catalogues.
        neighbours : int
            Maximum number of neighbours to use for the kNN-CDF calculation.
        Rmax : float
            Maximum radius of the sphere in which to sample random points for
            the knn-CDF calculation. This should match the CSiBORG catalogues.
        nsamples : int
            Number of random points to sample for the knn-CDF calculation.
        rmin : float
            Minimum distance to evaluate the CDF.
        rmax : float
            Maximum distance to evaluate the CDF.
        neval : int
            Number of points to evaluate the CDF.
        random_state : int, optional
            Random state for the random number generator.
        dtype : numpy dtype, optional
            Calculation data type. By default `numpy.float32`.

        Returns
        -------
        rs : 1-dimensional array
            Distances at which the CDF is evaluated.
        cdfs : 2-dimensional array
            CDFs evaluated at `rs`.
        """
        rand = self.rvs_in_sphere(nsamples, Rmax, random_state=random_state)

        dist, __ = knn.kneighbors(rand, nneighbours)
        dist = dist.astype(dtype)

        cdf = [None] * nneighbours
        for j in range(nneighbours):
            rs, cdf[j] = self.cdf_from_samples(dist[:, j], rmin=rmin,
                                               rmax=rmax, neval=neval)

        cdf = numpy.asanyarray(cdf)
        return rs, cdf

    def __call__(self, *knns, nneighbours, Rmax, nsamples, rmin, rmax, neval,
                batch_size=None, verbose=True, random_state=42,
                left_nan=True, right_nan=True, dtype=numpy.float32):
        """
        Calculate the CDF for a set of kNNs of CSiBORG halo catalogues.

        Parameters
        ----------
        *knns : `sklearn.neighbors.NearestNeighbors` instances
            kNNs of CSiBORG halo catalogues.
        neighbours : int
            Maximum number of neighbours to use for the kNN-CDF calculation.
        Rmax : float
            Maximum radius of the sphere in which to sample random points for
            the knn-CDF calculation. This should match the CSiBORG catalogues.
        nsamples : int
            Number of random points to sample for the knn-CDF calculation.
        rmin : float
            Minimum distance to evaluate the CDF.
        rmax : float
            Maximum distance to evaluate the CDF.
        neval : int
            Number of points to evaluate the CDF.
        batch_size : int, optional
            Number of random points to sample in each batch. By default equal
            to `nsamples`, however recommeded to be smaller to avoid requesting
            too much memory,
        verbose : bool, optional
            Verbosity flag.
        random_state : int, optional
            Random state for the random number generator.
        left_nan : bool, optional
            Whether to set values where the CDF is 0 to `numpy.nan`. By
            default `True`.
        right_nan : bool, optional
            Whether to set values where the CDF is 1 to `numpy.nan` after its
            first occurence to 1. By default `True`.
        dtype : numpy dtype, optional
            Calculation data type. By default `numpy.float32`.

        Returns
        -------
        rs : 1-dimensional array
            Distances at which the CDF is evaluated.
        cdfs : 2 or 3-dimensional array
            CDFs evaluated at `rs`.
        """
        batch_size = nsamples if batch_size is None else batch_size
        assert nsamples >= batch_size
        nbatches = nsamples // batch_size  # Number of batches

        # Preallocate the bins and the CDF array
        bins = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), neval)
        cdfs = numpy.zeros((len(knns), nneighbours, neval - 1), dtype=dtype)
        for i, knn in enumerate(tqdm(knns) if verbose else knns):
            # Loop over batches. This is to avoid generating large mocks
            # requiring a lot of memory. Add counts to the CDF array
            for j in range(nbatches):
                rand = self.rvs_in_sphere(batch_size, Rmax,
                                          random_state=random_state + j)
                dist, _indxs = knn.kneighbors(rand, nneighbours)
                for k in range(nneighbours):  # Count for each neighbour
                    _counts, __, __ = binned_statistic(
                        dist[:, k], dist[:, k], bins=bins, statistic="count",
                        range=(rmin, rmax))
                    cdfs[i, k, :] += _counts

        rs = (bins[1:] + bins[:-1]) / 2     # Bin centers
        cdfs = numpy.cumsum(cdfs, axis=-1)  # Cumulative sum, i.e. the CDF
        for i in range(len(knns)):
            for k in range(nneighbours):
                cdfs[i, k, :] /= cdfs[i, k, -1]
                # Set to NaN values after the first point where the CDF is 1
                if right_nan:
                    ns = numpy.where(cdfs[i, k, :] == 1.)[0]
                    if ns.size > 1:
                        cdfs[i, k, ns[1]:] = numpy.nan

        # Set to NaN values where the CDF is 0
        if left_nan:
            cdfs[cdfs == 0] = numpy.nan

        cdfs = cdfs[0, ...] if len(knns) == 1 else cdfs
        return rs, cdfs
