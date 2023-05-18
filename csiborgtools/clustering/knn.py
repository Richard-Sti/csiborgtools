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
kNN-CDF calculation.
"""
import numpy
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic

from .utils import BaseRVS


class kNN_1DCDF:
    """
    Object to calculate the 1-dimensional kNN-CDF statistic.
    """
    @staticmethod
    def cdf_from_samples(r, rmin=None, rmax=None, neval=None,
                         dtype=numpy.float32):
        """
        Calculate the kNN-CDF from a sampled PDF.

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
    def joint_to_corr(cdf0, cdf1, joint_cdf):
        """
        Calculate the correlation function from the joint kNN-CDFs.

        Parameters
        ----------
        cdf0 : 2-dimensional array
            CDF evaluated at `rs` of the first kNN.
        cdf1 : 2-dimensional array
            CDF evaluated at `rs` of the second kNN.
        joint_cdf : 2-dimensional array
            Joint CDF evaluated at `rs`.

        Returns
        -------
        corr : 2-dimensional array
            Correlation function evaluated at `rs`.
        """
        assert cdf0.ndim == cdf1.ndim == joint_cdf.ndim == 2
        corr = numpy.zeros_like(joint_cdf)
        for k in range(joint_cdf.shape[0]):
            corr[k, :] = joint_cdf[k, :] - cdf0[k, :] * cdf1[k, :]
        return corr

    def brute_cdf(self, knn, rvs_gen, nneighbours, nsamples, rmin, rmax, neval,
                  random_state=42, dtype=numpy.float32):
        """
        Calculate the kNN-CDF without batch sizing. This can become memory
        intense for large numbers of randoms and, therefore, is primarily for
        testing purposes.

        Parameters
        ----------
        knn : `sklearn.neighbors.NearestNeighbors`
            Catalogue NN object.
        rvs_gen : :py:class:`csiborgtools.clustering.BaseRVS`
            Uniform RVS generator matching `knn`.
        nneighbours : int
            Maximum number of neighbours to use for the kNN-CDF calculation.
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
        assert isinstance(rvs_gen, BaseRVS)
        rand = rvs_gen(nsamples, random_state=random_state)

        dist, __ = knn.kneighbors(rand, nneighbours)
        dist = dist.astype(dtype)

        cdf = [None] * nneighbours
        for j in range(nneighbours):
            rs, cdf[j] = self.cdf_from_samples(dist[:, j], rmin=rmin,
                                               rmax=rmax, neval=neval)

        cdf = numpy.asanyarray(cdf)
        return rs, cdf

    def joint(self, knn0, knn1, rvs_gen, nneighbours, nsamples, rmin, rmax,
              neval, batch_size=None, random_state=42,
              dtype=numpy.float32):
        """
        Calculate the joint knn-CDF.

        Parameters
        ----------
        knn0 : `sklearn.neighbors.NearestNeighbors` instance
            NN object of the first catalogue.
        knn1 : `sklearn.neighbors.NearestNeighbors` instance
            NN object of the second catalogue.
        rvs_gen : :py:class:`csiborgtools.clustering.BaseRVS`
            Uniform RVS generator matching `knn1` and `knn2`.
        nneighbours : int
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
        random_state : int, optional
            Random state for the random number generator.
        dtype : numpy dtype, optional
            Calculation data type. By default `numpy.float32`.

        Returns
        -------
        rs : 1-dimensional array
            Distances at which the CDF is evaluated.
        cdf0 : 2-dimensional array
            CDF evaluated at `rs` of the first kNN.
        cdf1 : 2-dimensional array
            CDF evaluated at `rs` of the second kNN.
        joint_cdf : 2-dimensional array
            Joint CDF evaluated at `rs`.
        """
        assert isinstance(rvs_gen, BaseRVS)
        batch_size = nsamples if batch_size is None else batch_size
        assert nsamples >= batch_size
        nbatches = nsamples // batch_size

        bins = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), neval)
        joint_cdf = numpy.zeros((nneighbours, neval - 1), dtype=dtype)
        cdf0 = numpy.zeros_like(joint_cdf)
        cdf1 = numpy.zeros_like(joint_cdf)

        jointdist = numpy.zeros((batch_size, 2), dtype=dtype)
        for j in range(nbatches):
            rand = rvs_gen(batch_size, random_state=random_state + j)
            dist0, __ = knn0.kneighbors(rand, nneighbours)
            dist1, __ = knn1.kneighbors(rand, nneighbours)

            for k in range(nneighbours):
                jointdist[:, 0] = dist0[:, k]
                jointdist[:, 1] = dist1[:, k]
                maxdist = numpy.max(jointdist, axis=1)
                # Joint CDF
                _counts, __, __ = binned_statistic(
                    maxdist, maxdist, bins=bins, statistic="count",
                    range=(rmin, rmax))
                joint_cdf[k, :] += _counts
                # First CDF
                _counts, __, __ = binned_statistic(
                    dist0[:, k], dist0[:, k], bins=bins, statistic="count",
                    range=(rmin, rmax))
                cdf0[k, :] += _counts
                # Second CDF
                _counts, __, __ = binned_statistic(
                    dist1[:, k], dist1[:, k], bins=bins, statistic="count",
                    range=(rmin, rmax))
                cdf1[k, :] += _counts

        joint_cdf = numpy.cumsum(joint_cdf, axis=-1)
        cdf0 = numpy.cumsum(cdf0, axis=-1)
        cdf1 = numpy.cumsum(cdf1, axis=-1)
        for k in range(nneighbours):
            joint_cdf[k, :] /= joint_cdf[k, -1]
            cdf0[k, :] /= cdf0[k, -1]
            cdf1[k, :] /= cdf1[k, -1]

        rs = (bins[1:] + bins[:-1]) / 2     # Bin centers
        return rs, cdf0, cdf1, joint_cdf

    def __call__(self, knn, rvs_gen, nneighbours, nsamples, rmin, rmax, neval,
                 batch_size=None, random_state=42, dtype=numpy.float32):
        """
        Calculate the CDF for a set of kNNs of halo catalogues.

        Parameters
        ----------
        knn : `sklearn.neighbors.NearestNeighbors`
            Catalogue NN object.
        rvs_gen : :py:class:`csiborgtools.clustering.BaseRVS`
            Uniform RVS generator matching `knn1` and `knn2`.
        nneighbours : int
            Maximum number of neighbours to use for the kNN-CDF calculation.
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
        random_state : int, optional
            Random state for the random number generator.
        dtype : numpy dtype, optional
            Calculation data type. By default `numpy.float32`.

        Returns
        -------
        rs : 1-dimensional array
            Distances at which the CDF is evaluated.
        cdf : 2-dimensional array
            CDF evaluated at `rs`.
        """
        assert isinstance(rvs_gen, BaseRVS)
        batch_size = nsamples if batch_size is None else batch_size
        assert nsamples >= batch_size
        nbatches = nsamples // batch_size

        # Preallocate the bins and the CDF array
        bins = numpy.logspace(numpy.log10(rmin), numpy.log10(rmax), neval)
        cdf = numpy.zeros((nneighbours, neval - 1), dtype=dtype)
        for i in range(nbatches):
            rand = rvs_gen(batch_size, random_state=random_state + i)
            dist, __ = knn.kneighbors(rand, nneighbours)

            for k in range(nneighbours):  # Count for each neighbour
                _counts, __, __ = binned_statistic(
                    dist[:, k], dist[:, k], bins=bins, statistic="count",
                    range=(rmin, rmax))
                cdf[k, :] += _counts

        cdf = numpy.cumsum(cdf, axis=-1)  # Cumulative sum, i.e. the CDF
        for k in range(nneighbours):
            cdf[k, :] /= cdf[k, -1]

        rs = (bins[1:] + bins[:-1]) / 2     # Bin centers
        return rs, cdf
