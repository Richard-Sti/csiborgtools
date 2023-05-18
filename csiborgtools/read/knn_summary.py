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
"""kNN-CDF reader."""
import joblib
import numpy
from scipy.special import factorial


class kNNCDFReader:
    """
    Shortcut object to read in the kNN CDF data.

    Parameters
    ----------
    paths : py:class`csiborgtools.read.Paths`
    """
    _paths = None

    def __init__(self, paths):
        self.paths = paths

    @property
    def paths(self):
        """
        Paths manager.

        Parameters
        ----------
        paths : py:class`csiborgtools.read.Paths`
        """
        return self._paths

    @paths.setter
    def paths(self, paths):
        self._paths = paths

    def read(self, simname, run, kind, rmin=None, rmax=None, to_clip=True):
        """
        Read the auto- or cross-correlation kNN-CDF data. Infers the type from
        the data files.

        Parameters
        ----------
        simname : str
            Simulation name. Must be either `csiborg` or `quijote`.
        run : str
            Run ID to read in.
        kind : str
            Type of correlation. Can be either `auto` or `cross`.
        rmin : float, optional
            Minimum separation. By default ignored.
        rmax : float, optional
            Maximum separation. By default ignored.
        to_clip : bool, optional
            Whether to clip the auto-correlation CDF. Ignored for
            cross-correlation.

        Returns
        -------
        rs : 1-dimensional array of shape `(neval, )`
            Separations where the CDF is evaluated.
        out : 3-dimensional array of shape `(len(files), len(ks), neval)`
            Array of CDFs or cross-correlations.
        ndensity : 1-dimensional array of shape `(len(files), )`
            Number density of halos in the simulation.
        """
        assert kind in ["auto", "cross"]
        assert simname in ["csiborg", "quijote"]
        if kind == "auto":
            files = self.paths.knnauto(simname, run)
        else:
            files = self.paths.knncross(simname, run)
        if len(files) == 0:
            raise RuntimeError(f"No files found for run `{run}`.")

        for i, file in enumerate(files):
            data = joblib.load(file)
            if i == 0:  # Initialise the array
                if "corr" in data.keys():
                    kind = "corr"
                    isauto = False
                else:
                    kind = "cdf"
                    isauto = True
                out = numpy.full((len(files), *data[kind].shape), numpy.nan,
                                 dtype=numpy.float32)
                ndensity = numpy.full(len(files), numpy.nan,
                                      dtype=numpy.float32)
                rs = data["rs"]
            out[i, ...] = data[kind]
            ndensity[i] = data["ndensity"]

            if isauto and to_clip:
                out[i, ...] = self.clipped_cdf(out[i, ...])

        # Apply separation cuts
        mask = (rs >= rmin if rmin is not None else rs > 0)
        mask &= (rs <= rmax if rmax is not None else rs < numpy.infty)
        rs = rs[mask]
        out = out[..., mask]

        return rs, out, ndensity

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

    @staticmethod
    def clipped_cdf(cdf):
        """
        Clip the CDF, setting values where the CDF is either 0 or after the
        first occurence of 1 to `numpy.nan`.

        Parameters
        ----------
        cdf : 2- or 3-dimensional array
            CDF to be clipped.

        Returns
        -------
        clipped_cdf : 2- or 3-dimensional array
            The clipped CDF.
        """
        cdf = numpy.copy(cdf)
        if cdf.ndim == 2:
            cdf = cdf.reshape(1, *cdf.shape)
        nknns, nneighbours, __ = cdf.shape

        for i in range(nknns):
            for k in range(nneighbours):
                ns = numpy.where(cdf[i, k, :] == 1.)[0]
                if ns.size > 1:
                    cdf[i, k, ns[1]:] = numpy.nan
        cdf[cdf == 0] = numpy.nan

        cdf = cdf[0, ...] if nknns == 1 else cdf  # Reshape if necessary
        return cdf

    @staticmethod
    def prob_k(cdf):
        r"""
        Calculate the PDF that a spherical volume of radius :math:`r` contains
        :math:`k` objects, i.e. :math:`P(k | V = 4 \pi r^3 / 3)`.

        Parameters
        ----------
        cdf : 3-dimensional array of shape `(len(files), len(ks), len(rs))`
            Array of CDFs

        Returns
        -------
        pk : 3-dimensional array of shape `(len(files), len(ks)- 1, len(rs))`
        """
        out = numpy.full_like(cdf[..., 1:, :], numpy.nan, dtype=numpy.float32)
        nks = cdf.shape[-2]
        out[..., 0, :] = 1 - cdf[..., 0, :]

        for k in range(1, nks - 1):
            out[..., k, :] = cdf[..., k - 1, :] - cdf[..., k, :]

        return out

    def mean_prob_k(self, cdf):
        r"""
        Calculate the mean PDF that a spherical volume of radius :math:`r`
        contains :math:`k` objects, i.e. :math:`P(k | V = 4 \pi r^3 / 3)`,
        averaged over the IC realisations.

        Parameters
        ----------
        cdf : 3-dimensional array of shape `(len(files), len(ks), len(rs))`
            Array of CDFs

        Returns
        -------
        out : 3-dimensional array of shape `(len(ks) - 1, len(rs), 2)`
            Mean :math:`P(k | V = 4 \pi r^3 / 3) and its standard deviation,
            stored along the last dimension, respectively.
        """
        pk = self.prob_k(cdf)
        return numpy.stack([numpy.mean(pk, axis=0), numpy.std(pk, axis=0)],
                           axis=-1)

    def poisson_prob_k(self, rs, k, ndensity):
        r"""
        Calculate the analytical PDF that a spherical volume of
        radius :math:`r` contains :math:`k` objects, i.e.
        :math:`P(k | V = 4 \pi r^3 / 3)`, assuming a Poisson field (uniform
        distribution of points).

        Parameters
        ----------
        rs : 1-dimensional array
            Array of separations.
        k : int or 1-dimensional array
            Number of objects.
        ndensity : float or 1-dimensional array
            Number density of objects.

        Returns
        -------
        pk : 1-dimensional array or 3-dimensional array
            The PDF that a spherical volume of radius :math:`r` contains
            :math:`k` objects. If `k` and `ndensity` are both arrays, the shape
            is `(len(ndensity), len(k), len(rs))`.
        """
        V = 4 * numpy.pi / 3 * rs**3

        def probk(k, ndensity):
            return (ndensity * V)**k / factorial(k) * numpy.exp(-ndensity * V)

        if isinstance(k, int) and isinstance(ndensity, float):
            return probk(k, ndensity)

        # If either k or ndensity is an array, make sure the other is too.
        assert isinstance(k, numpy.ndarray) and k.ndim == 1
        assert isinstance(ndensity, numpy.ndarray) and ndensity.ndim == 1

        out = numpy.full((ndensity.size, k.size, rs.size), numpy.nan,
                         dtype=numpy.float32)
        for i in range(ndensity.size):
            for j in range(k.size):
                out[i, j, :] = probk(k[j], ndensity[i])
        return out
