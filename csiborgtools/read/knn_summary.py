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
from os.path import join
from glob import glob
import numpy
from scipy.special import factorial
import joblib


class kNNCDFReader:
    """
    Shortcut object to read in the kNN CDF data.
    """
    def read(self, run, folder, rmin=None, rmax=None, to_clip=True):
        """
        Read the auto- or cross-correlation kNN-CDF data. Infers the type from
        the data files.

        Parameters
        ----------
        run : str
            Run ID to read in.
        folder : str
            Path to the folder where the auto-correlation kNN-CDF is stored.
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
        """
        run += ".p"
        files = [f for f in glob(join(folder, "*")) if run in f]
        if len(files) == 0:
            raise RuntimeError("No files found for run `{}`.".format(run[:-2]))

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
                rs = data["rs"]
            out[i, ...] = data[kind]

            if isauto and to_clip:
                out[i, ...] = self.clipped_cdf(out[i, ...])

        # Apply separation cuts
        mask = (rs >= rmin if rmin is not None else rs > 0)
        mask &= (rs <= rmax if rmax is not None else rs < numpy.infty)
        rs = rs[mask]
        out = out[..., mask]

        return rs, out

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
        k : int
            Number of objects.
        ndensity : float
            Number density of objects.

        Returns
        -------
        pk : 1-dimensional array
            The PDF that a spherical volume of radius :math:`r` contains
            :math:`k` objects.
        """
        V = 4 * numpy.pi / 3 * rs**3
        return (ndensity * V)**k / factorial(k) * numpy.exp(-ndensity * V)

    @staticmethod
    def cross_files(ic, folder):
        """
        Return the file paths corresponding to the cross-correlation of a given
        IC.

        Parameters
        ----------
        ic : int
            The desired IC.
        folder : str
            The folder containing the cross-correlation files.

        Returns
        -------
        filepath : list of str
        """
        return [file for file in glob(join(folder, "*")) if str(ic) in file]
