# Copyright (C) 2022 Richard Stiskalek, Harry Desmond
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
Tools for summarising various results.
"""
from os.path import (join, isfile)
from glob import glob
import numpy
from scipy.special import factorial
import joblib
from tqdm import tqdm


###############################################################################
#                            PKReader                                         #
###############################################################################


class PKReader:
    """
    A shortcut object for reading in the power spectrum files.

    Parameters
    ----------
    ic_ids : list of int
        IC IDs to be read.
    hw : float
        Box half-width.
    fskel : str, optional
        The skeleton path. By default
        `/mnt/extraspace/rstiskalek/csiborg/crosspk/out_{}_{}_{}.p`, where
        the formatting options are `ic0, ic1, hw`.
    dtype : dtype, optional
        Output precision. By default `numpy.float32`.
    """
    def __init__(self, ic_ids, hw, fskel=None, dtype=numpy.float32):
        self.ic_ids = ic_ids
        self.hw = hw
        if fskel is None:
            fskel = "/mnt/extraspace/rstiskalek/csiborg/crosspk/out_{}_{}_{}.p"
        self.fskel = fskel
        self.dtype = dtype

    @staticmethod
    def _set_klim(kmin, kmax):
        """
        Sets limits on the wavenumber to 0 and infinity if `None`s provided.
        """
        if kmin is None:
            kmin = 0
        if kmax is None:
            kmax = numpy.infty
        return kmin, kmax

    def read_autos(self, kmin=None, kmax=None):
        """
        Read in the autocorrelation power spectra.

        Parameters
        ----------
        kmin : float, optional
            The minimum wavenumber. By default `None`, i.e. 0.
        kmin : float, optional
            The maximum wavenumber. By default `None`, i.e. infinity.

        Returns
        -------
        ks : 1-dimensional array
            Array of wavenumbers.
        pks : 2-dimensional array of shape `(len(self.ic_ids), ks.size)`
            Autocorrelation of each simulation.
        """
        kmin, kmax = self._set_klim(kmin, kmax)
        ks, pks, sel = None, None, None
        for i, nsim in enumerate(self.ic_ids):
            pk = joblib.load(self.fskel.format(nsim, nsim, self.hw))
            # Get cuts and pre-allocate arrays
            if i == 0:
                x = pk.k3D
                sel = (kmin < x) & (x < kmax)
                ks = x[sel].astype(self.dtype)
                pks = numpy.full((len(self.ic_ids), numpy.sum(sel)), numpy.nan,
                                 dtype=self.dtype)
            pks[i, :] = pk.Pk[sel, 0, 0]

        return ks, pks

    def read_single_cross(self, ic0, ic1, kmin=None, kmax=None):
        """
        Read cross-correlation between IC IDs `ic0` and `ic1`.

        Parameters
        ----------
        ic0 : int
            The first IC ID.
        ic1 : int
            The second IC ID.
        kmin : float, optional
            The minimum wavenumber. By default `None`, i.e. 0.
        kmin : float, optional
            The maximum wavenumber. By default `None`, i.e. infinity.

        Returns
        -------
        ks : 1-dimensional array
            Array of wavenumbers.
        xpk : 1-dimensional array of shape `(ks.size, )`
            Cross-correlation.
        """
        if ic0 == ic1:
            raise ValueError("Requested cross correlation for the same ICs.")
        kmin, kmax = self._set_klim(kmin, kmax)
        # Check their ordering. The latter must be larger.
        ics = (ic0, ic1)
        if ic0 > ic1:
            ics = ics[::-1]

        pk = joblib.load(self.fskel.format(*ics, self.hw))
        ks = pk.k3D
        sel = (kmin < ks) & (ks < kmax)
        ks = ks[sel].astype(self.dtype)
        xpk = pk.XPk[sel, 0, 0].astype(self.dtype)

        return ks, xpk

    def read_cross(self, kmin=None, kmax=None):
        """
        Read cross-correlation between all IC pairs.

        Parameters
        ----------
        kmin : float, optional
            The minimum wavenumber. By default `None`, i.e. 0.
        kmin : float, optional
            The maximum wavenumber. By default `None`, i.e. infinity.

        Returns
        -------
        ks : 1-dimensional array
            Array of wavenumbers.
        xpks : 3-dimensional array of shape (`nics, nics - 1, ks.size`)
            Cross-correlations. The first column is the the IC and is being
            cross-correlated with the remaining ICs, in the second column.
        """
        nics = len(self.ic_ids)

        ks, xpks = None, None
        for i, ic0 in enumerate(tqdm(self.ic_ids)):
            k = 0
            for ic1 in self.ic_ids:
                # We don't want cross-correlation
                if ic0 == ic1:
                    continue
                x, y = self.read_single_cross(ic0, ic1, kmin, kmax)
                # If in the first iteration pre-allocate arrays
                if ks is None:
                    ks = x
                    xpks = numpy.full((nics, nics - 1, ks.size), numpy.nan,
                                      dtype=self.dtype)
                xpks[i, k, :] = y
                # Bump up the iterator
                k += 1

        return ks, xpks


###############################################################################
#                            PKReader                                         #
###############################################################################


class kNNCDFReader:
    """
    Shortcut object to read in the kNN CDF data.
    """
    def read_auto(self, run, folder, rmin=None, rmax=None, to_clip=True):
        """
        Read the autocorrelation kNN CDF data.

        Parameters
        ----------
        run : str
            Run ID to read in.
        folder : str
            Path to the folder where the autocorrelation kNN-CDF is stored.
        rmin : float, optional
            Minimum separation. By default ignored.
        rmax : float, optional
            Maximum separation. By default ignored.
        to_clip : bool, optional
            Whether to clip the auto-correlation CDF.

        Returns
        -------
        rs : 1-dimensional array of shape `(neval, )`
            Separations where the CDF is evaluated.
        cdf: 3-dimensional array of shape `(len(files), len(ks), neval)`
            Array of CDFs.
        """
        run += ".p"
        files = [f for f in glob(join(folder, "*")) if run in f]
        if len(files) == 0:
            raise RuntimeError("No files found for run `{}`.".format(run[:-2]))
        kind = "corr" if "cross" in run else "cdf"

        for i, file in enumerate(files):
            data = joblib.load(file)
            if i == 0:  # Initialise the array
                out = numpy.full((len(files), *data[kind].shape), numpy.nan,
                                 dtype=numpy.float32)
                rs = data["rs"]
            out[i, ...] = data[kind]

            if to_clip:
                out[i, ...] = self.clipped_cdf(out[i, ...])

        # Apply separation cuts
        mask = (rs >= rmin if rmin is not None else rs > 0)
        mask &= (rs <= rmax if rmax is not None else rs < numpy.infty)
        rs = rs[mask]
        out = out[..., mask]

        return rs, out

    def read_cross(self, files, ks, rmin=None, rmax=None, to_clip=True):
        """
        Read the kNN CDF data can be either the auto- or cross-correlation.

        Parameters
        ----------
        files : list of str
            List of file paths to read in.
        ks : list of int
            kNN values to read in.
        rmin : float, optional
            Minimum separation. By default ignored.
        rmax : float, optional
            Maximum separation. By default ignored.
        to_clip : bool, optional
            Whether to clip the auto-correlation CDF. Ignored if reading in the
            cross-correlation.

        Returns
        -------
        rs : 1-dimensional array
            Array of separations.
        out : 4-dimensional array
            Auto-correlation or cross-correlation kNN CDFs. The shape is
            `(len(files), len(mass_thresholds), len(ks), neval)`.
        mass_thresholds : 1-dimensional array
            Array of mass thresholds.
        """
        raise RuntimeError("Not implemented yet.")
        data = joblib.load(files[0])
        if "cdf_0" in data.keys():
            isauto = True
            kind = "cdf"
        elif "corr_0" in data.keys():
            isauto = False
            kind = "corr"
        else:
            raise ValueError("Unknown data format.")
        rs = data["rs"]
        mass_thresholds = data["mass_threshold"]
        neval = data["{}_0".format(kind)].shape[1]
        out = numpy.full((len(files), len(mass_thresholds), len(ks), neval),
                         numpy.nan, dtype=numpy.float32)

        for i, file in enumerate(tqdm(files)):
            data = joblib.load(file)
            for j in range(len(mass_thresholds)):
                out[i, j, ...] = data["{}_{}".format(kind, j)][ks, :]
                if isauto and to_clip:
                    out[i, j, ...] = self.clipped_cdf(out[i, j, ...])

        # Apply separation cuts
        mask = (rs >= rmin if rmin is not None else rs > 0)
        mask &= (rs <= rmax if rmax is not None else rs < numpy.infty)
        rs = rs[mask]
        out = out[..., mask]

        return rs, out, mass_thresholds

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
        """
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
        """
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


###############################################################################
#                            PKReader                                         #
###############################################################################


class PairOverlap:
    r"""
    A shortcut object for reading in the results of matching two simulations.

    Parameters
    ----------
    cat0, catx: :py:class:`csiborgtools.read.HaloCatalogue`
        Halo catalogues corresponding to the reference and cross
        simulations.
    fskel : str, optional
        Path to the overlap. By default `None`, i.e.
        `/mnt/extraspace/rstiskalek/csiborg/overlap/cross_{}_{}.npz`.
    min_mass : float, optional
        Minimum :math:`M_{\rm tot} / M_\odot` mass in the reference catalogue.
        By default no threshold.
    max_dist : float, optional
        Maximum comoving distance in the reference catalogue. By default upper
        limit.
    """
    _cat0 = None
    _catx = None
    _data = None

    def __init__(self, cat0, catx, fskel=None, min_mass=None, max_dist=None):
        self._cat0 = cat0
        self._catx = catx

        if fskel is None:
            fskel = join("/mnt/extraspace/rstiskalek/csiborg/overlap",
                         "cross_{}_{}.npz")

        fpath = fskel.format(cat0.n_sim, catx.n_sim)
        fpath_inv = fskel.format(catx.n_sim, cat0.n_sim)
        if isfile(fpath):
            is_inverted = False
        elif isfile(fpath_inv):
            fpath = fpath_inv
            is_inverted = True
        else:
            raise FileNotFoundError(
                "No overlap file found for combination `{}` and `{}`."
                .format(cat0.n_sim, catx.n_sim))

        # We can set catalogues already now even if inverted
        d = numpy.load(fpath, allow_pickle=True)
        ngp_overlap = d["ngp_overlap"]
        smoothed_overlap = d["smoothed_overlap"]
        match_indxs = d["match_indxs"]
        if is_inverted:
            indxs = d["cross_indxs"]
            # Invert the matches
            match_indxs, ngp_overlap, smoothed_overlap = self._invert_match(
                match_indxs, ngp_overlap, smoothed_overlap, indxs.size,)
        else:
            indxs = d["ref_indxs"]

        self._data = {
            "index": indxs,
            "match_indxs": match_indxs,
            "ngp_overlap": ngp_overlap,
            "smoothed_overlap": smoothed_overlap,
            }

        self._make_refmask(min_mass, max_dist)

    @staticmethod
    def _invert_match(match_indxs, ngp_overlap, smoothed_overlap, cross_size):
        """
        Invert reference and cross matching, possible since the overlap
        definition is symmetric.

        Parameters
        ----------
        match_indxs : array of 1-dimensional arrays
            Indices of halos from the original cross catalogue matched to the
            reference catalogue.
        ngp_overlap : array of 1-dimensional arrays
            NGP pair overlap of halos between the original reference and cross
            simulations.
        smoothed_overlap : array of 1-dimensional arrays
            Smoothed pair overlap of halos between the original reference and
            cross simulations.
        cross_size : int
            The size of the cross catalogue.

        Returns
        -------
        inv_match_indxs : array of 1-dimensional arrays
            The inverted match indices.
        ind_ngp_overlap : array of 1-dimensional arrays
            The corresponding NGP overlaps to `inv_match_indxs`.
        ind_smoothed_overlap : array of 1-dimensional arrays
            The corresponding smoothed overlaps to `inv_match_indxs`.
        """
        # 1. Invert the match. Each reference halo has a list of counterparts
        # so loop over those to each counterpart assign a reference halo
        # and at the same time also add the overlaps
        inv_match_indxs = [[] for __ in range(cross_size)]
        inv_ngp_overlap = [[] for __ in range(cross_size)]
        inv_smoothed_overlap = [[] for __ in range(cross_size)]
        for ref_id in range(match_indxs.size):
            for cross_id, ngp_cross, smoothed_cross in zip(match_indxs[ref_id],
                                                           ngp_overlap[ref_id],
                                                           smoothed_overlap[ref_id]):  # noqa
                inv_match_indxs[cross_id].append(ref_id)
                inv_ngp_overlap[cross_id].append(ngp_cross)
                inv_smoothed_overlap[cross_id].append(smoothed_cross)

        # 2. Convert the cross matches and overlaps to proper numpy arrays
        # and ensure that the overlaps are ordered.
        for n in range(len(inv_match_indxs)):
            inv_match_indxs[n] = numpy.asanyarray(inv_match_indxs[n],
                                                  dtype=numpy.int32)
            inv_ngp_overlap[n] = numpy.asanyarray(inv_ngp_overlap[n],
                                                  dtype=numpy.float32)
            inv_smoothed_overlap[n] = numpy.asanyarray(inv_smoothed_overlap[n],
                                                       dtype=numpy.float32)

            ordering = numpy.argsort(inv_ngp_overlap[n])[::-1]
            inv_match_indxs[n] = inv_match_indxs[n][ordering]
            inv_ngp_overlap[n] = inv_ngp_overlap[n][ordering]
            inv_smoothed_overlap[n] = inv_smoothed_overlap[n][ordering]

        inv_match_indxs = numpy.asarray(inv_match_indxs, dtype=object)
        inv_ngp_overlap = numpy.asarray(inv_ngp_overlap, dtype=object)
        inv_smoothed_overlap = numpy.asarray(inv_smoothed_overlap,
                                             dtype=object)

        return inv_match_indxs, inv_ngp_overlap, inv_smoothed_overlap

    def _make_refmask(self, min_mass, max_dist):
        r"""
        Create a mask for the reference catalogue that accounts for the mass
        and distance cuts. Note that *no* masking is applied to the cross
        catalogue.

        Parameters
        ----------
        min_mass : float, optional
            The minimum :math:`M_{rm tot} / M_\odot` mass.
        max_dist : float, optional
            The maximum comoving distance of a halo.

        Returns
        -------
        None
        """
        # Enforce a cut on the reference catalogue
        min_mass = 0 if min_mass is None else min_mass
        max_dist = numpy.infty if max_dist is None else max_dist
        m = ((self.cat0()["totpartmass"] > min_mass)
             & (self.cat0()["dist"] < max_dist))
        # Now remove indices that are below this cut
        for p in ("index", "match_indxs", "ngp_overlap", "smoothed_overlap"):
            self._data[p] = self._data[p][m]

        self._data["refmask"] = m

    def overlap(self, from_smoothed):
        """
        Pair overlap of matched halos between the reference and cross
        simulations.

        Parameters
        ----------
        from_smoothed : bool
            Whether to use the smoothed overlap.

        Returns
        -------
        overlap : 1-dimensional array of arrays
        """
        if from_smoothed:
            return self["smoothed_overlap"]
        return self["ngp_overlap"]

    def summed_overlap(self, from_smoothed):
        """
        Summed overlap of each halo in the reference simulation with the cross
        simulation.

        Parameters
        ----------
        from_smoothed : bool
            Whether to use the smoothed overlap or not.

        Returns
        -------
        summed_overlap : 1-dimensional array of shape `(nhalos, )`
        """
        overlap = self.overlap(from_smoothed)
        return numpy.array([numpy.sum(cross)for cross in overlap])

    def prob_nomatch(self, from_smoothed):
        """
        Probability of no match for each halo in the reference simulation with
        the cross simulation. Defined as a product of 1 - overlap with other
        halos.

        Parameters
        ----------
        from_smoothed : bool
            Whether to use the smoothed overlap or not.

        Returns
        -------
        prob_nomatch : 1-dimensional array of shape `(nhalos, )`
        """
        overlap = self.overlap(from_smoothed)
        return numpy.array([numpy.product(1 - overlap) for overlap in overlap])

    def dist(self, in_initial, norm_kind=None):
        """
        Pair distances of matched halos between the reference and cross
        simulations.

        Parameters
        ----------
        in_initial : bool
            Whether to calculate separation in the initial or final snapshot.
        norm_kind : str, optional
            The kind of normalisation to apply to the distances. Can be `r200`,
            `ref_patch` or `sum_patch`.

        Returns
        -------
        dist : array of 1-dimensional arrays of shape `(nhalos, )`
        """
        assert (norm_kind is None
                or norm_kind in ("r200", "ref_patch", "sum_patch"))
        # Get positions either in the initial or final snapshot
        if in_initial:
            pos0, posx = self.cat0().positions0, self.catx().positions0
        else:
            pos0, posx = self.cat0().positions, self.catx().positions
        pos0 = pos0[self["refmask"], :]  # Apply the reference catalogue mask

        # Get the normalisation array if applicable
        if norm_kind == "r200":
            norm = self.cat0("r200")
        if norm_kind == "ref_patch":
            norm = self.cat0("lagpatch")
        if norm_kind == "sum_patch":
            patch0 = self.cat0("lagpatch")
            patchx = self.catx("lagpatch")
            norm = [None] * len(self)
            for i, ind in enumerate(self["match_indxs"]):
                norm[i] = patch0[i] + patchx[ind]
            norm = numpy.array(norm, dtype=object)

        # Now calculate distances
        dist = [None] * len(self)
        for i, ind in enumerate(self["match_indxs"]):
            # n refers to the reference halo catalogue position
            dist[i] = numpy.linalg.norm(pos0[i, :] - posx[ind, :], axis=1)

            if norm_kind is not None:
                dist[i] /= norm[i]

        return numpy.array(dist, dtype=object)

    def mass_ratio(self, mass_kind="totpartmass", in_log=True, in_abs=True):
        """
        Pair mass ratio of matched halos between the reference and cross
        simulations.

        Parameters
        ----------
        mass_kind : str, optional
            The mass kind whose ratio is to be calculated. Must be a valid
            catalogue key. By default `totpartmass`, i.e. the total particle
            mass associated with a halo.
        in_log : bool, optional
            Whether to return logarithm of the ratio. By default `True`.
        in_abs : bool, optional
            Whether to return absolute value of the ratio. By default `True`.

        Returns
        -------
        ratio : array of 1-dimensional arrays of shape `(nhalos, )`
        """
        mass0, massx = self.cat0(mass_kind), self.catx(mass_kind)

        ratio = [None] * len(self)
        for i, ind in enumerate(self["match_indxs"]):
            ratio[i] = mass0[i] / massx[ind]
            if in_log:
                ratio[i] = numpy.log10(ratio[i])
            if in_abs:
                ratio[i] = numpy.abs(ratio[i])
        return numpy.array(ratio, dtype=object)

    def counterpart_mass(self, from_smoothed, overlap_threshold=0.,
                         in_log=False, mass_kind="totpartmass"):
        """
        Calculate the expected counterpart mass of each halo in the reference
        simulation from the crossed simulation.

        Parameters
        -----------
        from_smoothed : bool
            Whether to use the smoothed overlap or not.
        overlap_threshold : float, optional
            Minimum overlap required for a halo to be considered a match. By
            default 0.0, i.e. no threshold.
        in_log : bool, optional
            Whether to calculate the expectation value in log space. By default
            `False`.
        mass_kind : str, optional
            The mass kind whose ratio is to be calculated. Must be a valid
            catalogue key. By default `totpartmass`, i.e. the total particle
            mass associated with a halo.

        Returns
        -------
        mean, std : 1-dimensional arrays of shape `(nhalos, )`
        """
        mean = numpy.full(len(self), numpy.nan, dtype=numpy.float32)
        std = numpy.full(len(self), numpy.nan, dtype=numpy.float32)

        massx = self.catx(mass_kind)           # Create references to speed
        overlap = self.overlap(from_smoothed)  # up the loop below

        for i, match_ind in enumerate(self["match_indxs"]):
            # Skip if no match
            if match_ind.size == 0:
                continue

            massx_ = massx[match_ind]  # Again just create references
            overlap_ = overlap[i]      # to the appropriate elements

            # Optionally apply overlap threshold
            if overlap_threshold > 0.:
                mask = overlap_ > overlap_threshold
                if numpy.sum(mask) == 0:
                    continue
                massx_ = massx_[mask]
                overlap_ = overlap_[mask]

            massx_ = numpy.log10(massx_) if in_log else massx_
            # Weighted average and *biased* standard deviation
            mean_ = numpy.average(massx_, weights=overlap_)
            std_ = numpy.average((massx_ - mean_)**2, weights=overlap_)**0.5

            # If in log, convert back to linear
            mean_ = 10**mean_ if in_log else mean_
            std_ = mean_ * std_ * numpy.log(10) if in_log else std_

            mean[i] = mean_
            std[i] = std_

        return mean, std

    def copy_per_match(self, par):
        """
        Make an array like `self.match_indxs` where each of its element is an
        equal value array of the pair clump property from the reference
        catalogue.

        Parameters
        ----------
        par : str
            Property to be copied over.

        Returns
        -------
        out : 1-dimensional array of shape `(nhalos, )`
        """
        vals = self.cat0(par)
        out = [None] * len(self)
        for i, ind in enumerate(self["match_indxs"]):
            out[i] = numpy.ones(ind.size) * vals[i]
        return numpy.array(out, dtype=object)

    def cat0(self, key=None, index=None):
        """
        Return the reference halo catalogue if `key` is `None`, otherwise
        return  values from the reference catalogue and apply `refmask`.

        Parameters
        ----------
        key : str, optional
            Key to get. If `None` return the whole catalogue.
        index : int or array, optional
            Indices to get, if `None` return all.

        Returns
        -------
        out : :py:class:`csiborgtools.read.HaloCatalogue` or array
        """
        if key is None:
            return self._cat0
        out = self._cat0[key][self["refmask"]]
        return out if index is None else out[index]

    def catx(self, key=None, index=None):
        """
        Return the cross halo catalogue if `key` is `None`, otherwise
        return  values from the reference catalogue.

        Parameters
        ----------
        key : str, optional
            Key to get. If `None` return the whole catalogue.
        index : int or array, optional
            Indices to get, if `None` return all.

        Returns
        -------
        out : :py:class:`csiborgtools.read.HaloCatalogue` or array
        """
        if key is None:
            return self._catx
        out = self._catx[key]
        return out if index is None else out[index]

    def __getitem__(self, key):
        """
        Must be one of `index`, `match_indxs`, `ngp_overlap`,
        `smoothed_overlap` or `refmask`.
        """
        assert key in ("index", "match_indxs", "ngp_overlap",
                       "smoothed_overlap", "refmask")
        return self._data[key]

    def __len__(self):
        return self["index"].size


class NPairsOverlap:
    r"""
    A shortcut object for reading in the results of matching a reference
    simulation with many cross simulations.

    Parameters
    ----------
    cat0 : :py:class:`csiborgtools.read.HaloCatalogue`
        Reference simulation halo catalogue.
    catxs : list of :py:class:`csiborgtools.read.HaloCatalogue`
        List of cross simulation halo catalogues.
    fskel : str, optional
        Path to the overlap. By default `None`, i.e.
        `/mnt/extraspace/rstiskalek/csiborg/overlap/cross_{}_{}.npz`.
    min_mass : float, optional
        Minimum :math:`M_{\rm tot} / M_\odot` mass in the reference catalogue.
        By default no threshold.
    max_dist : float, optional
        Maximum comoving distance in the reference catalogue. By default upper
        limit.
    """
    _pairs = None

    def __init__(self, cat0, catxs, fskel=None, min_mass=None, max_dist=None):
        self._pairs = [PairOverlap(cat0, catx, fskel=fskel, min_mass=min_mass,
                                   max_dist=max_dist) for catx in catxs]

    def summed_overlap(self, from_smoothed, verbose=False):
        """
        Summed overlap of each halo in the reference simulation with the cross
        simulations.

        Parameters
        ----------
        from_smoothed : bool
            Whether to use the smoothed overlap or not.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        summed_overlap : 2-dimensional array of shape `(nhalos, ncatxs)`
        """
        out = [None] * len(self)
        for i, pair in enumerate(tqdm(self.pairs) if verbose else self.pairs):
            out[i] = pair.summed_overlap(from_smoothed)
        return numpy.vstack(out).T

    def prob_nomatch(self, from_smoothed, verbose=False):
        """
        Probability of no match for each halo in the reference simulation with
        the cross simulation.

        Parameters
        ----------
        from_smoothed : bool
            Whether to use the smoothed overlap or not.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        prob_nomatch : 2-dimensional array of shape `(nhalos, ncatxs)`
        """
        out = [None] * len(self)
        for i, pair in enumerate(tqdm(self.pairs) if verbose else self.pairs):
            out[i] = pair.prob_nomatch(from_smoothed)
        return numpy.vstack(out).T

    def counterpart_mass(self, from_smoothed, overlap_threshold=0.,
                         in_log=False, mass_kind="totpartmass",
                         return_full=True, verbose=False):
        """
        Calculate the expected counterpart mass of each halo in the reference
        simulation from the crossed simulation.

        Parameters
        -----------
        from_smoothed : bool
            Whether to use the smoothed overlap or not.
        overlap_threshold : float, optional
            Minimum overlap required for a halo to be considered a match. By
            default 0.0, i.e. no threshold.
        in_log : bool, optional
            Whether to calculate the expectation value in log space. By default
            `False`.
        mass_kind : str, optional
            The mass kind whose ratio is to be calculated. Must be a valid
            catalogue key. By default `totpartmass`, i.e. the total particle
            mass associated with a halo.
        return_full : bool, optional
            Whether to return the full results of matching each pair or
            calculate summary statistics by Gaussian averaging.
        verbose : bool, optional
            Verbosity flag. By default `False`.

        Returns
        -------
        mu, std : 1-dimensional arrays of shape `(nhalos,)`
            Summary expected mass and standard deviation from all cross
            simulations.
        mus, stds : 2-dimensional arrays of shape `(nhalos, ncatx)`, optional
            Expected mass and standard deviation from each cross simulation.
            Returned only if `return_full` is `True`.
        """
        mus, stds = [None] * len(self), [None] * len(self)
        for i, pair in enumerate(tqdm(self.pairs) if verbose else self.pairs):
            mus[i], stds[i] = pair.counterpart_mass(
                from_smoothed=from_smoothed,
                overlap_threshold=overlap_threshold, in_log=in_log,
                mass_kind=mass_kind)
        mus, stds = numpy.vstack(mus).T, numpy.vstack(stds).T

        probmatch = 1 - self.prob_nomatch(from_smoothed)  # Prob of > 0 matches
        # Normalise it for weighted sums etc.
        norm_probmatch = numpy.apply_along_axis(
            lambda x: x / numpy.sum(x), axis=1, arr=probmatch)

        # Mean and standard deviation of weighted stacked Gaussians
        mu = numpy.sum(norm_probmatch * mus, axis=1)
        std = numpy.sum(norm_probmatch * (mus**2 + stds**2), axis=1) - mu**2
        std **= 0.5

        if return_full:
            return mu, std, mus, stds
        return mu, std

    @property
    def pairs(self):
        """
        List of `PairOverlap` objects in this reader.

        Returns
        -------
        pairs : list of :py:class:`csiborgtools.read.PairOverlap`
        """
        return self._pairs

    @property
    def cat0(self):
        return self.pairs[0].cat0  # All pairs have the same ref catalogue

    def __len__(self):
        return len(self.pairs)


def binned_resample_mean(x, y, prob, bins, nresample=50, seed=42):
    """
    Calculate binned average of `y` by MC resampling. Each point is kept with
    probability `prob`.

    Parameters
    ----------
    x : 1-dimensional array
        Independent variable.
    y : 1-dimensional array
        Dependent variable.
    prob : 1-dimensional array
        Sample probability.
    bins : 1-dimensional array
        Bin edges to bin `x`.
    nresample : int, optional
        Number of MC resamples. By default 50.
    seed : int, optional
        Random seed.

    Returns
    -------
    bin_centres : 1-dimensional array
        Bin centres.
    stat : 2-dimensional array
        Mean and its standard deviation from MC resampling.
    """
    assert (x.ndim == 1) & (x.shape == y.shape == prob.shape)

    gen = numpy.random.RandomState(seed)

    loop_stat = numpy.full(nresample, numpy.nan)      # Preallocate loop arr
    stat = numpy.full((bins.size - 1, 2), numpy.nan)  # Preallocate output

    for i in range(bins.size - 1):
        mask = (x > bins[i]) & (x <= bins[i + 1])
        nsamples = numpy.sum(mask)

        loop_stat[:] = numpy.nan  # Clear it
        for j in range(nresample):
            loop_stat[j] = numpy.mean(y[mask][gen.rand(nsamples) < prob[mask]])

        stat[i, 0] = numpy.mean(loop_stat)
        stat[i, 1] = numpy.std(loop_stat)

    bin_centres = (bins[1:] + bins[:-1]) / 2

    return bin_centres, stat
