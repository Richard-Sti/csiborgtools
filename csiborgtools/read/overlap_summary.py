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
from functools import lru_cache
from os.path import isfile

import numpy
from tqdm import tqdm, trange

###############################################################################
#                         Overlap of two simulations                          #
###############################################################################


class PairOverlap:
    r"""
    A shortcut object for reading in the results of matching two simulations.

    Parameters
    ----------
    cat0 : :py:class:`csiborgtools.read.HaloCatalogue`
        Halo catalogue corresponding to the reference simulation.
    catx : :py:class:`csiborgtools.read.HaloCatalogue`
        Halo catalogue corresponding to the cross simulation.
    paths : py:class`csiborgtools.read.Paths`
        CSiBORG paths object.
    """
    _cat0 = None
    _catx = None
    _data = None

    def __init__(self, cat0, catx, paths):
        self._cat0 = cat0
        self._catx = catx
        self.load(cat0, catx, paths)

    def load(self, cat0, catx, paths):
        """
        Load overlap calculation results. Matches the results back to the two
        catalogues in question.

        Parameters
        ----------
        cat0 : :py:class:`csiborgtools.read.HaloCatalogue`
            Halo catalogue corresponding to the reference simulation.
        catx : :py:class:`csiborgtools.read.HaloCatalogue`
            Halo catalogue corresponding to the cross simulation.
        paths : py:class`csiborgtools.read.Paths`
            CSiBORG paths object.

        Returns
        -------
        None
        """
        nsim0 = cat0.nsim
        nsimx = catx.nsim

        # We first load in the output files. We need to find the right
        # combination of the reference and cross simulation.
        fname = paths.overlap(nsim0, nsimx, smoothed=False)
        fname_inv = paths.overlap(nsimx, nsim0, smoothed=False)
        if isfile(fname):
            data_ngp = numpy.load(fname, allow_pickle=True)
            to_invert = False
        elif isfile(fname_inv):
            data_ngp = numpy.load(fname_inv, allow_pickle=True)
            to_invert = True
            cat0, catx = catx, cat0
        else:
            raise FileNotFoundError(f"No file found for {nsim0} and {nsimx}.")

        fname_smooth = paths.overlap(cat0.nsim, catx.nsim, smoothed=True)
        data_smooth = numpy.load(fname_smooth, allow_pickle=True)

        # Create mapping from halo indices to array positions in the catalogue.
        # In case of the cross simulation use caching for speed.
        hid2ind0 = {hid: i for i, hid in enumerate(cat0["index"])}
        _hid2indx = {hid: i for i, hid in enumerate(catx["index"])}

        @lru_cache(maxsize=8192)
        def hid2indx(hid):
            return _hid2indx[hid]

        # Unpack the overlaps, making sure that their ordering matches the
        # catalogue
        ref_hids = data_ngp["ref_hids"]
        match_hids = data_ngp["match_hids"]
        raw_ngp_overlap = data_ngp["ngp_overlap"]
        raw_smoothed_overlap = data_smooth["smoothed_overlap"]

        match_indxs = [[] for __ in range(len(cat0))]
        ngp_overlap = [[] for __ in range(len(cat0))]
        smoothed_overlap = [[] for __ in range(len(cat0))]
        for i in range(ref_hids.size):
            _matches = numpy.copy(match_hids[i])
            # Read off the orderings from the reference catalogue
            for j, match_hid in enumerate(match_hids[i]):
                _matches[j] = hid2indx(match_hid)

            k = hid2ind0[ref_hids[i]]
            match_indxs[k] = _matches
            ngp_overlap[k] = raw_ngp_overlap[i]
            smoothed_overlap[k] = raw_smoothed_overlap[i]

        match_indxs = numpy.asanyarray(match_indxs, dtype=object)
        ngp_overlap = numpy.asanyarray(ngp_overlap, dtype=object)
        smoothed_overlap = numpy.asanyarray(smoothed_overlap, dtype=object)

        # If needed, we now invert the matches.
        if to_invert:
            match_indxs, ngp_overlap, smoothed_overlap = self._invert_match(
                match_indxs, ngp_overlap, smoothed_overlap, len(catx),)

        self._data = {"match_indxs": match_indxs,
                      "ngp_overlap": ngp_overlap,
                      "smoothed_overlap": smoothed_overlap,
                      }

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
            Size of the cross catalogue.

        Returns
        -------
        inv_match_indxs : array of 1-dimensional arrays
            Inverted match indices.
        ind_ngp_overlap : array of 1-dimensional arrays
            The NGP overlaps corresponding to `inv_match_indxs`.
        ind_smoothed_overlap : array of 1-dimensional arrays
            The smoothed overlaps corresponding to `inv_match_indxs`.
        """
        # 1. Invert the match. Each reference halo has a list of counterparts
        # so loop over those to each counterpart assign a reference halo
        # and at the same time also add the overlaps
        inv_match_indxs = [[] for __ in range(cross_size)]
        inv_ngp_overlap = [[] for __ in range(cross_size)]
        inv_smoothed_overlap = [[] for __ in range(cross_size)]
        for ref_id in range(match_indxs.size):
            iters = zip(match_indxs[ref_id], ngp_overlap[ref_id],
                        smoothed_overlap[ref_id])
            for cross_id, ngp_cross, smoothed_cross in iters:
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
        Calculate summed overlap of each halo in the reference simulation with
        the cross simulation.

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
        return numpy.array([numpy.product(numpy.subtract(1, cross))
                            for cross in overlap])

    def dist(self, in_initial, norm_kind=None):
        """
        Pair distances of matched halos between the reference and cross
        simulations.

        Parameters
        ----------
        in_initial : bool
            Whether to calculate separation in the initial or final snapshot.
        norm_kind : str, optional
            The kind of normalisation to apply to the distances.
            Can be `r200c`, `ref_patch` or `sum_patch`.

        Returns
        -------
        dist : array of 1-dimensional arrays of shape `(nhalos, )`
        """
        assert (norm_kind is None
                or norm_kind in ("r200c", "ref_patch", "sum_patch"))
        # Get positions either in the initial or final snapshot
        pos0 = self.cat0().position(in_initial=in_initial)
        posx = self.catx().position(in_initial=in_initial)

        # Get the normalisation array if applicable
        if norm_kind == "r200c":
            norm = self.cat0("r200c")
        if norm_kind == "ref_patch":
            norm = self.cat0("lagpatch_size")
        if norm_kind == "sum_patch":
            patch0 = self.cat0("lagpatch_size")
            patchx = self.catx("lagpatch_size")
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
        ----------
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
            if len(match_ind) == 0:
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
            out[i] = numpy.ones(len(ind)) * vals[i]
        return numpy.array(out, dtype=object)

    def cat0(self, key=None, index=None):
        """
        Return the reference halo catalogue if `key` is `None`, otherwise
        return  values from the reference catalogue.

        Parameters
        ----------
        key : str, optional
            Key to get. If `None` return the whole catalogue.
        index : int or array, optional
            Indices to get, if `None` return all.

        Returns
        -------
        out : :py:class:`csiborgtools.read.ClumpsCatalogue` or array
        """
        if key is None:
            return self._cat0
        out = self._cat0[key]
        return out if index is None else out[index]

    def catx(self, key=None, index=None):
        """
        Return the cross halo catalogue if `key` is `None`, otherwise
        return  values from the cross catalogue.

        Parameters
        ----------
        key : str, optional
            Key to get. If `None` return the whole catalogue.
        index : int or array, optional
            Indices to get, if `None` return all.

        Returns
        -------
        out : :py:class:`csiborgtools.read.ClumpsCatalogue` or array
        """
        if key is None:
            return self._catx
        out = self._catx[key]
        return out if index is None else out[index]

    def __getitem__(self, key):
        assert key in ["match_indxs", "ngp_overlap", "smoothed_overlap"]
        return self._data[key]

    def __len__(self):
        return self["match_indxs"].size


def weighted_stats(x, weights, min_weight=0, verbose=False):
    """
    Calculate the weighted mean and standard deviation of `x` using `weights`
    for each array of `x`.

    Parameters
    ----------
    x : array of arrays
        Array of arrays of values to calculate the weighted mean and standard
        deviation for.
    weights : array of arrays
        Array of arrays of weights to use for the calculation.
    min_weight : float, optional
        Minimum weight required for a value to be included in the calculation.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    stat : 2-dimensional array of shape `(len(x), 2)`
        The first column is the weighted mean and the second column is the
        weighted standard deviation.
    """
    out = numpy.full((x.size, 2), numpy.nan, dtype=numpy.float32)

    for i in trange(len(x)) if verbose else range(len(x)):
        x_, w_ = numpy.asarray(x[i]), numpy.asarray(weights[i])
        mask = w_ > min_weight
        x_ = x_[mask]
        w_ = w_[mask]
        if len(w_) == 0:
            continue
        out[i, 0] = numpy.average(x_, weights=w_)
        out[i, 1] = numpy.average((x_ - out[i, 0])**2, weights=w_)**0.5
    return out


###############################################################################
#                  Overlap of many pairs of simulations.                      #
###############################################################################


class NPairsOverlap:
    r"""
    A shortcut object for reading in the results of matching a reference
    simulation with many cross simulations.

    Parameters
    ----------
    cat0 : :py:class:`csiborgtools.read.HaloCatalogue`
        Single reference simulation halo catalogue.
    catxs : list of :py:class:`csiborgtools.read.HaloCatalogue`
        List of cross simulation halo catalogues.
    paths : py:class`csiborgtools.read.Paths`
        CSiBORG paths object.
    verbose : bool, optional
        Verbosity flag for loading the overlap objects.
    """
    _pairs = None

    def __init__(self, cat0, catxs, paths, verbose=True):
        pairs = [None] * len(catxs)
        if verbose:
            print("Loading individual overlap objects...", flush=True)
        for i, catx in enumerate(tqdm(catxs) if verbose else catxs):
            pairs[i] = PairOverlap(cat0, catx, paths)

        self._pairs = pairs

    def summed_overlap(self, from_smoothed, verbose=True):
        """
        Calculate summed overlap of each halo in the reference simulation with
        the cross simulations.

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
        if verbose:
            print("Calculating summed overlap...", flush=True)
        for i, pair in enumerate(tqdm(self.pairs) if verbose else self.pairs):
            out[i] = pair.summed_overlap(from_smoothed)
        return numpy.vstack(out).T

    def prob_nomatch(self, from_smoothed, verbose=True):
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
        if verbose:
            print("Calculating probability of no match...", flush=True)
        for i, pair in enumerate(tqdm(self.pairs) if verbose else self.pairs):
            out[i] = pair.prob_nomatch(from_smoothed)
        return numpy.vstack(out).T

    def counterpart_mass(self, from_smoothed, overlap_threshold=0.,
                         in_log=False, mass_kind="totpartmass",
                         return_full=False, verbose=True):
        """
        Calculate the expected counterpart mass of each halo in the reference
        simulation from the crossed simulation.

        Parameters
        ----------
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
            Verbosity flag.

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
        if verbose:
            print("Calculating counterpart masses...", flush=True)
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


###############################################################################
#                       Various support functions.                            #
###############################################################################


def get_cross_sims(nsim0, paths, smoothed):
    """
    Get the list of cross simulations for a given reference simulation for
    which the overlap has been calculated.

    Parameters
    ----------
    nsim0 : int
        Reference simulation number.
    paths : :py:class:`csiborgtools.paths.Paths`
        Paths object.
    smoothed : bool
        Whether to use the smoothed overlap or not.
    """
    nsimxs = []
    for nsimx in paths.get_ics("csiborg"):
        if nsimx == nsim0:
            continue
        f1 = paths.overlap(nsim0, nsimx, smoothed)
        f2 = paths.overlap(nsimx, nsim0, smoothed)
        if isfile(f1) or isfile(f2):
            nsimxs.append(nsimx)
    return nsimxs


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
