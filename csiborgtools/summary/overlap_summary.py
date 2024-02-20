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

from ..utils import periodic_distance


###############################################################################
#                           Utility functions                             #
###############################################################################

def find_peak(x, weights, shrink=0.95, min_obs=5):
    """
    Find the peak of a 1D distribution using a shrinking window.
    """
    if not shrink < 1:
        raise ValueError("`shrink` must be less than 1.")

    xmin, xmax = numpy.min(x), numpy.max(x)
    xpos = (xmax + xmin) / 2
    rad = (xmax - xmin) / 2

    while True:
        mask = numpy.abs(x - xpos) < rad
        if mask.sum() < min_obs:
            return xpos

        xpos = numpy.average(x[mask], weights=weights[mask])
        rad *= shrink


###############################################################################
#                         Overlap of two simulations                          #
###############################################################################


class PairOverlap:
    r"""
    A shortcut object for reading in the results of matching two simulations.

    Parameters
    ----------
    cat0 : instance of :py:class:`csiborgtools.read.BaseCatalogue`
        Halo catalogue corresponding to the reference simulation.
    catx : instance of :py:class:`csiborgtools.read.BaseCatalogue`
        Halo catalogue corresponding to the cross simulation.
    min_logmass : float
        Minimum halo mass in :math:`\log_{10} M_\odot / h` to consider.
    maxdist : float, optional
        Maximum halo distance in :math:`\mathrm{Mpc} / h` from the centre of
        the high-resolution region. Removes overlaps of haloes outside it.
    """
    _cat0 = None
    _catx = None
    _data = None
    _paths = None

    def __init__(self, cat0, catx, min_logmass, maxdist=None):
        if cat0.simname != catx.simname:
            raise ValueError("The two catalogues must be from the same "
                             "simulation.")

        self._cat0 = cat0
        self._catx = catx
        self._paths = cat0.paths
        self.load(cat0, catx, min_logmass, maxdist)

    def load(self, cat0, catx, paths, min_logmass, maxdist=None):
        r"""
        Load overlap calculation results. Matches the results back to the two
        catalogues in question.

        Parameters
        ----------
        cat0 : instance of :py:class:`csiborgtools.read.BaseCatalogue`
            Halo catalogue corresponding to the reference simulation.
        catx : instance of :py:class:`csiborgtools.read.BaseCatalogue`
            Halo catalogue corresponding to the cross simulation.
        min_logmass : float
            Minimum halo mass in :math:`\log_{10} M_\odot / h` to consider.
        maxdist : float, optional
            Maximum halo distance in :math:`\mathrm{Mpc} / h` from the centre
            of the high-resolution region.

        Returns
        -------
        None
        """
        nsim0 = cat0.nsim
        nsimx = catx.nsim
        paths = cat0.paths

        # We first load in the output files. We need to find the right
        # combination of the reference and cross simulation.
        fname = paths.overlap(cat0.simname, nsim0, nsimx, min_logmass,
                              smoothed=False)
        fname_inv = paths.overlap(cat0.simname, nsimx, nsim0, min_logmass,
                                  smoothed=False)
        if isfile(fname):
            data_ngp = numpy.load(fname, allow_pickle=True)
            to_invert = False
        elif isfile(fname_inv):
            data_ngp = numpy.load(fname_inv, allow_pickle=True)
            to_invert = True
            cat0, catx = catx, cat0
        else:
            raise FileNotFoundError(f"No file found for {nsim0} and {nsimx}.")

        fname_smooth = paths.overlap(cat0.simname, cat0.nsim, catx.nsim,
                                     min_logmass, smoothed=True)
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

        if maxdist is not None:
            dist = cat0.radial_distance(in_initial=False)
            for i in range(len(cat0)):
                if dist[i] > maxdist:
                    match_indxs[i] = numpy.array([], dtype=int)
                    ngp_overlap[i] = numpy.array([], dtype=float)
                    smoothed_overlap[i] = numpy.array([], dtype=float)

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
        out = numpy.zeros(len(overlap), dtype=numpy.float32)

        for i in range(len(overlap)):
            if len(overlap[i]) > 0:
                out[i] = numpy.sum(overlap[i])
        return out

    def dist(self, in_initial, boxsize, norm_kind=None):
        """
        Pair distances of matched halos between the reference and cross
        simulations.

        Parameters
        ----------
        in_initial : bool
            Whether to calculate separation in the initial or final snapshot.
        boxsize : float
            The size of the simulation box.
        norm_kind : str, optional
            The kind of normalisation to apply to the distances.
            Can be `r200c`, `ref_patch` or `sum_patch`.

        Returns
        -------
        dist : array of 1-dimensional arrays of shape `(nhalos, )`
        """
        assert (norm_kind is None or norm_kind in ("r200c", "ref_patch", "sum_patch"))  # noqa
        # Get positions either in the initial or final snapshot
        if in_initial:
            pos0 = self.cat0("lagpatch_coordinates")
            posx = self.catx("lagpatch_coordinates")
        else:
            pos0 = self.cat0("cartesian_pos")
            posx = self.catx("cartesian_pos")

        # Get the normalisation array if applicable
        if norm_kind == "r200c":
            norm = self.cat0("r200c")
        if norm_kind == "ref_patch":
            norm = self.cat0("lagpatch_radius")
        if norm_kind == "sum_patch":
            patch0 = self.cat0("lagpatch_radius")
            patchx = self.catx("lagpatch_radius")
            norm = [None] * len(self)
            for i, ind in enumerate(self["match_indxs"]):
                norm[i] = patch0[i] + patchx[ind]
            norm = numpy.array(norm, dtype=object)

        # Now calculate distances
        dist = [None] * len(self)
        for i, ind in enumerate(self["match_indxs"]):
            dist[i] = periodic_distance(posx[ind, :], pos0[i, :], boxsize)

            if norm_kind is not None:
                dist[i] /= norm[i]
        return numpy.array(dist, dtype=object)

    def mass_ratio(self,  in_log=True, in_abs=True):
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
        mass0, massx = self.cat0("totmass"), self.catx("totmass")

        ratio = [None] * len(self)
        for i, ind in enumerate(self["match_indxs"]):
            ratio[i] = mass0[i] / massx[ind]
            if in_log:
                ratio[i] = numpy.log10(ratio[i])
            if in_abs:
                ratio[i] = numpy.abs(ratio[i])
        return numpy.array(ratio, dtype=object)

    def max_overlap_key(self, key, min_overlap, from_smoothed):
        """
        Calculate the maximum overlap mass of each halo in the reference
        simulation from the cross simulation.

        Parameters
        ----------
        key : str
            Key to the maximum overlap statistic to calculate.
        min_overlap : float
            Minimum pair overlap to consider.
        from_smoothed : bool
            Whether to use the smoothed overlap or not.

        Returns
        -------
        out : 1-dimensional array of shape `(nhalos, )`
        """
        out = numpy.full(len(self), numpy.nan, dtype=numpy.float32)
        y = self.catx(key)
        overlap = self.overlap(from_smoothed)

        for i, match_ind in enumerate(self["match_indxs"]):
            # Skip if no match
            if len(match_ind) == 0:
                continue

            k = numpy.argmax(overlap[i])
            if overlap[i][k] > min_overlap:
                out[i] = y[match_ind][k]

        return out

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


###############################################################################
#                 Support functions for pair overlaps                         #
###############################################################################


def max_overlap_agreement(cat0, catx, min_logmass, maxdist):
    r"""
    Calculate whether for a halo `A` from catalogue `cat0` that has a maximum
    overlap with halo `B` from catalogue `catx` it is also `B` that has a
    maximum overlap with `A`.

    Parameters
    ----------
    cat0 : instance of :py:class:`csiborgtools.read.BaseCatalogue`
        Halo catalogue corresponding to the reference simulation.
    catx : instance of :py:class:`csiborgtools.read.BaseCatalogue`
        Halo catalogue corresponding to the cross simulation.
    min_logmass : float
        Minimum halo mass in :math:`\log_{10} M_\odot / h` to consider.
    maxdist : float, optional
        Maximum halo distance in :math:`\mathrm{Mpc} / h` from the centre
        of the high-resolution region.

    Returns
    -------
    agreement : 1-dimensional array of shape `(nhalos, )`
    """
    kwargs = {"min_logmass": min_logmass, "maxdist": maxdist}
    pair_forward = PairOverlap(cat0, catx, **kwargs)
    pair_backward = PairOverlap(catx, cat0, **kwargs)

    nhalos = len(pair_forward.cat0())
    agreement = numpy.full(nhalos, numpy.nan, dtype=numpy.float32)

    for i in range(nhalos):
        match_indxs_forward = pair_forward["match_indxs"][i]

        if len(match_indxs_forward) == 0:
            continue

        overlap_forward = pair_forward["smoothed_overlap"][i]

        kmax = match_indxs_forward[numpy.argmax(overlap_forward)]
        match_indxs_backward = pair_backward["match_indxs"][kmax]
        overlap_backward = pair_backward["smoothed_overlap"][kmax]

        imatch = match_indxs_backward[numpy.argmax(overlap_backward)]
        agreement[i] = imatch == i

    return agreement


def max_overlap_agreements(cat0, catxs, min_logmass, maxdist, verbose=True):
    """
    Repeat `max_overlap_agreement` for many cross simulations.

    Parameters
    ----------
    ...

    Returns
    -------
    agreements : 2-dimensional array of shape `(ncatxs, nhalos)`
    """
    agreements = [None] * len(catxs)
    desc = "Calculating maximum overlap agreement"
    for i, catx in enumerate(tqdm(catxs, desc=desc, disable=not verbose)):
        agreements[i] = max_overlap_agreement(cat0, catx, min_logmass, maxdist)

    return numpy.asanyarray(agreements)


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
    mu, std : 1-dimensional arrays of shape `(len(x), )`
    """
    mu = numpy.full(x.size, numpy.nan, dtype=numpy.float32)
    std = numpy.full(x.size, numpy.nan, dtype=numpy.float32)

    for i in trange(len(x), disable=not verbose):
        x_, w_ = numpy.asarray(x[i]), numpy.asarray(weights[i])
        mask = w_ > min_weight
        x_ = x_[mask]
        w_ = w_[mask]
        if len(w_) == 0:
            continue
        mu[i] = numpy.average(x_, weights=w_)
        std[i] = numpy.average((x_ - mu[i])**2, weights=w_)**0.5
    return mu, std


###############################################################################
#                  Overlap of many pairs of simulations.                      #
###############################################################################


class NPairsOverlap:
    r"""
    A shortcut object for reading in the results of matching a reference
    simulation with many cross simulations.

    Parameters
    ----------
    cat0 : :py:class:`csiborgtools.read.CSiBORGHaloCatalogue`
        Single reference simulation halo catalogue.
    catxs : list of :py:class:`csiborgtools.read.CSiBORGHaloCatalogue`
        List of cross simulation halo catalogues.
    min_logmass : float
        Minimum log mass of halos to consider.
    verbose : bool, optional
        Verbosity flag for loading the overlap objects.
    """
    _pairs = None

    def __init__(self, cat0, catxs, min_logmass, verbose=True):
        pairs = [None] * len(catxs)
        for i, catx in enumerate(tqdm(catxs, desc="Loading overlap objects",
                                      disable=not verbose)):
            pairs[i] = PairOverlap(cat0, catx, min_logmass)

        self._pairs = pairs

    def max_overlap(self, min_overlap, from_smoothed, verbose=True):
        """
        Calculate maximum overlap of each halo in the reference simulation with
        the cross simulations.

        Parameters
        ----------
        min_overlap : float
            Minimum pair overlap to consider.
        from_smoothed : bool
            Whether to use the smoothed overlap or not.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        max_overlap : 2-dimensional array of shape `(nhalos, ncatxs)`
        """
        def get_max(y_):
            if len(y_) == 0:
                return 0
            out = numpy.max(y_)

            return out if out >= min_overlap else 0

        iterator = tqdm(self.pairs,
                        desc="Calculating maximum overlap",
                        disable=not verbose
                        )
        out = [None] * len(self)
        for i, pair in enumerate(iterator):
            out[i] = numpy.asanyarray([get_max(y_)
                                       for y_ in pair.overlap(from_smoothed)])
        return numpy.vstack(out).T

    def max_overlap_key(self, key, min_overlap, from_smoothed, verbose=True):
        """
        Calculate maximum overlap mass of each halo in the reference
        simulation with the cross simulations.

        Parameters
        ----------
        key : str
            Key to the maximum overlap statistic to calculate.
        min_overlap : float
            Minimum pair overlap to consider.
        from_smoothed : bool
            Whether to use the smoothed overlap or not.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        out : 2-dimensional array of shape `(nhalos, ncatxs)`
        """
        iterator = tqdm(self.pairs,
                        desc=f"Calculating maximum overlap {key}",
                        disable=not verbose
                        )
        out = [None] * len(self)
        for i, pair in enumerate(iterator):
            out[i] = pair.max_overlap_key(key, min_overlap, from_smoothed)

        return numpy.vstack(out).T

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
        iterator = tqdm(self.pairs,
                        desc="Calculating summed overlap",
                        disable=not verbose)
        out = [None] * len(self)
        for i, pair in enumerate(iterator):
            out[i] = pair.summed_overlap(from_smoothed)
        return numpy.vstack(out).T

    def expected_property_single(self, k, key, from_smoothed,  in_log=True):
        ys = [None] * len(self)
        overlaps = [None] * len(self)
        for i, pair in enumerate(self):
            overlap = pair.overlap(from_smoothed)
            if len(overlap[k]) == 0:
                ys[i] = numpy.nan
                overlaps[i] = numpy.nan
                continue
            match_indxs = pair["match_indxs"]
            j = numpy.argmax(overlap[k])

            ys[i] = pair.catx(key)[match_indxs[k][j]]
            if in_log:
                ys[i] = numpy.log10(ys[i])
            overlaps[i] = overlap[k][j]

        return ys, overlaps

    def expected_property(self, key, from_smoothed, min_logmass,
                          in_log=True, mass_kind="totpartmass", verbose=True):
        """
        Calculate the expected counterpart mass of each halo in the reference
        simulation from the crossed simulation.

        Parameters
        ----------
        key : str
            Property key.
        from_smoothed : bool
            Whether to use the smoothed overlap or not.
        min_logmass : float
            Minimum log mass of reference halos to consider.
        in_log : bool, optional
            Whether to calculated the expected property in log10.
        mass_kind : str, optional
            The mass kind whose ratio is to be calculated. Must be a valid
            catalogue key. By default `totpartmass`, i.e. the total particle
            mass associated with a halo.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        mean_expected : 1-dimensional array of shape `(nhalos, )`
            Expected property from all cross simulations.
        std_expected : 1-dimensional array of shape `(nhalos, )`
            Standard deviation of the expected property.
        """
        log_mass0 = numpy.log10(self.cat0(mass_kind))
        ntot = len(log_mass0)
        mean_expected = numpy.full(ntot, numpy.nan, dtype=numpy.float32)
        std_expected = numpy.full(ntot, numpy.nan, dtype=numpy.float32)

        indxs = numpy.where(log_mass0 > min_logmass)[0]
        for i in tqdm(indxs, disable=not verbose,
                      desc="Calculating expectation"):
            ys = numpy.full(len(self), numpy.nan, dtype=numpy.float32)
            weights = numpy.full(len(self), numpy.nan, dtype=numpy.float32)
            for j, pair in enumerate(self):
                overlap = pair.overlap(from_smoothed)
                if len(overlap[i]) == 0:
                    continue

                k = numpy.argmax(overlap[i])
                ys[j] = pair.catx(key)[pair["match_indxs"][i][k]]
                weights[j] = overlap[i][k]

                if in_log:
                    ys[j] = numpy.log10(ys[j])

            mask = numpy.isfinite(ys) & numpy.isfinite(weights)
            if numpy.sum(mask) <= 2:
                continue

            mean_expected[i] = find_peak(ys[mask], weights=weights[mask])
            std_expected[i] = numpy.average((ys[mask] - mean_expected[i])**2,
                                            weights=weights[mask])**0.5
            print(log_mass0[i], mean_expected[i], std_expected[i])

        return mean_expected, std_expected

    @property
    def pairs(self):
        """
        List of `PairOverlap` objects in this reader.

        Returns
        -------
        pairs : list of :py:class:`csiborgtools.summary.PairOverlap`
        """
        return self._pairs

    @property
    def cat0(self):
        return self.pairs[0].cat0  # All pairs have the same ref catalogue

    def __getitem__(self, key):
        if not isinstance(key, int):
            raise TypeError("Key must be an integer.")
        return self.pairs[key]

    def __len__(self):
        return len(self.pairs)


###############################################################################
#                       Various support functions.                            #
###############################################################################


def get_cross_sims(simname, nsim0, paths, min_logmass, smoothed):
    """
    Get the list of cross simulations for a given reference simulation for
    which the overlap has been calculated.

    Parameters
    ----------
    simname : str
        Simulation name.
    nsim0 : int
        Reference simulation number.
    paths : :py:class:`csiborgtools.paths.Paths`
        Paths object.
    min_logmass : float
        Minimum log mass of halos to consider.
    smoothed : bool
        Whether to use the smoothed overlap or not.
    """
    nsimxs = []
    for nsimx in paths.get_ics(simname):
        if nsimx == nsim0:
            continue
        f1 = paths.overlap(simname, nsim0, nsimx, min_logmass, smoothed)
        f2 = paths.overlap(simname, nsimx, nsim0, min_logmass, smoothed)
        if isfile(f1) or isfile(f2):
            nsimxs.append(nsimx)
    return nsimxs
