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
import numpy
import joblib
from tqdm import tqdm
from .make_cat import HaloCatalogue


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


class OverlapReader:
    """
    A shortcut object for reading in the results of matching two simulations.

    Parameters
    ----------
    nsim0 : int
        The reference simulation ID.
    nsimx : int
        The cross simulation ID.
    fskel : str, optional
        Path to the overlap. By default `None`, i.e.
        `/mnt/extraspace/rstiskalek/csiborg/overlap/cross_{}_{}.npz`.
    """
    def __init__(self, nsim0, nsimx, fskel=None):
        if fskel is None:
            fskel = "/mnt/extraspace/rstiskalek/csiborg/overlap/"
            fskel += "cross_{}_{}.npz"
        self._data = numpy.load(fskel.format(nsim0, nsimx), allow_pickle=True)
        self._set_cats(nsim0, nsimx)

    @property
    def nsim0(self):
        """
        The reference simulation ID.

        Returns
        -------
        nsim0 : int
        """
        return self._nsim0

    @property
    def nsimx(self):
        """
        The cross simulation ID.

        Returns
        -------
        nsimx : int
        """
        return self._nsimx

    @property
    def cat0(self):
        """
        The reference halo catalogue.

        Returns
        -------
        cat0 : :py:class:`csiborgtools.read.HaloCatalogue`
        """
        return self._cat0

    @property
    def catx(self):
        """
        The cross halo catalogue.

        Returns
        -------
        catx : :py:class:`csiborgtools.read.HaloCatalogue`
        """
        return self._catx

    def _set_cats(self, nsim0, nsimx):
        """
        Set the simulation IDs and catalogues.

        Parameters
        ----------
        nsim0, nsimx : int
            The reference and cross simulation IDs.

        Returns
        -------
        None
        """
        self._nsim0 = nsim0
        self._nsimx = nsimx
        self._cat0 = HaloCatalogue(nsim0)
        self._catx = HaloCatalogue(nsimx)

    @property
    def indxs(self):
        """
        Indices of halos from the reference catalogue.

        Returns
        -------
        indxs : 1-dimensional array
        """
        return self._data["indxs"]

    @property
    def match_indxs(self):
        """
        Indices of halos from the cross catalogue.

        Returns
        -------
        match_indxs : array of 1-dimensional arrays of shape `(nhalos, )`
        """
        return self._data["match_indxs"]

    @property
    def overlap(self):
        """
        Pair overlap of halos between the reference and cross simulations.

        Returns
        -------
        ovelap : array of 1-dimensional arrays of shape `(nhalos, )`
        """
        return self._data["cross"]

    def dist(self, in_initial, norm=None):
        """
        Final snapshot pair distances.

        Parameters
        ----------
        in_initial : bool
            Whether to calculate separation in the initial or final snapshot.

        Returns
        -------
        dist : array of 1-dimensional arrays of shape `(nhalos, )`
        """
        assert norm is None or norm in ("r200", "ref_patch", "sum_patch")
        # Positions either in the initial or final snapshot
        if in_initial:
            pos0 = self.cat0.positions0
            posx = self.catx.positions0
        else:
            pos0 = self.cat0.positions
            posx = self.catx.positions

        dist = [None] * self.indxs.size
        for n, ind in enumerate(self.match_indxs):
            dist[n] = numpy.linalg.norm(pos0[n, :] - posx[ind, :], axis=1)

            # Normalisation
            if norm == "r200":
                dist[n] /= self.cat0["r200"][n]
            if norm == "ref_patch":
                dist[n] /= self.cat0["lagpatch"][n]
            if norm == "sum_patch":
                dist[n] /= (self.cat0["lagpatch"][n]
                            + self.catx["lagpatch"][ind])
        return numpy.array(dist, dtype=object)

    def mass_ratio(self, mass_kind="totpartmass", in_log=True, in_abs=True):
        """
        Pair mass ratio.

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
        mass0 = self.cat0[mass_kind]
        massx = self.catx[mass_kind]

        ratio = [None] * self.indxs.size
        for n, ind in enumerate(self.match_indxs):
            ratio[n] = mass0[n] / massx[ind]
            if in_log:
                ratio[n] = numpy.log10(ratio[n])
            if in_abs:
                ratio[n] = numpy.abs(ratio[n])
        return numpy.array(ratio, dtype=object)

    def summed_overlap(self):
        """
        Summed overlap of each halo in the reference simulation with the cross
        simulation.

        Parameters
        ----------
        None

        Returns
        -------
        summed_overlap : 1-dimensional array of shape `(nhalos, )`
        """
        return numpy.array([numpy.sum(cross) for cross in self._data["cross"]])

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
        out = [None] * self.indxs.size
        for n, ind in enumerate(self.match_indxs):
            out[n] = numpy.ones(ind.size) * self.cat0[par][n]
        return numpy.array(out, dtype=object)


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
