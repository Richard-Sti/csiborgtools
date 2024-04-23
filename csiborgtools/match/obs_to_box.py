# Copyright (C) 2024 Richard Stiskalek
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
Code to match observations to a constrained simulation.
"""
from abc import ABC

import numpy as np
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy.interpolate import interp1d
from sklearn.neighbors import NearestNeighbors
from tqdm import trange

###############################################################################
#                     Matching probability class                              #
###############################################################################


class BaseMatchingProbability(ABC):
    """Base class for `MatchingProbability`."""

    @property
    def halo_pos(self):
        """
        Halo positions in the constrained simulation.

        Returns
        -------
        2-dimensional array of shape `(n, 3)`
        """
        return self._halo_pos

    @halo_pos.setter
    def halo_pos(self, x):
        if not isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 3:
            raise ValueError("Invalid halo positions.")
        self._halo_pos = x

    @property
    def halo_log_mass(self):
        """
        Halo log mass in the constrained simulation.

        Returns
        -------
        1-dimensional array of shape `(n,)`
        """
        return self._halo_log_mass

    @halo_log_mass.setter
    def halo_log_mass(self, x):
        if not isinstance(x, np.ndarray) and x.ndim == 1 and len(x) != len(self.halo_pos):  # noqa
            raise ValueError("Invalid halo log mass.")
        self._halo_log_mass = x

    @property
    def nhalo(self):
        """"
        Number of haloes in the constrained simulation that are used for
        matching.

        Returns
        -------
        int
        """
        return self.halo_log_mass.size

    def HMF(self, log_mass):
        """
        Evaluate the halo mass function at a given mass.

        Parameters
        ----------
        log_mass : float
            Logarithmic mass of the halo in `Msun / h`.

        Returns
        -------
        HMF : float
            The HMF in `h^3 Mpc^-3 dex^-1`.
        """
        return self._hmf(log_mass)


class MatchingProbability(BaseMatchingProbability):
    """"
    Matching probability by calculating the CDF of finding a halo of a certain
    mass at a given distance from a reference point. Calibrated against a HMF,
    by assuming that the haloes are uniformly distributed. This is only
    approximate treatment, as the haloes are not uniformly distributed, however
    it is sufficient for the present purposes.

    NOTE: The method currently does not account for uncertainty in distance.

    Parameters
    ----------
    halo_pos : 2-dimensional array of shape `(n, 3)`
        Halo positions in the constrained simulation in `Mpc / h`.
    halo_log_mass : 1-dimensional array of shape `(n,)`
        Halo log mass in the constrained simulation in `Msun / h`.
    mdef : str, optional
        Definition of the halo mass. Default is 'fof'.
    cosmo_params : dict, optional
        Cosmological parameters of the constrained simulation.
    """
    def __init__(self, halo_pos, halo_log_mass, mdef="fof",
                 cosmo_params={'flat': True, 'H0': 67.66, 'Om0': 0.3111,
                               'Ob0': 0.0489, 'sigma8': 0.8101, 'ns': 0.9665}):
        self.halo_pos = halo_pos
        self.halo_log_mass = halo_log_mass

        # Define the kNN object and fit it to the halo positions, so that we
        # can quickly query distances to an arbitrary point.
        self._knn = NearestNeighbors()
        self._knn.fit(halo_pos)

        # Next, get the HMF from colossus and create its interpolant.
        cosmology.addCosmology("myCosmo", **cosmo_params)
        cosmology.setCosmology("myCosmo")

        x = np.logspace(10, 16, 10000)
        y = mass_function.massFunction(
            x, 0.0, mdef=mdef, model="angulo12", q_out="dndlnM") * np.log(10)
        self._hmf = interp1d(np.log10(x), y, kind="cubic")

    def pdf(self, r, log_mass):
        """
        Calculate the PDF of finding a halo of a given mass at a given distance
        from a random point.

        Parameters
        ----------
        r : float
            Distance from the random point in `Mpc / h`.
        log_mass : float
            Logarithmic mass of the halo in `Msun / h`.

        Returns
        -------
        float
        """
        nd = self.HMF(log_mass)
        return 4 * np.pi * r**2 * nd * np.exp(-4 / 3 * np.pi * r**3 * nd)

    def cdf(self, r, log_mass):
        """
        Calculate the CDF of finding a halo of a given mass at a given distance
        from a random point.

        Parameters
        ----------
        r : float
            Distance from the random point in `Mpc / h`.
        log_mass : float
            Logarithmic mass of the halo in `Msun / h`.

        Returns
        -------
        float
        """
        nd = self.HMF(log_mass)
        return 1 - np.exp(-4 / 3 * np.pi * r**3 * nd)

    def inverse_cdf(self, cdf, log_mass):
        """
        Calculate the inverse CDF of finding a halo of a given mass at a given
        distance from a random point.

        Parameters
        ----------
        cdf : float
            CDF of finding a halo of a given mass at a given distance.
        log_mass : float
            Logarithmic mass of the halo in `Msun / h`.

        Returns
        -------
        float
        """
        nd = self.HMF(log_mass)
        return (np.log(1 - cdf) / (-4 / 3 * np.pi * nd))**(1 / 3)

    def cdf_per_halo(self, refpos, ref_log_mass=None, rmax=50,
                     return_full=True):
        """
        Calculate the CDF per each halo in the constrained simulation.

        Parameters
        ----------
        refpos : 1-dimensional array of shape `(3,)`
            Reference position in `Mpc / h`.
        ref_log_mass : float, optional
            Reference log mass, used to calculate the difference in log mass
            between the reference and each halo.
        rmax : float, optional
            Maximum distance from the reference point to consider. Below this,
            the CDF is simply set to 1.
        return_full : bool, optional
            If `True`, return the CDF, dlogmass and indxs for all haloes,
            otherwise return only the haloes within `rmax`.

        Returns
        -------
        cdf : 1-dimensional array of shape `(nhalo,)`
            CDF per halo.
        dlogmass : 1-dimensional array of shape `(nhalo,)`
            Difference in log mass between the reference and each halo.
        indxs : 1-dimensional array of shape `(nhalo,)`
            Indices of the haloes.
        """
        if not (isinstance(refpos, np.ndarray) and refpos.ndim == 1):
            raise ValueError("Invalid reference position.")
        if ref_log_mass is not None and not isinstance(ref_log_mass, (float, int, np.float32, np.float64)):  # noqa
            raise ValueError("Invalid reference log mass.")

        # Use the kNN  to pick out the haloes within `rmax` of the reference
        # point.
        dist, indxs = self._knn.radius_neighbors(
            refpos.reshape(-1, 3), rmax, return_distance=True)
        dist, indxs = dist[0], indxs[0]

        cdf_ = self.cdf(dist, self.halo_log_mass[indxs])
        if ref_log_mass is not None:
            dlogmass_ = self.halo_log_mass[indxs] - ref_log_mass
        else:
            dlogmass_ = None

        if return_full:
            cdf = np.ones(self.nhalo)
            cdf[indxs] = cdf_
            if ref_log_mass is not None:
                dlogmass = np.full(self.nhalo, np.infty)
                dlogmass[indxs] = dlogmass_
            else:
                dlogmass = dlogmass_

            indxs = np.arange(self.nhalo)
        else:
            cdf, dlogmass = cdf_, dlogmass_

        return cdf, dlogmass, indxs

    def match_halo(self, refpos, ref_log_mass, pvalue_threshold=0.005,
                   max_absdlogmass=1., rmax=50, verbose=True,
                   catalogue_index=0):
        """
        Match a halo in the constrained simulation to a reference halo.
        Considers match the highest significance halo within `rmax` and
        within `max_absdlogmass` of the reference halo mass. In case of no
        match, returns `None`.

        Parameters
        ----------
        refpos : 1-dimensional array of shape `(3,)`
            Reference position.
        ref_log_mass : float
            Reference log mass.
        pvalue_threshold : float, optional
            Threshold for the CDF to be considered a match.
        max_absdlogmass : float, optional
            Maximum difference in log mass between the reference and the
            matched halo.
        rmax : float, optional
            Maximum distance from the reference point to consider.
        verbose : bool, optional
            If `True`, print information about the match.
        catalogue_index : int, optional
            Optional catalogue index for more informative printing.

        Returns
        -------
        cdf : float, or None
            CDF of the matched halo (significance), if any.
        index : int, or None
            Index of the matched halo, if any.
        """
        cdf, dlogmass, indxs = self.cdf_per_halo(
            refpos, ref_log_mass, rmax, return_full=False)

        dlogmass = np.abs(dlogmass)
        ks = np.argsort(cdf)
        cdf, dlogmass, indxs = cdf[ks], dlogmass[ks], indxs[ks]

        matches = np.where(
            (cdf < pvalue_threshold) & (dlogmass < max_absdlogmass))[0]

        if len(matches) == 0:
            return None, None

        if verbose and len(matches) > 1:
            print(f"Found {len(matches)} plausible matches in catalogue {catalogue_index}.")  # noqa
            for i, k in enumerate(matches):
                j = indxs[k]
                logM = self.halo_log_mass[j]
                dx = np.linalg.norm(self.halo_pos[j] - refpos)
                print(f"    {i + 1}: CDF = {cdf[k]:.3e}, index = {j}, logM = {logM:.3f} Msun / h, dx = {dx:.3f} Mpc / h.")  # noqa

            print(flush=True)

        k = matches[0]
        return cdf[k], indxs[k]


class MatchCatalogues:
    """
    A wrapper for `MatchingProbability` that allows to match observed clusters
    to haloes in multiple catalogues.

    Parameters
    ----------
    catalogues : list
        List of halo catalogues of constrained simulations.
    cosmo_params : dict, optional
        Cosmological parameters of the constrained simulation to calculate
        the corresponding FOF mass function.
    """
    def __init__(self, catalogues,
                 cosmo_params={'flat': True, 'H0': 67.66, 'Om0': 0.3111,
                               'Ob0': 0.0489, 'sigma8': 0.8101, 'ns': 0.9665}):
        mdef = "fof"
        self._catalogues = catalogues
        self._prob_models = [None] * len(catalogues)

        for i in trange(len(catalogues)):
            pos = catalogues[i]["cartesian_pos"]
            log_mass = np.log10(catalogues[i]["totmass"])

            self._prob_models[i] = MatchingProbability(
                pos, log_mass, mdef, cosmo_params)

    def __getitem__(self, index):
        return self._prob_models[index]

    def __len__(self):
        return len(self._catalogues)

    def __call__(self, refpos, ref_log_mass, pvalue_threshold=0.05,
                 max_absdlogmass=1., rmax=50, verbose=True):
        """
        Calculate the CDFs of finding a halo of a certain mass at a given
        distance from a reference point for all catalogues.

        Parameters
        ----------
        refpos : 1-dimensional array of shape `(3,)`
            Reference position.
        ref_log_mass : float
            Reference log mass.
        pvalue_threshold : float, optional
            Threshold for the CDF to be considered a match.
        max_absdlogmass : float, optional
            Maximum difference in log mass between the reference and the
            matched halo.
        rmax : float, optional
            Maximum distance from the reference point to consider.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        cdfs : dict
            Dictionary of CDFs per halo, with keys being the simulation
            indices.
        indxs : dict
            Dictionary of indices of the matched haloes, with keys being the
            simulation indices.
        """
        cdfs, indxs = {}, {}
        for i in trange(len(self), desc="Matching catalogues",
                        disable=not verbose):
            cdf, indx = self._prob_models[i].match_halo(
                refpos, ref_log_mass, pvalue_threshold, max_absdlogmass, rmax,
                verbose, i)

            if cdf is not None:
                cdfs[i] = cdf
                indxs[i] = indx

        n = len(self) - len(cdfs)
        if n > 0 and verbose:
            print(f"Failed to assign haloes in {n} catalogues.")

        return cdfs, indxs
