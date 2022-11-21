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
Functions to read in the particle and clump files.
"""

import numpy
from os.path import join
from tqdm import trange
from sklearn.neighbors import NearestNeighbors
from .readsim import (get_sim_path, read_mmain, get_csiborg_ids,
                      get_maximum_snapshot)
from ..utils import (flip_cols, add_columns)
from ..units import (BoxUnits, cartesian_to_radec)


class HaloCatalogue:
    r"""
    Processed halo catalogue, the data should be calculated in `run_fit_halos`.

    Parameters
    ----------
    n_sim: int
        Initial condition index.
    n_snap: int
        Snapshot index.
    minimum_m500 : float, optional
        The minimum :math:`M_{rm 500c} / M_\odot` mass. By default no
        threshold.
    dumpdir : str, optional
        Path to where files from `run_fit_halos` are stored. By default
        `/mnt/extraspace/rstiskalek/csiborg/`.
    mmain_path : str, optional
        Path to where mmain files are stored. By default
        `/mnt/zfsusers/hdesmond/Mmain`.
    """
    _box = None
    _n_sim = None
    _n_snap = None
    _data = None
    _knn = None
    _positions = None

    def __init__(self, n_sim, n_snap, minimum_m500=None,
                 dumpdir="/mnt/extraspace/rstiskalek/csiborg/",
                 mmain_path="/mnt/zfsusers/hdesmond/Mmain"):
        self._box = BoxUnits(n_snap, get_sim_path(n_sim))
        minimum_m500 = 0 if minimum_m500 is None else minimum_m500
        self._set_data(n_sim, n_snap, dumpdir, mmain_path, minimum_m500)
        self._nsim = n_sim
        self._nsnap = n_snap
        # Initialise the KNN
        knn = NearestNeighbors()
        knn.fit(self.positions)
        self._knn = knn

    @property
    def data(self):
        """
        Halo catalogue.

        Returns
        -------
        cat : structured array
            Catalogue.
        """
        if self._data is None:
            raise ValueError("`data` is not set!")
        return self._data

    @property
    def box(self):
        """
        Box object, useful for change of units.

        Returns
        -------
        box : :py:class:`csiborgtools.units.BoxUnits`
            The box object.
        """
        return self._box

    @property
    def cosmo(self):
        """
        The box cosmology.

        Returns
        -------
        cosmo : `astropy` cosmology object
            Box cosmology.
        """
        return self.box.cosmo

    @property
    def n_snap(self):
        """
        The snapshot ID.

        Returns
        -------
        n_snap : int
            Snapshot ID.
        """
        return self._n_snap

    @property
    def n_sim(self):
        """
        The initiali condition (IC) realisation ID.

        Returns
        -------
        n_sim : int
            The IC ID.
        """
        return self._n_sim

    def _set_data(self, n_sim, n_snap, dumpdir, mmain_path, minimum_m500):
        """
        Loads the data, merges with mmain, does various coordinate transforms.
        """
        # Load the processed data
        fname = "ramses_out_{}_{}.npy".format(
            str(n_sim).zfill(5), str(n_snap).zfill(5))
        data = numpy.load(join(dumpdir, fname))

        # Load the mmain file and add it to the data
        mmain = read_mmain(n_sim, mmain_path)
        data = self.merge_mmain_to_clumps(data, mmain)
        flip_cols(data, "peak_x", "peak_z")

        # Cut on number of particles and finite m200
        data = data[(data["npart"] > 100) & numpy.isfinite(data["m200"])]

        # Unit conversion
        convert_cols = ["m200", "m500", "totpartmass", "mass_mmain",
                        "r200", "r500", "Rs", "rho0",
                        "peak_x", "peak_y", "peak_z"]
        data = self.box.convert_from_boxunits(data, convert_cols)

        # Cut on mass. Note that this is in Msun
        data = data[data["m500"] > minimum_m500]

        # Now calculate spherical coordinates
        d, ra, dec = cartesian_to_radec(data)
        data = add_columns(data, [d, ra, dec], ["dist", "ra", "dec"])

        # Pre-allocate the positions array
        self._positions = numpy.vstack(
            [data["peak_{}".format(p)] for p in ("x", "y", "z")]).T

        self._data = data

    def merge_mmain_to_clumps(self, clumps, mmain):
        """
        Merge columns from the `mmain` files to the `clump` file, matches them
        by their halo index while assuming that the indices `index` in both
        arrays are sorted.

        Parameters
        ----------
        clumps : structured array
            Clumps structured array.
        mmain : structured array
            Parent halo array whose information is to be merged into `clumps`.

        Returns
        -------
        out : structured array
            Array with added columns.
        """
        X = numpy.full((clumps.size, 2), numpy.nan)
        # Mask of which clumps have a mmain index
        mask = numpy.isin(clumps["index"], mmain["index"])

        X[mask, 0] = mmain["mass_cl"]
        X[mask, 1] = mmain["sub_frac"]
        return add_columns(clumps, X, ["mass_mmain", "sub_frac"])

    @property
    def positions(self):
        """
        3D positions of halos.

        Returns
        -------
        X : 2-dimensional array
            Array of shape `(n_halos, 3)`, where the latter axis represents
            `x`, `y` and `z`.
        """
        return self._positions

    def radius_neigbours(self, X, radius):
        """
        Return sorted nearest neigbours within `radius` or `X`.

        Parameters
        ----------
        X : 2-dimensional array
            Array of shape `(n_queries, 3)`, where the latter axis represents
            `x`, `y` and `z`.
        radius : float
            Limiting distance of neighbours.

        Returns
        -------
        dist : list of 1-dimensional arrays
            List of length `n_queries` whose elements are arrays of distances
            to the nearest neighbours.
        knns : list of 1-dimensional arrays
            List of length `n_queries` whose elements are arrays of indices of
            nearest neighbours in this catalogue.
        """
        if not (X.ndim == 2 and X.shape[1] == 3):
            raise TypeError("`X` must be an array of shape `(n_samples, 3)`.")
        # Query the KNN
        return self._knn.radius_neighbors(X, radius, sort_results=True)

    @property
    def keys(self):
        """Catalogue keys."""
        return self.data.dtype.names

    def __getitem__(self, key):
        return self._data[key]


class CombinedHaloCatalogue:
    r"""
    A combined halo catalogue, containing `HaloCatalogue` for each IC
    realisation at the latest redshift.

    Parameters
    ----------
    minimum_m500 : float, optional
        The minimum :math:`M_{rm 500c} / M_\odot` mass. By default no
        threshold.
    dumpdir : str, optional
        Path to where files from `run_fit_halos` are stored. By default
        `/mnt/extraspace/rstiskalek/csiborg/`.
    mmain_path : str, optional
        Path to where mmain files are stored. By default
        `/mnt/zfsusers/hdesmond/Mmain`.
    verbose : bool, optional
        Verbosity flag for reading the catalogues.
    """
    _n_sims = None
    _n_snaps = None
    _cats = None

    def __init__(self, minimum_m500=None,
                 dumpdir="/mnt/extraspace/rstiskalek/csiborg/",
                 mmain_path="/mnt/zfsusers/hdesmond/Mmain", verbose=True):
        # Read simulations and their maximum snapshots
        # NOTE remove this later and take all cats
        self._n_sims = get_csiborg_ids("/mnt/extraspace/hdesmond")[:3]
        n_snaps = [get_maximum_snapshot(get_sim_path(i)) for i in self._n_sims]
        self._n_snaps = numpy.asanyarray(n_snaps)

        cats = [None] * self.N
        for i in trange(self.N) if verbose else range(self.N):
            cats[i] = HaloCatalogue(self._n_sims[i], self._n_snaps[i],
                                    minimum_m500, dumpdir, mmain_path)
        self._cats = cats

    @property
    def N(self):
        """
        Number of IC realisations in this combined catalogue.

        Returns
        -------
        N : int
            Number of catalogues.
        """
        return len(self.n_sims)

    @property
    def n_sims(self):
        """
        IC realisations CSiBORG identifiers.

        Returns
        -------
        ids : 1-dimensional array
            Array of IDs.
        """
        return self._n_sims

    @property
    def n_snaps(self):
        """
        Snapshot numbers corresponding to `self.n_sims`.

        Returns
        -------
        n_snaps : 1-dimensional array
            Array of snapshot numbers.
        """
        return self._n_snaps

    @property
    def cats(self):
        """
        Catalogues associated with this object.

        Returns
        -------
        cats : list of `HaloCatalogue`
            Catalogues.
        """
        return self._cats

    def __getitem__(self, n):
        if n > self.N:
            raise ValueError("Catalogue count is {}, requested catalogue {}."
                             .format(self.N, n))
        return self.cats[n]
