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
Simulation catalogues:
    - CSiBORG: FoF halo catalogue.
    - Quijote: FoF halo catalogue.
"""
from abc import ABC, abstractproperty
from copy import deepcopy
from functools import lru_cache
from itertools import product
from math import floor

import numpy
from readfof import FoF_catalog
from sklearn.neighbors import NearestNeighbors

from ..utils import (cartesian_to_radec, periodic_distance_two_points,
                     radec_to_cartesian, real2redshift)
from .box_units import CSiBORGBox, QuijoteBox
from .paths import Paths
from .readsim import CSiBORGReader
from .utils import add_columns, cols_to_structured, flip_cols


class BaseCatalogue(ABC):
    """
    Base halo catalogue.
    """
    _data = None
    _paths = None
    _nsim = None
    _observer_location = None
    _observer_velocity = None

    @property
    def nsim(self):
        """
        The IC realisation index.

        Returns
        -------
        int
        """
        if self._nsim is None:
            raise RuntimeError("`nsim` is not set!")
        return self._nsim

    @nsim.setter
    def nsim(self, nsim):
        if not isinstance(nsim, (int, numpy.integer)):
            raise TypeError("`nsim` must be an integer!")
        self._nsim = nsim

    @abstractproperty
    def nsnap(self):
        """
        Catalogue's snapshot index.

        Returns
        -------
        int
        """
        pass

    @abstractproperty
    def simname(self):
        """
        Simulation name.

        Returns
        -------
        str
        """
        pass

    @property
    def paths(self):
        """
        Paths manager.

        Returns
        -------
        :py:class:`csiborgtools.read.Paths`
        """
        if self._paths is None:
            raise RuntimeError("`paths` is not set!")
        return self._paths

    @paths.setter
    def paths(self, paths):
        assert isinstance(paths, Paths)
        self._paths = paths

    @property
    def data(self):
        """
        The catalogue.

        Returns
        -------
        structured array
        """
        if self._data is None:
            raise RuntimeError("`data` is not set!")
        return self._data

    @abstractproperty
    def box(self):
        """
        Box object.

        Returns
        -------
        instance of :py:class:`csiborgtools.units.BoxUnits`
        """
        pass

    def load_initial(self, data, paths, simname):
        """
        Load initial snapshot fits from the script `fit_init.py`.

        Parameters
        ----------
        data : structured array
            The catalogue to which append the new data.
        paths : :py:class:`csiborgtools.read.Paths`
            Paths manager.
        simname : str
            Simulation name.

        Returns
        -------
        structured array
        """
        fits = numpy.load(paths.initmatch(self.nsim, simname, "fit"))
        X, cols = [], []

        for col in fits.dtype.names:
            if col == "index":
                continue
            cols.append(col + "0" if col in ['x', 'y', 'z'] else col)
            X.append(fits[col])

        data = add_columns(data, X, cols)
        for p in ('x0', 'y0', 'z0', 'lagpatch_size'):
            data[p] = self.box.box2mpc(data[p])

        return data

    def load_fitted(self, data, paths, simname):
        """
        Load halo fits from the script `fit_halos.py`.

        Parameters
        ----------
        data : structured array
            The catalogue to which append the new data.
        paths : :py:class:`csiborgtools.read.Paths`
            Paths manager.
        simname : str
            Simulation name.

        Returns
        -------
        structured array
        """
        fits = numpy.load(paths.structfit(self.nsnap, self.nsim, simname))

        cols = [col for col in fits.dtype.names if col != "index"]
        X = [fits[col] for col in cols]
        data = add_columns(data, X, cols)
        box = self.box

        data["r200c"] = box.box2mpc(data["r200c"])

        return data

    def filter_data(self, data, bounds):
        """
        Filters data based on specified bounds for each key.

        Parameters
        ----------
        data : structured array
            The data to be filtered.
        bounds : dict
            A dictionary with keys corresponding to data columns or `dist` and
            values as a tuple of `(xmin, xmax)`. If `xmin` or `xmax` is `None`,
            it defaults to negative infinity and positive infinity,
            respectively.

        Returns
        -------
        structured array
        """
        for key, (xmin, xmax) in bounds.items():
            if key == "dist":
                pos = numpy.vstack([data[p] - self.observer_location[i]
                                    for i, p in enumerate("xyz")]).T
                values_to_filter = numpy.linalg.norm(pos, axis=1)
            else:
                values_to_filter = data[key]

            min_bound = xmin if xmin is not None else -numpy.inf
            max_bound = xmax if xmax is not None else numpy.inf

            data = data[(values_to_filter > min_bound)
                        & (values_to_filter <= max_bound)]

        return data

    @property
    def observer_location(self):
        r"""
        Location of the observer in units :math:`\mathrm{Mpc} / h`.

        Returns
        -------
        1-dimensional array of shape `(3,)`
        """
        if self._observer_location is None:
            raise RuntimeError("`observer_location` is not set!")
        return self._observer_location

    @observer_location.setter
    def observer_location(self, obs_pos):
        assert isinstance(obs_pos, (list, tuple, numpy.ndarray))
        obs_pos = numpy.asanyarray(obs_pos)
        assert obs_pos.shape == (3,)
        self._observer_location = obs_pos

    @property
    def observer_velocity(self):
        r"""
        Velocity of the observer in units :math:`\mathrm{km} / \mathrm{s}`.

        Returns
        1-dimensional array of shape `(3,)`
        """
        if self._observer_velocity is None:
            raise RuntimeError("`observer_velocity` is not set!")
        return self._observer_velocity

    @observer_velocity.setter
    def observer_velocity(self, obs_vel):
        if obs_vel is None:
            self._observer_velocity = None
            return

        assert isinstance(obs_vel, (list, tuple, numpy.ndarray))
        obs_vel = numpy.asanyarray(obs_vel)
        assert obs_vel.shape == (3,)
        self._observer_velocity = obs_vel

    def position(self, in_initial=False, cartesian=True,
                 subtract_observer=False):
        r"""
        Return position components (Cartesian or RA/DEC).

        Parameters
        ----------
        in_initial : bool, optional
            If True, return positions from the initial snapshot, otherwise the
            final snapshot.
        cartesian : bool, optional
            If True, return Cartesian positions. Otherwise, return dist/RA/DEC
            centered at the observer.
        subtract_observer : bool, optional
            If True, subtract the observer's location from the returned
            positions. This is only relevant if `cartesian` is True.

        Returns
        -------
        ndarray, shape `(nobjects, 3)`
        """
        suffix = '0' if in_initial else ''
        component_keys = [f"{comp}{suffix}" for comp in ('x', 'y', 'z')]

        pos = numpy.vstack([self[key] for key in component_keys]).T

        if subtract_observer or not cartesian:
            pos -= self.observer_location

        return cartesian_to_radec(pos) if not cartesian else pos

    def radial_distance(self, in_initial=False):
        r"""
        Distance of haloes from the observer in :math:`\mathrm{cMpc}`.

        Parameters
        ----------
        in_initial : bool, optional
            Whether to calculate in the initial snapshot.

        Returns
        -------
        1-dimensional array of shape `(nobjects,)`
        """
        pos = self.position(in_initial=in_initial, cartesian=True,
                            subtract_observer=True)
        return numpy.linalg.norm(pos, axis=1)

    def velocity(self):
        r"""
        Return Cartesian velocity in :math:`\mathrm{km} / \mathrm{s}`.

        Returns
        -------
        vel : 2-dimensional array of shape `(nobjects, 3)`
        """
        return numpy.vstack([self["v{}".format(p)] for p in ("x", "y", "z")]).T

    def redshift_space_position(self, cartesian=True):
        """
        Calculates the position of objects in redshift space. Positions can be
        returned  in either Cartesian coordinates (default) or spherical
        coordinates (dist/RA/dec).

        Parameters
        ----------
        cartesian : bool, optional
            Returns position in Cartesian coordinates if True, else in
            spherical coordinates.
        subtract_observer : bool, optional
            If True, subtract the observer's location from the returned
            positions.

        Returns
        -------
        2-dimensional array of shape `(nobjects, 3)`
        """
        if self.simname == "quijote":
            raise NotImplementedError("Redshift space positions not "
                                      "implemented for Quijote.")

        rsp = real2redshift(self.position(cartesian=True), self.velocity(),
                            self.observer_location, self.observer_velocity,
                            self.box, make_copy=False)
        if cartesian:
            return rsp

        return cartesian_to_radec(rsp - self.observer_location)

    def angmomentum(self):
        """
        Cartesian angular momentum components of halos in the box coordinate
        system. Likely in box units.

        Returns
        -------
        2-dimensional array of shape `(nobjects, 3)`
        """
        return numpy.vstack([self["L{}".format(p)] for p in ("x", "y", "z")]).T

    @lru_cache(maxsize=2)
    def knn(self, in_initial, subtract_observer, periodic):
        r"""
        kNN object for catalogue objects with caching. Positions are centered
        on the observer.

        Parameters
        ----------
        in_initial : bool
            Whether to define the kNN on the initial or final snapshot.
        subtract_observer : bool
            Whether to subtract the observer's location from the positions.
        periodic : bool
            Whether to use periodic boundary conditions.

        Returns
        -------
        :py:class:`sklearn.neighbors.NearestNeighbors`
        """
        if subtract_observer and periodic:
            raise ValueError("Subtracting observer is not supported for "
                             "periodic boundary conditions.")

        pos = self.position(in_initial=in_initial,
                            subtract_observer=subtract_observer)

        if periodic:
            L = self.box.boxsize
            knn = NearestNeighbors(
                metric=lambda a, b: periodic_distance_two_points(a, b, L))
        else:
            knn = NearestNeighbors()

        knn.fit(pos)
        return knn

    def nearest_neighbours(self, X, radius, in_initial, knearest=False,
                           return_mass=False, mass_key=None):
        r"""
        Return nearest neighbours within `radius` of `X` in a given snapshot.

        Parameters
        ----------
        X : 2D array, shape `(n_queries, 3)`
            Query positions in :math:`\mathrm{cMpc} / h`. Expected to be
            centered on the observer.
        radius : float or int
            Limiting distance or number of neighbours, depending on `knearest`.
        in_initial : bool
            Use the initial or final snapshot for kNN.
        knearest : bool, optional
            If True, `radius` is the number of neighbours to return.
        return_mass : bool, optional
            Return masses of the nearest neighbours.
        mass_key : str, optional
            Mass column key. Required if `return_mass` is True.

        Returns
        -------
        dist : list of arrays
            Distances to the nearest neighbours for each query.
        indxs : list of arrays
            Indices of nearest neighbours for each query.
        mass (optional): list of arrays
            Masses of the nearest neighbours for each query.
        """
        if X.shape != (len(X), 3):
            raise ValueError("`X` must be of shape `(n_samples, 3)`.")
        if knearest and not isinstance(radius, int):
            raise ValueError("`radius` must be an integer if `knearest`.")
        if return_mass and not mass_key:
            raise ValueError("`mass_key` must be provided if `return_mass`.")

        knn = self.knn(in_initial, subtract_observer=False, periodic=True)

        if knearest:
            dist, indxs = knn.kneighbors(X, radius)
        else:
            dist, indxs = knn.radius_neighbors(X, radius, sort_results=True)

        if not return_mass:
            return dist, indxs

        mass = [self[mass_key][indx] for indx in indxs]
        return dist, indxs, mass

    def angular_neighbours(self, X, ang_radius, in_rsp, rad_tolerance=None):
        r"""
        Find nearest neighbours within `ang_radius` of query points `X` in the
        final snaphot. Optionally applies radial distance tolerance, which is
        expected to be in :math:`\mathrm{cMpc} / h`.

        Parameters
        ----------
        X : 2-dimensional array of shape `(n_queries, 2)` or `(n_queries, 3)`
            Query positions. Either RA/dec in degrees or dist/RA/dec with
            distance in :math:`\mathrm{cMpc} / h`.
        in_rsp : bool
            If True, use redshift space positions of haloes.
        ang_radius : float
            Angular radius in degrees.
        rad_tolerance : float, optional
            Radial distance tolerance in :math:`\mathrm{cMpc} / h`.

        Returns
        -------
        dist : array of 1-dimensional arrays of shape `(n_neighbours,)`
            Distance of each neighbour to the query point.
        ind : array of 1-dimensional arrays of shape `(n_neighbours,)`
            Indices of each neighbour in this catalogue.
        """
        assert X.ndim == 2

        # Get positions of haloes in this catalogue
        if in_rsp:
            pos = self.redshift_space_position(cartesian=True,
                                               subtract_observer=True)
        else:
            pos = self.position(in_initial=False, cartesian=True,
                                subtract_observer=True)

        # Convert halo positions to unit vectors.
        raddist = numpy.linalg.norm(pos, axis=1)
        pos /= raddist.reshape(-1, 1)

        # Convert RA/dec query positions to unit vectors. If no radial
        # distance is provided artificially add it.
        if X.shape[1] == 2:
            X = numpy.vstack([numpy.ones_like(X[:, 0]), X[:, 0], X[:, 1]]).T
            radquery = None
        else:
            radquery = X[:, 0]
        X = radec_to_cartesian(X)

        # Find neighbours
        knn = NearestNeighbors(metric="cosine")
        knn.fit(pos)
        metric_maxdist = 1 - numpy.cos(numpy.deg2rad(ang_radius))
        dist, ind = knn.radius_neighbors(X, radius=metric_maxdist,
                                         sort_results=True)

        # Convert cosine difference to angular distance
        for i in range(X.shape[0]):
            dist[i] = numpy.rad2deg(numpy.arccos(1 - dist[i]))

        # Apply radial tolerance
        if rad_tolerance and radquery:
            for i in range(X.shape[0]):
                mask = numpy.abs(raddist[ind[i]] - radquery[i]) < rad_tolerance
                dist[i], ind[i] = dist[i][mask], ind[i][mask]

        return dist, ind

    def keys(self):
        """
        Return catalogue keys.

        Returns
        -------
        keys : list of strings
        """
        return self.data.dtype.names

    def __getitem__(self, key):
        # If key is an integer, return the corresponding row.
        if isinstance(key, (int, numpy.integer)):
            assert key >= 0
        elif key not in self.keys():
            raise KeyError(f"Key '{key}' not in catalogue.")

        return self.data[key]

    def __len__(self):
        return self.data.size


###############################################################################
#                        CSiBORG halo catalogue                               #
###############################################################################


class CSiBORGHaloCatalogue(BaseCatalogue):
    r"""
    CSiBORG FoF halo catalogue with units:
        - Length: :math:`cMpc / h`
        - Velocity: :math:`km / s`
        - Mass: :math:`M_\odot / h`

    Parameters
    ----------
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.Paths`
        Paths object.
    bounds : dict
        Parameter bounds; keys as names, values as (min, max) tuples. Use
        `dist` for radial distance, `None` for no bound.
    load_fitted : bool, optional
        Load fitted quantities.
    load_initial : bool, optional
        Load initial positions.
    with_lagpatch : bool, optional
        Load halos with a resolved Lagrangian patch.
    observer_velocity : 1-dimensional array, optional
        Observer's velocity in :math:`\mathrm{km} / \mathrm{s}`.
    """

    def __init__(self, nsim, paths, bounds={"dist": (0, 155.5)},
                 load_fitted=True, load_initial=True, with_lagpatch=False,
                 observer_velocity=None):
        self.nsim = nsim
        self.paths = paths
        self.observer_location = [338.85, 338.85, 338.85]
        self.observer_velocity = observer_velocity
        reader = CSiBORGReader(paths)
        data = reader.read_fof_halos(self.nsim)
        box = self.box

        # We want coordinates to be [0, 677.7] in Mpc / h
        for p in ('x', 'y', 'z'):
            data[p] = data[p] * box.h + box.box2mpc(1) / 2
        # Similarly mass in units of Msun / h
        data["fof_totpartmass"] *= box.h
        data["fof_m200c"] *= box.h
        # Because of a RAMSES bug, we must flip the x and z coordinates
        flip_cols(data, 'x', 'z')

        if load_initial:
            data = self.load_initial(data, paths, "csiborg")
            flip_cols(data, "x0", "z0")
        if load_fitted:
            data = self.load_fitted(data, paths, "csiborg")
            flip_cols(data, "vx", "vz")

        if load_initial and with_lagpatch:
            data = data[numpy.isfinite(data["lagpatch_size"])]

        if bounds is not None:
            data = self.filter_data(data, bounds)

        self._data = data

    @property
    def nsnap(self):
        return max(self.paths.get_snapshots(self.nsim, "csiborg"))

    @property
    def box(self):
        """
        CSiBORG box object handling unit conversion.

        Returns
        -------
        box : instance of :py:class:`csiborgtools.units.BaseBox`
        """
        return CSiBORGBox(self.nsnap, self.nsim, self.paths)

    @property
    def simname(self):
        return "csiborg"


###############################################################################
#                         Quijote halo catalogue                              #
###############################################################################


class QuijoteHaloCatalogue(BaseCatalogue):
    r"""
    Quijote FoF halo catalogue with units:
        - Length: :math:`cMpc / h`
        - Velocity: :math:`km / s`
        - Mass: :math:`M_\odot / h`

    Parameters
    ----------
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.Paths`
        Paths object.
    nsnap : int
        Snapshot index.
    observer_location : array, optional
        Observer's location in :math:`\mathrm{Mpc} / h`.
    bounds : dict
        Parameter bounds; keys as parameter names, values as (min, max)
        tuples. Use `dist` for radial distance, `None` for no bound.
    load_fitted : bool, optional
        Load fitted quantities from `fit_halos.py`.
    load_initial : bool, optional
        Load initial positions from `fit_init.py`.
    with_lagpatch : bool, optional
        Load halos with a resolved Lagrangian patch.
    load_backup : bool, optional
        Load halos from the backup catalogue that do not have corresponding
        snapshots.
    """
    _nsnap = None
    _origin = None

    def __init__(self, nsim, paths, nsnap,
                 observer_location=[500., 500., 500.],
                 bounds=None, load_fitted=True, load_initial=True,
                 with_lagpatch=False, load_backup=False):
        self.nsim = nsim
        self.paths = paths
        self.nsnap = nsnap
        self.observer_location = observer_location
        # NOTE watch out about here setting nsim = 0
        self._box = QuijoteBox(nsnap, 0, paths)

        fpath = self.paths.fof_cat(nsim, "quijote", load_backup)
        fof = FoF_catalog(fpath, self.nsnap, long_ids=False, swap=False,
                          SFR=False, read_IDs=False)

        cols = [("x", numpy.float32), ("y", numpy.float32),
                ("z", numpy.float32), ("fof_vx", numpy.float32),
                ("fof_vy", numpy.float32), ("fof_vz", numpy.float32),
                ("group_mass", numpy.float32), ("fof_npart", numpy.int32),
                ("index", numpy.int32)]
        data = cols_to_structured(fof.GroupLen.size, cols)

        pos = fof.GroupPos / 1e3
        vel = fof.GroupVel * (1 + self.redshift)
        for i, p in enumerate(["x", "y", "z"]):
            data[p] = pos[:, i]
            data["fof_v" + p] = vel[:, i]
        data["group_mass"] = fof.GroupMass * 1e10
        data["fof_npart"] = fof.GroupLen
        # We want to start indexing from 1. Index 0 is reserved for
        # particles unassigned to any FoF group.
        data["index"] = 1 + numpy.arange(data.size, dtype=numpy.int32)

        if load_backup and (load_initial or load_fitted):
            raise ValueError("Cannot load initial/fitted data for the backup "
                             "catalogues.")

        if load_initial:
            data = self.load_initial(data, paths, "quijote")
        if load_fitted:
            data = self.load_fitted(data, paths, "quijote")

        if load_initial and with_lagpatch:
            data = data[numpy.isfinite(data["lagpatch_size"])]

        if bounds is not None:
            data = self.filter_data(data, bounds)

        self._data = data

    @property
    def nsnap(self):
        """
        Snapshot number.

        Returns
        -------
        int
        """
        return self._nsnap

    @nsnap.setter
    def nsnap(self, nsnap):
        assert nsnap in [0, 1, 2, 3, 4]
        self._nsnap = nsnap

    @property
    def simname(self):
        return "quijote"

    @property
    def redshift(self):
        """
        Redshift of the snapshot.

        Returns
        -------
        float
        """
        return {4: 0.0, 3: 0.5, 2: 1.0, 1: 2.0, 0: 3.0}[self.nsnap]

    @property
    def box(self):
        return self._box

    def pick_fiducial_observer(self, n, rmax):
        r"""
        Return a copy of itself, storing only halos within `rmax` of the new
        fiducial observer.

        Parameters
        ----------
        n : int
            Fiducial observer index.
        rmax : float
            Max. distance from the fiducial obs. in :math:`\mathrm{cMpc} / h`.

        Returns
        -------
        instance of `csiborgtools.read.QuijoteHaloCatalogue`
        """
        cat = deepcopy(self)
        cat.observer_location = fiducial_observers(self.box.boxsize, rmax)[n]
        cat._data = cat.filter_data(cat._data, {"dist": (0, rmax)})
        return cat


###############################################################################
#                     Utility functions for halo catalogues                   #
###############################################################################


def fiducial_observers(boxwidth, radius):
    """
    Compute observer positions in a box, subdivided into spherical regions.

    Parameters
    ----------
    boxwidth : float
        Width of the box.
    radius : float
        Radius of the spherical regions.

    Returns
    -------
    origins : list of lists
        Positions of the observers, with each position as a len-3 list.
    """
    nobs = floor(boxwidth / (2 * radius))
    return [[val * radius for val in position]
            for position in product([1, 3, 5], repeat=nobs)]
