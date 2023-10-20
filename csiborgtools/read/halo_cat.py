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
from h5py import File
from gc import collect

import numpy
from readfof import FoF_catalog
from sklearn.neighbors import NearestNeighbors

from ..utils import (cartesian_to_radec, periodic_distance_two_points,
                     real2redshift)
from .box_units import CSiBORGBox, QuijoteBox
from .paths import Paths
# from .readsim import CSiBORGReader
from .utils import add_columns, cols_to_structured
from ..utils import fprint
from .readsim import make_halomap_dict, load_halo_particles

from collections import OrderedDict


###############################################################################
#                           Base catalogue                                    #
###############################################################################


class BaseCatalogue(ABC):
    """
    Base halo catalogue.
    """
    _simname = None
    _nsim = None
    _nsnap = None
    _catalogue_name = None

    _paths = None
    _box = None

    _data = None
    _observer_location = None
    _observer_velocity = None
    _mass_key = None

    _cache = OrderedDict()
    _cache_maxsize = None
    _catalogue_length = None
    _is_closed = None

    _derived_properties = ["cartesian_pos",
                           "spherical_pos",
                           "dist",
                           "cartesian_redshiftspace_pos",
                           "spherical_redshiftspace_pos",
                           "redshiftspace_dist",
                           "cartesian_vel",
                           "angular_momentum",
                           "particle_offset"
                           ]

    def __init__(self, simname, nsim, nsnap, halo_finder, catalogue_name,
                 paths, mass_key, cache_maxsize=64):
        self.simname = simname
        self.nsim = nsim
        self.nsnap = nsnap
        self.paths = paths
        self.mass_key = mass_key

        fname = self.paths.processed_output(nsim, simname, halo_finder)
        fprint(f"opening `{fname}`.")
        self._data = File(fname, "r")
        self._is_closed = False

        self.cache_maxsize = cache_maxsize
        self.catalogue_name = catalogue_name

    @property
    def simname(self):
        """Simulation name."""
        if self._simname is None:
            raise RuntimeError("`simname` is not set!")
        return self._simname

    @simname.setter
    def simname(self, simname):
        assert isinstance(simname, str)
        self._simname = simname

    @property
    def nsim(self):
        """The IC realisation index."""
        if self._nsim is None:
            raise RuntimeError("`nsim` is not set!")
        return self._nsim

    @nsim.setter
    def nsim(self, nsim):
        assert isinstance(nsim, (int, numpy.integer))
        self._nsim = nsim

    @property
    def nsnap(self):
        """Catalogue's snapshot index."""
        if self._nsnap is None:
            raise RuntimeError("`nsnap` is not set!")
        return self._nsnap

    @nsnap.setter
    def nsnap(self, nsnap):
        assert isinstance(nsnap, (int, numpy.integer))
        self._nsnap = nsnap

    @property
    def catalogue_name(self):
        """Name of the halo catalogue."""
        if self._catalogue_name is None:
            raise RuntimeError("`catalogue_name` is not set!")
        return self._catalogue_name

    @catalogue_name.setter
    def catalogue_name(self, catalogue_name):
        assert isinstance(catalogue_name, str)
        assert catalogue_name in self.data.keys()
        self._catalogue_name = catalogue_name

    @property
    def paths(self):
        """Paths manager."""
        if self._paths is None:
            raise RuntimeError("`paths` is not set!")
        return self._paths

    @paths.setter
    def paths(self, paths):
        assert isinstance(paths, Paths)
        self._paths = paths

    @property
    def box(self):
        """Box object."""
        pass

    @box.setter
    def box(self, box):
        self._box = box

    @property
    def data(self):
        """The HDF5 catalogue."""
        if self._data is None:
            raise RuntimeError("`data` is not set!")
        return self._data

    @property
    def cache_maxsize(self):
        """Maximum size of the cache."""
        if self._cache_maxsize is None:
            raise RuntimeError("`cache_maxsize` is not set!")
        return self._cache_maxsize

    @cache_maxsize.setter
    def cache_maxsize(self, cache_maxsize):
        assert isinstance(cache_maxsize, int)
        self._cache_maxsize = cache_maxsize

    def cache_length(self):
        return len(self._cache)

    @property
    def observer_location(self):
        """Observer location."""
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
        """Observer velocity."""
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

    @property
    def mass_key(self):
        """Mass key of this catalogue."""
        if self._mass_key is None:
            raise RuntimeError("`mass_key` is not set!")
        return self._mass_key

    @mass_key.setter
    def mass_key(self, mass_key):
        if mass_key is None:
            self._mass_key = None
            return

        if mass_key not in self.data[self.catalogue_name].keys():
            raise ValueError(f"Mass key '{mass_key}' is not available.")

        self._mass_key = mass_key

    def halo_particles(self, hid, kind, in_initial=False):
        """
        Load particle information for a given halo. If the halo ID is invalid,
        returns `None`.

        Parameters
        ----------
        hid : int
            Halo ID.
        kind : str
            Must be position or velocity, i.e. either 'pos' or 'vel'.
        in_initial : bool, optional
            Whether to load the initial or final snapshot.

        Returns
        -------
        out : 2-dimensional array
        """
        if hid == 0:
            raise ValueError("ID 0 is reserved for unassigned particles.")

        if kind not in ["pos", "vel"]:
            raise ValueError("`kind` must be either 'pos' or 'vel'.")

        key = f"snapshot_{'initial' if in_initial else 'final'}/{kind}"
        return load_halo_particles(hid, self[key], self["particle_offset"])

    @lru_cache(maxsize=2)
    def knn(self, in_initial):
        r"""
        Periodic kNN object in real space in Cartesian coordinates trained on
        `self["cartesian_pos"]`.

        Parameters
        ----------
        in_initial : bool
            Whether to define the kNN on the initial or final snapshot.

        Returns
        -------
        :py:class:`sklearn.neighbors.NearestNeighbors`
        """
        # TODO improve the caching
        pos = self["lagpatch_pos"] if in_initial else self["cartesian_pos"]
        L = self.box.boxsize
        knn = NearestNeighbors(
            metric=lambda a, b: periodic_distance_two_points(a, b, L))
        knn.fit(pos)
        return knn

#     def nearest_neighbours(self, X, radius, in_initial, knearest=False,
#                            return_mass=False):
#         r"""
#         Return nearest neighbours within `radius` of `X` from this catalogue.
#
#         Parameters
#         ----------
#         X : 2-dimensional array, shape `(n_queries, 3)`
#             Query positions.
#         radius : float or int
#             Limiting distance or number of neighbours, depending on `knearest`.
#         in_initial : bool
#             Find nearest neighbours in the initial or final snapshot.
#         knearest : bool, optional
#             If True, `radius` is the number of neighbours to return.
#         return_mass : bool, optional
#             Return masses of the nearest neighbours.
#
#         Returns
#         -------
#         dist : list of arrays
#             Distances to the nearest neighbours for each query.
#         indxs : list of arrays
#             Indices of nearest neighbours for each query.
#         mass (optional): list of arrays
#             Masses of the nearest neighbours for each query.
#         """
#         if X.shape != (len(X), 3):
#             raise ValueError("`X` must be of shape `(n_samples, 3)`.")
#         if knearest and not isinstance(radius, int):
#             raise ValueError("`radius` must be an integer if `knearest`.")
#         # if return_mass and not mass_key:
#         #     raise ValueError("`mass_key` must be provided if `return_mass`.")
#
#         knn = self.knn(in_initial, subtract_observer=False, periodic=True)
#
#         if knearest:
#             dist, indxs = knn.kneighbors(X, radius)
#         else:
#             dist, indxs = knn.radius_neighbors(X, radius, sort_results=True)
#
#         if not return_mass:
#             return dist, indxs
#
#         mass = [self[self.mass_key][indx] for indx in indxs]
#         return dist, indxs, mass

#     def angular_neighbours(self, X, ang_radius, in_rsp, rad_tolerance=None):
#         r"""
#         Find nearest neighbours within `ang_radius` of query points `X` in the
#         final snaphot. Optionally applies radial distance tolerance, which is
#         expected to be in :math:`\mathrm{cMpc} / h`.
#
#         Parameters
#         ----------
#         X : 2-dimensional array of shape `(n_queries, 2)` or `(n_queries, 3)`
#             Query positions. Either RA/dec in degrees or dist/RA/dec with
#             distance in :math:`\mathrm{cMpc} / h`.
#         in_rsp : bool
#             If True, use redshift space positions of haloes.
#         ang_radius : float
#             Angular radius in degrees.
#         rad_tolerance : float, optional
#             Radial distance tolerance in :math:`\mathrm{cMpc} / h`.
#
#         Returns
#         -------
#         dist : array of 1-dimensional arrays of shape `(n_neighbours,)`
#             Distance of each neighbour to the query point.
#         ind : array of 1-dimensional arrays of shape `(n_neighbours,)`
#             Indices of each neighbour in this catalogue.
#         """
#         assert X.ndim == 2
#
#         # Get positions of haloes in this catalogue
#         if in_rsp:
#             pos = self.redshift_space_position(cartesian=True,
#                                                subtract_observer=True)
#         else:
#             pos = self.position(in_initial=False, cartesian=True,
#                                 subtract_observer=True)
#
#         # Convert halo positions to unit vectors.
#         raddist = numpy.linalg.norm(pos, axis=1)
#         pos /= raddist.reshape(-1, 1)
#
#         # Convert RA/dec query positions to unit vectors. If no radial
#         # distance is provided artificially add it.
#         if X.shape[1] == 2:
#             X = numpy.vstack([numpy.ones_like(X[:, 0]), X[:, 0], X[:, 1]]).T
#             radquery = None
#         else:
#             radquery = X[:, 0]
#         X = radec_to_cartesian(X)
#
#         # Find neighbours
#         knn = NearestNeighbors(metric="cosine")
#         knn.fit(pos)
#         metric_maxdist = 1 - numpy.cos(numpy.deg2rad(ang_radius))
#         dist, ind = knn.radius_neighbors(X, radius=metric_maxdist,
#                                          sort_results=True)
#
#         # Convert cosine difference to angular distance
#         for i in range(X.shape[0]):
#             dist[i] = numpy.rad2deg(numpy.arccos(1 - dist[i]))
#
#         # Apply radial tolerance
#         if rad_tolerance and radquery:
#             for i in range(X.shape[0]):
#                 mask = numpy.abs(raddist[ind[i]] - radquery[i]) < rad_tolerance
#                 dist[i], ind[i] = dist[i][mask], ind[i][mask]
#
#         return dist, ind

#     def filter_data(self, data, bounds):
#         """
#         Filters data based on specified bounds for each key.
#
#         Parameters
#         ----------
#         data : structured array
#             The data to be filtered.
#         bounds : dict
#             A dictionary with keys corresponding to data columns or `dist` and
#             values as a tuple of `(xmin, xmax)`. If `xmin` or `xmax` is `None`,
#             it defaults to negative infinity and positive infinity,
#             respectively.
#
#         Returns
#         -------
#         structured array
#         """
#         for key, (xmin, xmax) in bounds.items():
#             if key == "dist":
#                 pos = numpy.vstack([data[p] - self.observer_location[i]
#                                     for i, p in enumerate("xyz")]).T
#                 values_to_filter = numpy.linalg.norm(pos, axis=1)
#             else:
#                 values_to_filter = data[key]
#
#             min_bound = xmin if xmin is not None else -numpy.inf
#             max_bound = xmax if xmax is not None else numpy.inf
#
#             data = data[(values_to_filter > min_bound)
#                         & (values_to_filter <= max_bound)]
#
#         return data

    def keys(self):
        """Catalogue keys."""
        keys = []

        if "snapshot_final" in self.data.keys():
            for key in self.data["snapshot_final"].keys():
                keys.append(f"snapshot_final/{key}")

        if "snapshot_initial" in self.data.keys():
            for key in self.data["snapshot_initial"].keys():
                keys.append(f"snapshot_initial/{key}")

        for key in self.data[f"{self.catalogue_name}"].keys():
            keys.append(f"{self.catalogue_name}/{key}")

        for key in self._derived_properties:
            keys.append(key)

        return keys

    def __getitem__(self, key):
        if "snapshot" in key:
            return self.data[key]

        # For non-snapshot keys, we cache the results.
        try:
            return self._cache[key]
        except KeyError:
            if key == "cartesian_pos":
                out = numpy.vstack([self["x"], self["y"], self["z"]]).T
            elif key == "spherical_pos":
                out = cartesian_to_radec(
                    self["cartesian_pos"] - self.observer_location)
            elif key == "dist":
                out = numpy.linalg.norm(
                    self["cartesian_pos"] - self.observer_location, axis=1)
            elif key == "cartesian_vel":
                out = numpy.vstack([self["vx"], self["vy"], self["vz"]])
            elif key == "cartesian_redshift_pos":
                out = real2redshift(
                    self["cartesian_pos"], self["cartesian_vel"],
                    self.observer_location, self.observer_velocity, self.box,
                    make_copy=False)
            elif key == "spherical_redshift_pos":
                out = cartesian_to_radec(
                    self["cartesian_redshift_pos"] - self.observer_location)
            elif key == "redshift_dist":
                out = self["cartesian_redshift_pos"]
                out = numpy.linalg.norm(out - self.observer_location, axis=1)
            elif key == "angular_momentum":
                out = numpy.vstack([self["Lx"], self["Ly"], self["Lz"]]).T
            elif key == "particle_offset":
                out = make_halomap_dict(self["snapshot_final/halo_map"][:])
            elif key == "npart":
                halomap = self["particle_offset"]
                out = numpy.zeros(len(halomap), dtype=numpy.int32)
                for i, hid in enumerate(self["index"]):
                    if hid == 0:
                        continue
                    start, end = halomap[hid]
                    out[i] = end - start
            elif key in self.data[self.catalogue_name].keys():
                out = self.data[f"{self.catalogue_name}/{key}"][:]
            else:
                raise KeyError(f"Key '{key}' is not available.")

        # TODO: Enfore the masking somewhere here?
        if self.cache_length() > self.cache_maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = out
        return out

    @property
    def is_closed(self):
        """Whether the HDF5 catalogue is closed."""
        return self._is_closed

    def close(self):
        """Close the HDF5 catalogue file and clear the cache."""
        if not self._is_closed:
            self.data.close()
            self._is_closed = True
        self._cache.clear()
        collect()

    def clear_cache(self):
        """Clear the cache dictionary."""
        self._cache.clear()
        collect()

    def __len__(self):
        if self._catalogue_length is None:
            self._catalogue_length = len(self["index"])
        return self._catalogue_length


###############################################################################
#                        CSiBORG halo catalogue                               #
###############################################################################


class CSiBORGCatalogue(BaseCatalogue):
    r"""
    CSiBORG halo catalogue. Units typically used are:
        - Length: :math:`cMpc / h`
        - Velocity: :math:`km / s`
        - Mass: :math:`M_\odot / h`

    Parameters
    ----------
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.Paths`
        Paths object.
    catalogue_name : str
        Name of the halo catalogue.
    halo_finder : str
        Halo finder name.
    mass_key : str, optional
        Mass key of the catalogue.
    observer_velocity : 1-dimensional array, optional
        Observer's velocity in :math:`\mathrm{km} / \mathrm{s}`.
    """
    def __init__(self, nsim, paths, catalogue_name, halo_finder, mass_key=None,
                 observer_velocity=None, cache_maxsize=64):
        super().__init__("csiborg", nsim,
                         max(paths.get_snapshots(nsim, "csiborg")),
                         halo_finder, catalogue_name, paths, mass_key,
                         cache_maxsize)
        self.box = CSiBORGBox(self.nsnap, self.nsim, self.paths)
        self.observer_location = [338.85, 338.85, 338.85]  # Mpc / h
        self.observer_velocity = observer_velocity

###############################################################################
#                         Quijote halo catalogue                              #
###############################################################################


class QuijoteCatalogue(BaseCatalogue):
    r"""
    Quijote halo catalogue. Units typically are:
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
