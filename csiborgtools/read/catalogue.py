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
"""Simulation catalogues."""
from abc import ABC, abstractproperty
from collections import OrderedDict
from functools import lru_cache
from gc import collect
from itertools import product
from math import floor

import numpy
from sklearn.neighbors import NearestNeighbors

from ..utils import (cartesian_to_radec, great_circle_distance,
                     number_counts, periodic_distance_two_points,
                     real2redshift)


###############################################################################
#                           Base catalogue                                    #
###############################################################################


class BaseCatalogue(ABC):
    """
    Base halo catalogue.
    """
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

    def __init__(self):
        self._simname = None
        self._nsim = None
        self._nsnap = None
        self._catalogue_name = None

        self._paths = None
        self._box = None

        self._data = None
        self._observer_location = None
        self._observer_velocity = None
        self._mass_key = None

        self._cache = OrderedDict()
        self._cache_maxsize = None
        self._catalogue_length = None
        self._is_closed = None
        self._load_filtered = False
        self._filter_mask = None

    def init_with_snapshot(self, simname, nsim, nsnap,
                           paths, bounds,
                           observer_location, observer_velocity,
                           cache_maxsize=64):
        self.simname = simname
        self.nsim = nsim
        self.nsnap = nsnap
        self._paths = paths
        self.observer_location = observer_location
        self.observer_velocity = observer_velocity

        self.cache_maxsize = cache_maxsize

        if bounds is not None:
            self._make_mask(bounds)

    @property
    def simname(self):
        """Simulation name."""
        if self._simname is None:
            raise RuntimeError("`simname` is not set!")
        return self._simname

    @simname.setter
    def simname(self, simname):
        if not isinstance(simname, str):
            raise TypeError("`simname` must be a string.")
        self._simname = simname

    @property
    def nsim(self):
        """The IC realisation index."""
        if self._nsim is None:
            raise RuntimeError("`nsim` is not set!")
        return self._nsim

    @nsim.setter
    def nsim(self, nsim):
        if not isinstance(nsim, (int, numpy.integer)):
            raise TypeError("`nsim` must be an integer.")
        self._nsim = int(nsim)

    @property
    def nsnap(self):
        """Catalogue's snapshot index."""
        if self._nsnap is None:
            raise RuntimeError("`nsnap` is not set!")
        return self._nsnap

    @nsnap.setter
    def nsnap(self, nsnap):
        if not isinstance(nsnap, (int, numpy.integer)):
            raise TypeError("`nsnap` must be an integer.")
        self._nsnap = int(nsnap)

    @property
    def paths(self):
        """Paths manager."""
        if self._paths is None:
            raise RuntimeError("`paths` is not set!")
        return self._paths

    @property
    def boxsize(self):
        """Box size in `cMpc / h`."""
        if self._boxsize is None:
            raise RuntimeError("`boxsize` is not set!")
        return self._boxsize

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

    def cache_keys(self):
        """Keys of the cache dictionary."""
        return list(self._cache.keys())

    def cache_length(self):
        """Length of the cache dictionary."""
        return len(self._cache)

    @abstractproperty
    def coordinates(self):
        """
        Halo coordinates.

        Returns
        -------
        2-dimensional array
        """
        pass

    @abstractproperty
    def velocities(self):
        """
        Halo peculiar velocities.

        Returns
        -------
        2-dimensional array
        """
        pass

    @abstractproperty
    def npart(self):
        """
        Number of particles in a halo.

        Returns
        -------
        1-dimensional array
        """
        pass

    @abstractproperty
    def totmass(self):
        """
        Total particle mass of a halo.

        Returns
        -------
        1-dimensional array
        """
        pass

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

    def halo_mass_function(self, bin_edges, volume, mass_key=None):
        """
        Get the halo mass function.

        Parameters
        ----------
        bin_edges : 1-dimensional array
            Left mass bin edges.
        volume : float
            Volume in :math:`(cMpc / h)^3`.
        mass_key : str, optional
            Mass key of the catalogue.

        Returns
        -------
        x, y, yerr : 1-dimensional arrays
            Mass bin centres, halo number density, and Poisson error.
        """
        if mass_key is None:
            mass_key = self.mass_key

        counts = number_counts(self[mass_key], bin_edges)

        bin_edges = numpy.log10(bin_edges)
        bin_width = bin_edges[1:] - bin_edges[:-1]

        x = (bin_edges[1:] + bin_edges[:-1]) / 2
        y = counts / bin_width / volume
        yerr = numpy.sqrt(counts) / bin_width / volume

        return x, y, yerr

    @lru_cache(maxsize=4)
    def knn(self, in_initial, angular=False):
        r"""
        Periodic kNN object in real space in Cartesian coordinates trained on
        `self["cartesian_pos"]`.

        Parameters
        ----------
        in_initial : bool
            Whether to define the kNN on the initial or final snapshot.
        angular : bool, optional
            Whether to define the kNN in RA/dec.

        Returns
        -------
        :py:class:`sklearn.neighbors.NearestNeighbors`
        """
        if angular:
            assert not in_initial, "Angular kNN not available for initial."
            pos = self["spherical_pos"][:, 1:]
            knn = NearestNeighbors(metric=great_circle_distance)
        else:
            pos = self["lagpatch_pos"] if in_initial else self["cartesian_pos"]
            L = self.box.boxsize
            knn = NearestNeighbors(
                metric=lambda a, b: periodic_distance_two_points(a, b, L))

        knn.fit(pos)
        return knn

    def nearest_neighbours(self, X, radius, in_initial, knearest=False):
        r"""
        Return nearest neighbours within `radius` of `X` from this catalogue.
        Units of `X` are cMpc / h.

        Parameters
        ----------
        X : 2-dimensional array, shape `(n_queries, 3)`
            Query positions.
        radius : float or int
            Limiting distance or number of neighbours, depending on `knearest`.
        in_initial : bool
            Find nearest neighbours in the initial or final snapshot.
        knearest : bool, optional
            If True, `radius` is the number of neighbours to return.
        return_mass : bool, optional
            Return masses of the nearest neighbours.

        Returns
        -------
        dist : list of arrays
            Distances to the nearest neighbours for each query.
        indxs : list of arrays
            Indices of nearest neighbours for each query.
        """
        if knearest and not isinstance(radius, int):
            raise ValueError("`radius` must be an integer if `knearest`.")

        knn = self.knn(in_initial)
        if knearest:
            dist, indxs = knn.kneighbors(X, radius)
        else:
            dist, indxs = knn.radius_neighbors(X, radius, sort_results=True)

        return dist, indxs

    def angular_neighbours(self, X, in_rsp, angular_tolerance,
                           radial_tolerance=None):
        """
        Find nearest angular neighbours of query points. Optionally applies
        radial distance tolerance. Units of `X` are cMpc / h and degrees.

        Parameters
        ----------
        X : 2-dimensional array of shape `(n_queries, 3)`
            Query positions given as distance/RA/dec.
        in_rsp : bool
            Whether to find neighbours in redshift space.
        angular_tolerance : float
            Angular radius in degrees.
        radial_tolerance : float, optional
            Radial tolerance.

        Returns
        -------
        dist : array of 1-dimensional arrays of shape `(n_neighbours,)`
            Distance of each neighbour to the query point.
        ind : array of 1-dimensional arrays of shape `(n_neighbours,)`
            Indices of each neighbour in this catalogue.
        """
        knn = self.knn(in_initial=False, angular=True)
        dist, indxs = knn.radius_neighbors(X[:, 1:],
                                           radius=angular_tolerance,
                                           sort_results=True)

        if radial_tolerance is not None:
            radial_dist = self["redshift_dist"] if in_rsp else self["dist"]
            radial_query = X[:, 0]

            for i in range(X.shape[0]):
                rad_sep = numpy.abs(radial_dist[indxs[i]] - radial_query[i])
                mask = rad_sep < radial_tolerance
                dist[i], indxs[i] = dist[i][mask], indxs[i][mask]

        return dist, indxs

    def _make_mask(self, bounds):
        """Make an internal mask for the catalogue data."""
        self._load_filtered = False

        self._catalogue_length = None
        mask = numpy.ones(len(self), dtype=bool)
        self._catalogue_length = None  # Don't cache the length

        for key, lims in bounds.items():
            values_to_filter = self[f"__{key}"]

            if isinstance(lims, bool):
                mask &= values_to_filter == lims
            else:
                xmin, xmax = lims
                if xmin is not None:
                    mask &= (values_to_filter > xmin)
                if xmax is not None:
                    mask &= (values_to_filter <= xmax)

        self.clear_cache()
        self._filter_mask = mask
        self._load_filtered = True

    def keys(self):
        """Catalogue keys."""
        keys = []
        for key in self.data[f"{self.catalogue_name}"].keys():
            keys.append(f"{self.catalogue_name}/{key}")

        for key in self._derived_properties:
            keys.append(key)

        return keys

    def __getitem__(self, key):
        # For internal calls we don't want to load the filtered data and use
        # the __ prefixed keys. The internal calls are not being cached.
        if key.startswith("__"):
            is_internal = True
            key = key.lstrip("__")
        else:
            is_internal = False

        if not is_internal and key in self.cache_keys():
            return self._cache[key]
        else:
            if key == "cartesian_pos":
                out = self.coordinates
            elif key == "spherical_pos":
                out = cartesian_to_radec(
                    self["__cartesian_pos"] - self.observer_location)
            elif key == "dist":
                out = numpy.linalg.norm(
                    self["__cartesian_pos"] - self.observer_location, axis=1)
            elif key == "cartesian_vel":
                return self.velocities
            elif key == "cartesian_redshift_pos":
                out = real2redshift(
                    self["__cartesian_pos"], self["__cartesian_vel"],
                    self.observer_location, self.observer_velocity, self.box,
                    make_copy=False)
            elif key == "spherical_redshift_pos":
                out = cartesian_to_radec(
                    self["__cartesian_redshift_pos"] - self.observer_location)
            elif key == "redshift_dist":
                out = self["__cartesian_redshift_pos"]
                out = numpy.linalg.norm(out - self.observer_location, axis=1)
            # elif key == "is_main":
            #     out = self["__index"] == self["__parent"]
            elif key == "npart":
                out = self.npart
            elif key == "totmass":
                out = self.totmass
            elif key in self.data[self.catalogue_name].keys():
                out = self.data[f"{self.catalogue_name}/{key}"][:]
            else:
                raise KeyError(f"Key '{key}' is not available.")

        if self._load_filtered and not is_internal and isinstance(out, numpy.ndarray):  # noqa
            out = out[self._filter_mask]

        if not is_internal:
            self._cache[key] = out

        if self.cache_length() > self.cache_maxsize:
            self._cache.popitem(last=False)

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

    def __repr__(self):
        return (f"<{self.__class__.__name__}> "
                f"(nsim = {self.nsim}, nsnap = {self.nsnap}, "
                f"nhalo = {len(self)})")

    def __len__(self):
        if self._catalogue_length is None:
            if self._load_filtered:
                self._catalogue_length = self._filter_mask.sum()
            else:
                self._catalogue_length = len(self["__index"])
        return self._catalogue_length


###############################################################################
#                        CSiBORG halo catalogue                               #
###############################################################################


class CSiBORG1Catalogue(BaseCatalogue):
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
    bounds : dict, optional
        Parameter bounds; keys as parameter names, values as (min, max) or
        a boolean.
    observer_velocity : 1-dimensional array, optional
        Observer's velocity in :math:`\mathrm{km} / \mathrm{s}`.
    cache_maxsize : int, optional
        Maximum number of cached arrays.
    """
    def __init__(self, nsim, paths,
                 bounds=None, observer_velocity=None, cache_maxsize=64):
        super().__init__()
        super().init_with_snapshot(
            "csiborg1", nsim, max(paths.get_snapshots(nsim, "csiborg1")),
            paths, bounds,
            [338.85, 338.85, 338.85], observer_velocity, cache_maxsize)
        self._boxsize = 677.7

    @property
    def coordinates(self):
        raise RuntimeError("TODO")

    @property
    def velocities(self):
        raise RuntimeError("TODO")

    @property
    def npart(self):
        raise RuntimeError("TODO")

    @property
    def totmass(self):
        raise RuntimeError("TODO")


###############################################################################
#                        CSiBORG2 catalogue                                   #
###############################################################################

class CSiBORG2Catalogue(BaseCatalogue):
    """
    z = 0!
    """

    def __init__(self, nsim, paths, ):
        pass


###############################################################################
#                         Quijote halo catalogue                              #
###############################################################################


# class QuijoteCatalogue(BaseCatalogue):
#     r"""
#     Quijote halo catalogue. Units typically are:
#         - Length: :math:`cMpc / h`
#         - Velocity: :math:`km / s`
#         - Mass: :math:`M_\odot / h`
#
#     Parameters
#     ----------
#     nsim : int
#         IC realisation index.
#     paths : py:class`csiborgtools.read.Paths`
#         Paths object.
#     nsnap : int
#         Snapshot index.
#     observer_location : array, optional
#         Observer's location in :math:`\mathrm{Mpc} / h`.
#     bounds : dict
#         Parameter bounds; keys as parameter names, values as (min, max)
#         tuples. Use `dist` for radial distance, `None` for no bound.
#     load_fitted : bool, optional
#         Load fitted quantities from `fit_halos.py`.
#     load_initial : bool, optional
#         Load initial positions from `fit_init.py`.
#     with_lagpatch : bool, optional
#         Load halos with a resolved Lagrangian patch.
#     load_backup : bool, optional
#         Load halos from the backup catalogue that do not have corresponding
#         snapshots.
#     """
#     def __init__(self, nsim, paths, catalogue_name, halo_finder,
#                  mass_key=None, bounds=None, observer_velocity=None,
#                  cache_maxsize=64):
#         super().__init__()
#         super().init_with_snapshot(
#             "quijote", nsim, 4,
#             halo_finder, catalogue_name, paths, mass_key, bounds,
#             [500., 500., 500.,], observer_velocity, cache_maxsize)
#
#         # NOTE watch out about here setting nsim = 0 ?
#         self.box = QuijoteBox(self.nsnap, self.nsim, self.paths)
#         self._bounds = bounds
#
#     def pick_fiducial_observer(self, n, rmax):
#         r"""
#         Select a new fiducial observer in the box.
#
#         Parameters
#         ----------
#         n : int
#             Fiducial observer index.
#         rmax : float
#             Max. distance from the fiducial obs. in :math:`\mathrm{cMpc} / h`.
#         """
#         self.clear_cache()
#         # cat = deepcopy(self)
#         self.observer_location = fiducial_observers(self.box.boxsize, rmax)[n]
#         self.observer_velocity = None
#
#         if self._bounds is None:
#             bounds = {"dist": (0, rmax)}
#         else:
#             bounds = {**self._bounds, "dist": (0, rmax)}
#
#         self._make_mask(bounds)


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
