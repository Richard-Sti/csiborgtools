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
Unified interface for simulation catalogues. Currently supports CSiBORG1,
CSiBORG2 and Quijote. For specific implementation always check the relevant
classes in this module.
"""
from abc import ABC, abstractproperty
from collections import OrderedDict
from functools import lru_cache
from gc import collect
from itertools import product
from math import floor

import numpy
from h5py import File
from sklearn.neighbors import NearestNeighbors

from ..params import paths_glamdring
from ..utils import (cartesian_to_radec, great_circle_distance, number_counts,
                     periodic_distance_two_points, real2redshift)
from .paths import Paths
from .snapshot import is_instance_of_base_snapshot_subclass


###############################################################################
#                           Base catalogue                                    #
###############################################################################


class BaseCatalogue(ABC):
    """
    Base halo catalogue.
    """
    _properties = ["cartesian_pos",
                   "spherical_pos",
                   "dist",
                   "cartesian_redshiftspace_pos",
                   "spherical_redshiftspace_pos",
                   "redshiftspace_dist",
                   "cartesian_vel",
                   "particle_offset"
                   "npart",
                   "totmass",
                   "index",
                   "lagpatch_coordinates",
                   "lagpatch_radius"
                   ]

    def __init__(self):
        self._simname = None
        self._nsim = None
        self._nsnap = None
        self._snapshot = None

        self._paths = None

        self._observer_location = None
        self._observer_velocity = None
        self._flip_xz = False
        self._boxsize = None

        self._cache = OrderedDict()
        self._cache_maxsize = None
        self._catalogue_length = None
        self._load_filtered = False
        self._filter_mask = None

        self._custom_keys = []

    def init_with_snapshot(self, simname, nsim, nsnap, paths, snapshot,
                           bounds, boxsize, observer_location,
                           observer_velocity, flip_xz, cache_maxsize=64):
        self.simname = simname
        self.nsim = nsim
        self.nsnap = nsnap
        self._paths = paths
        self.boxsize = boxsize
        self.observer_location = observer_location
        self.observer_velocity = observer_velocity
        self.flip_xz = flip_xz

        self.cache_maxsize = cache_maxsize

        self.snapshot = snapshot

        if bounds is not None:
            self._make_mask(bounds)

    @property
    def simname(self):
        """
        Simulation name.

        Returns
        -------
        str
        """
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
        """
        Simulation IC realisation index.

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
            raise TypeError("`nsim` must be an integer.")
        self._nsim = int(nsim)

    @property
    def nsnap(self):
        """
        Catalogue snapshot index.

        Returns
        -------
        int
        """
        if self._nsnap is None:
            raise RuntimeError("`nsnap` is not set!")
        return self._nsnap

    @nsnap.setter
    def nsnap(self, nsnap):
        if not isinstance(nsnap, (int, numpy.integer)):
            raise TypeError("`nsnap` must be an integer.")
        self._nsnap = int(nsnap)

    @property
    def snapshot(self):
        """
        Corresponding particle snapshot. Can be either the final or initial
        one, depending on `which_snapshot`.

        Returns
        -------
        subclass of py:class:`csiborgtools.read.snapshot.BaseSnapshot`
        """
        if self._snapshot is None:
            raise RuntimeError("`snapshot` is not set!")
        return self._snapshot

    @snapshot.setter
    def snapshot(self, snapshot):
        if snapshot is None:
            self._snapshot = None
            return

        if not is_instance_of_base_snapshot_subclass(snapshot):
            raise TypeError("`snapshot` must be a subclass of `BaseSnapshot`.")

        self._snapshot = snapshot

    @property
    def paths(self):
        """
        Paths manager.

        Returns
        -------
        py:class:`csiborgtools.read.Paths`
        """
        if self._paths is None:
            return Paths(**paths_glamdring)
        return self._paths

    @property
    def boxsize(self):
        """
        Box size in `cMpc / h`.

        Returns
        -------
        float
        """
        if self._boxsize is None:
            raise RuntimeError("`boxsize` is not set!")
        return self._boxsize

    @boxsize.setter
    def boxsize(self, boxsize):
        if not isinstance(boxsize, (int, float)):
            raise TypeError("`boxsize` must be an integer or float.")
        self._boxsize = float(boxsize)

    @property
    def flip_xz(self):
        """
        Whether to flip the x- and z-coordinates to undo the MUSIC bug to match
        observations.

        Returns
        -------
        bool
        """
        return self._flip_xz

    @flip_xz.setter
    def flip_xz(self, flip_xz):
        if not isinstance(flip_xz, bool):
            raise TypeError("`flip_xz` must be a boolean.")
        self._flip_xz = flip_xz

    @property
    def cache_maxsize(self):
        """
        Maximum length of the cache dictionary.

        Returns
        -------
        int
        """
        if self._cache_maxsize is None:
            raise RuntimeError("`cache_maxsize` is not set!")
        return self._cache_maxsize

    @cache_maxsize.setter
    def cache_maxsize(self, cache_maxsize):
        assert isinstance(cache_maxsize, int)
        self._cache_maxsize = cache_maxsize

    def cache_keys(self):
        """
        Current keys of the cache dictionary.

        Parameters
        ----------
        list of str
        """
        return list(self._cache.keys())

    def cache_length(self):
        """
        Current length of the cache dictionary.

        Returns
        -------
        int
        """
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

    @abstractproperty
    def index(self):
        """
        Halo index.

        Returns
        -------
        1-dimensional array
        """
        pass

    @abstractproperty
    def lagpatch_coordinates(self):
        """
        Lagrangian patch coordinates.

        Returns
        -------
        2-dimensional array
        """
        pass

    @abstractproperty
    def lagpatch_radius(self):
        """
        Lagrangian patch radius.

        Returns
        -------
        1-dimensional array
        """
        pass

    @property
    def observer_location(self):
        """
        Observer location.

        Returns
        -------
        1-dimensional array
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
        """
        Observer velocity.

        Returns
        -------
        1-dimensional array
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
            Mass key to get the halo masses.

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
            if in_initial:
                raise ValueError("Angular kNN is not available for initial snapshot.")  # noqa
            pos = self["spherical_pos"][:, 1:]
            knn = NearestNeighbors(metric=great_circle_distance)
        else:
            if in_initial:
                pos = self["lagpatch_coordinates"]
            else:
                pos = self["cartesian_pos"]
            L = self.boxsize
            knn = NearestNeighbors(
                metric=lambda a, b: periodic_distance_two_points(a, b, L))

        knn.fit(pos)
        return knn

    def nearest_neighbours(self, X, radius, in_initial, knearest=False):
        r"""
        Return nearest neighbours within `radius` of `X` from this catalogue.
        Units of `X` are `cMpc / h`.

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
        radial distance tolerance. Units of `X` are `cMpc / h` and degrees.

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
        """
        Make an internal mask for the catalogue data.
        """
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

    def clear_cache(self):
        """
        Clear the cache dictionary.

        Returns
        -------
        None
        """
        self._cache.clear()
        collect()

    def keys(self):
        """
        Catalogue keys.

        Returns
        -------
        list
        """
        return self._properties + self._custom_keys

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
            elif key == "lagpatch_radius":
                out = self.lagpatch_radius
            elif key == "lagpatch_coordinates":
                out = self.lagpatch_coordinates
            elif key == "npart":
                out = self.npart
            elif key == "totmass":
                out = self.totmass
            elif key == "index":
                out = self.index
            elif key in self._custom_keys:
                out = getattr(self, key)
            else:
                raise KeyError(f"Key '{key}' is not available.")

        if self._load_filtered and not is_internal and isinstance(out, numpy.ndarray):  # noqa
            out = out[self._filter_mask]

        if not is_internal:
            self._cache[key] = out

        if self.cache_length() > self.cache_maxsize:
            self._cache.popitem(last=False)

        return out

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
    CSiBORG1 `z = 0` FoF halo catalogue.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.Paths`, optional
        Paths object.
    snapshot : subclass of py:class:`BaseSnapshot`, optional
        Snapshot object corresponding to the catalogue.
    bounds : dict, optional
        Parameter bounds; keys as parameter names, values as (min, max) or
        a boolean.
    observer_velocity : 1-dimensional array, optional
        Observer's velocity in :math:`\mathrm{km} / \mathrm{s}`.
    flip_xz : bool, optional
        Whether to flip the x- and z-coordinates to undo the MUSIC bug to match
        observations.
    cache_maxsize : int, optional
        Maximum number of cached arrays.
    """
    def __init__(self, nsim, paths=None, snapshot=None, bounds=None,
                 observer_velocity=None, flip_xz=True, cache_maxsize=64):
        super().__init__()

        if paths is None:
            paths = Paths(**paths_glamdring)

        super().init_with_snapshot(
            "csiborg1", nsim, max(paths.get_snapshots(nsim, "csiborg1")),
            paths, snapshot, bounds, 677.7, [338.85, 338.85, 338.85],
            observer_velocity, flip_xz, cache_maxsize)

        self._custom_keys = []

    def _read_fof_catalogue(self, kind):
        fpath = self.paths.snapshot_catalogue(self.nsnap, self.nsim,
                                              self.simname)

        with File(fpath, 'r') as f:
            if kind not in f.keys():
                raise ValueError(f"FoF catalogue key '{kind}' not available. Available keys are: {list(f.keys())}")  # noqa
            out = f[kind][...]
        return out

    @property
    def coordinates(self):
        x, y, z = [self._read_fof_catalogue(key) for key in ["x", "y", "z"]]

        if self.flip_xz:
            return numpy.vstack([z, y, x]).T
        else:
            return numpy.vstack([x, y, z]).T

    @property
    def velocities(self):
        raise RuntimeError("Velocities are not available in the FoF catalogue.")  # noqa

    @property
    def npart(self):
        offset = self._read_fof_catalogue("GroupOffset")
        return offset[:, 2] - offset[:, 1]

    @property
    def totmass(self):
        return self._read_fof_catalogue("totpartmass")

    @property
    def index(self):
        return self._read_fof_catalogue("index")

    @property
    def lagpatch_coordinates(self):
        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        data = numpy.load(fpath)

        if self.flip_xz:
            return numpy.vstack([data["z"], data["y"], data["x"]]).T
        else:
            return numpy.vstack([data["x"], data["y"], data["z"]]).T

    @property
    def lagpatch_radius(self):
        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        return numpy.load(fpath)["lagpatch_size"]


###############################################################################
#                        CSiBORG2 catalogue                                   #
###############################################################################

class CSiBORG2Catalogue(BaseCatalogue):
    r"""
    CSiBORG2 FoF halo catalogue.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    nsnap : int
        Snapshot index.
    kind : str
        Simulation kind. Must be one of 'main', 'varysmall', or 'random'.
    paths : py:class`csiborgtools.read.Paths`, optional
        Paths object.
    snapshot : subclass of py:class:`BaseSnapshot`, optional
        Snapshot object corresponding to the catalogue.
    bounds : dict, optional
        Parameter bounds; keys as parameter names, values as (min, max) or
        a boolean.
    observer_velocity : 1-dimensional array, optional
        Observer's velocity in :math:`\mathrm{km} / \mathrm{s}`.
    flip_xz : bool, optional
        Whether to flip the x- and z-coordinates to undo the MUSIC bug to match
        observations.
    cache_maxsize : int, optional
        Maximum number of cached arrays.
    """
    def __init__(self, nsim, nsnap, kind, paths=None, snapshot=None,
                 bounds=None, observer_velocity=None, flip_xz=True,
                 cache_maxsize=64):
        super().__init__()
        super().init_with_snapshot(
            f"csiborg2_{kind}", nsim, nsnap, paths, snapshot, bounds,
            676.6, [338.3, 338.3, 338.3], observer_velocity, flip_xz,
            cache_maxsize)

        self._custom_keys = ["GroupFirstSub", "GroupContamination",
                             "GroupNsubs", "Group_M_Crit200"]

    @property
    def kind(self):
        """
        Simulation kind.

        Returns
        -------
        str
        """
        return self._simname.split("_")[-1]

    def _read_fof_catalogue(self, kind):
        fpath = self.paths.snapshot_catalogue(self.nsnap, self.nsim,
                                              self._simname)

        with File(fpath, 'r') as f:
            grp = f["Group"]
            if kind not in grp.keys():
                raise ValueError(f"Group catalogue key '{kind}' not available. Available keys are: {list(f.keys())}")  # noqa
            out = grp[kind][...]
        return out

    @property
    def coordinates(self):
        out = self._read_fof_catalogue("GroupPos")
        if self.flip_xz:
            out[:, [0, 2]] = out[:, [2, 0]]
        return out

    @property
    def velocities(self):
        out = self._read_fof_catalogue("GroupVel")
        if self.flip_xz:
            out[:, [0, 2]] = out[:, [2, 0]]
        return out

    @property
    def npart(self):
        return self._read_fof_catalogue("GroupLen")

    @property
    def totmass(self):
        return self._read_fof_catalogue("GroupMass") * 1e10

    @property
    def index(self):
        # To grab the size, read some example column.
        nhalo = self._read_fof_catalogue("GroupMass").size
        return numpy.arange(nhalo, dtype=numpy.int32)

    @property
    def lagpatch_coordinates(self):
        if self.nsnap != 99:
            raise RuntimeError("Lagrangian patch information is only "
                               "available for haloes defined at the final "
                               f"snapshot (indexed 99). Chosen {self.nsnap}.")

        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        data = numpy.load(fpath)

        if self.flip_xz:
            return numpy.vstack([data["z"], data["y"], data["x"]]).T
        else:
            return numpy.vstack([data["x"], data["y"], data["z"]]).T

    @property
    def lagpatch_radius(self):
        if self.nsnap != 99:
            raise RuntimeError("Lagrangian patch information is only "
                               "available for haloes defined at the final "
                               f"snapshot (indexed 99). Chosen {self.nsnap}.")

        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        return numpy.load(fpath)["lagpatch_size"]

    @property
    def GroupFirstSub(self):
        return self._read_fof_catalogue("GroupFirstSub")

    @property
    def GroupNsubs(self):
        return self._read_fof_catalogue("GroupNsubs")

    @property
    def Group_M_Crit200(self):
        return self._read_fof_catalogue("Group_M_Crit200")

    @property
    def GroupContamination(self):
        mass_type = self._read_fof_catalogue("GroupMassType")
        return mass_type[:, 5] / (mass_type[:, 1] + mass_type[:, 5])


###############################################################################
#                      CSiBORG2 merger tree reader                            #
###############################################################################


class CSiBORG2MergerTreeReader:
    """
    Merger tree reader for CSiBORG2. Currently supports reading the main branch
    of the most massive FoF member at `z = 99`. Documentation of the merger
    trees can be found at:

        https://www.mtng-project.org/03_output/#descendant-and-progenitor-information       # noqa

    Parameters
    ----------
    nsim : int
        IC realisation index.
    kind : str
        Simulation kind. Must be one of 'main', 'varysmall', or 'random'.
    paths : py:class`csiborgtools.read.Paths`, optional
        Paths object.
    """
    _simname = None
    _nsim = None

    def __init__(self, nsim, kind, paths=None):
        self.simname = f"csiborg2_{kind}"
        self.nsim = nsim
        self._paths = paths

        self._group2treeid = self.make_group_to_treeid()

    @property
    def simname(self):
        """
        Simulation name.

        Returns
        -------
        str
        """
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
        """
        Simulation IC realisation index.

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
            raise TypeError("`nsim` must be an integer.")
        self._nsim = int(nsim)

    @property
    def kind(self):
        """
        Simulation kind.

        Returns
        -------
        str
        """
        return self._simname.split("_")[-1]

    @property
    def paths(self):
        """
        Paths manager.

        Returns
        -------
        py:class:`csiborgtools.read.Paths`
        """
        if self._paths is None:
            return Paths(**paths_glamdring)
        return self._paths

    def make_group_to_treeid(self):
        """
        Create a mapping from group number at snapshot 99 to its tree ID.

        Returns
        -------
        group2treeid : dict
            Dictionary with group number as key and tree ID as value.
        """
        print("Creating group to tree ID mapping...")
        with File(self.paths.trees(self.nsim, self.simname), 'r') as f:
            groupnr = f["TreeHalos/GroupNr"][:]
            snapnum = f["TreeHalos/SnapNum"][:]
            treeid = f["TreeHalos/TreeID"][:]

            # Select only groups in the last snapshot
            mask = snapnum == 99
            groupnr = groupnr[mask]
            treeid = treeid[mask]

            group2treeid = {g: t for g, t in zip(groupnr, treeid)}

        return group2treeid

    def get_tree(self, tree_id):
        """
        Extract a tree from the merger tree file.

        Parameters
        ----------
        tree_id : int
            Tree ID.

        Returns
        -------
        dict
        """
        keys_extract = ["GroupNr",
                        "SnapNum",
                        "TreeFirstHaloInFOFgroup",
                        "TreeMainProgenitor",
                        "TreeNextProgenitor",
                        "TreeProgenitor",
                        "SubhaloMass",
                        ]

        with File(self.paths.trees(self.nsim, self.simname), 'r') as f:
            max_treeid = f["TreeTable/TreeID"][-1]
            if not (0 <= tree_id <= max_treeid):
                raise ValueError(f"Tree ID must be between 0 and {max_treeid}.")  # noqa

            i = f["TreeTable/StartOffset"][tree_id]
            j = i + f["TreeTable/Length"][tree_id]

            tree = {}
            for key in keys_extract:
                tree[key] = f[f"TreeHalos/{key}"][i:j]

            tree["Redshift"] = f["TreeTimes/Redshift"][:]
            tree["Time"] = f["TreeTimes/Time"][:]

        return tree

    def find_first_fof_member(self, group_nr, tree):
        """
        Find the first FOF member of a group at snapshot 99.

        Parameters
        ----------
        group_nr : int
            Group number at snapshot 99.
        tree : dict
            Merger tree.

        Returns
        -------
        int
        """
        fsnap_mask = tree["SnapNum"] == 99
        group_mask = tree["GroupNr"][fsnap_mask] == group_nr
        first_fof = tree["TreeFirstHaloInFOFgroup"][fsnap_mask][group_mask]

        if not numpy.all(first_fof == first_fof[0]):
            raise RuntimeError("We get non-unique first FOF group.")

        return first_fof[0]

    def main_progenitor(self, group_nr):
        """
        Return the mass and redshift of the main progenitor of a group.

        Parameters
        ----------
        group_nr : int
            Group number at snapshot 99.

        Returns
        -------
        dict
        """
        tree_id = self._group2treeid[group_nr]
        tree = self.get_tree(tree_id)
        first_fof = self.find_first_fof_member(group_nr, tree)

        n = first_fof  # Index of the current main progenitor

        time, redshift, main_progenitor_mass = [], [], []
        max_next_progenitor_mass, total_next_progenitor_mass = [], []
        while True:
            # First off attempt to find the next progenitors of the current
            # halo. Deal with the main progenitor later.
            next_prog = tree["TreeNextProgenitor"][n]
            if next_prog != -1:
                minors = []
                while True:
                    minors.append(tree["SubhaloMass"][next_prog])

                    next_prog = tree["TreeNextProgenitor"][next_prog]

                    if next_prog == -1:
                        break
            else:
                # Fiducially set it to zero.
                minors = [0]

            # Update data with information from the current main progenitor.
            major = tree["SubhaloMass"][n]
            main_progenitor_mass.append(major)
            max_next_progenitor_mass.append(max(minors))
            total_next_progenitor_mass.append(sum(minors))
            redshift.append(tree["Redshift"][tree["SnapNum"][n]])
            time.append(tree["Time"][tree["SnapNum"][n]])

            # Update n to the next main progenitor.
            n = tree["TreeMainProgenitor"][n]

            if n == -1:
                break

        return {"Time": numpy.array(time),
                "Redshift": numpy.array(redshift),
                "MainProgenitorMass": numpy.array(main_progenitor_mass) * 1e10,
                "MaxNextProgenitorMass": numpy.array(max_next_progenitor_mass) * 1e10,  # noqa
                "TotalNextProgenitorMass": numpy.array(total_next_progenitor_mass) * 1e10,  # noqa
                }


class CSiBORG2SUBFINDCatalogue(BaseCatalogue):
    r"""
    CSiBORG2 SUBFIND halo catalogue.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    nsnap : int
        Snapshot index.
    kind : str
        Simulation kind. Must be one of 'main', 'varysmall', or 'random'.
    paths : py:class`csiborgtools.read.Paths`, optional
        Paths object.
    bounds : dict, optional
        Parameter bounds; keys as parameter names, values as (min, max) or
        a boolean.
    flip_xz : bool, optional
        Whether to flip the x- and z-coordinates to undo the MUSIC bug to match
        observations.
    cache_maxsize : int, optional
        Maximum number of cached arrays.
    """
    def __init__(self, nsim, nsnap, kind, paths=None,
                 bounds=None, flip_xz=True, cache_maxsize=64):
        # TODO: finish all this!
        super().__init__()
        super().init_with_snapshot(
            f"csiborg2_{kind}", nsim, nsnap, paths, None, bounds,
            676.6, [338.3, 338.3, 338.3], None, flip_xz,
            cache_maxsize)

        self._custom_keys = ["SubhaloSpin", "SubhaloVelDisp", "Central"]

    @property
    def kind(self):
        """
        Simulation kind.

        Returns
        -------
        str
        """
        return self._simname.split("_")[-1]

    def _read_subfind_catalogue(self, kind):
        fpath = self.paths.snapshot_catalogue(self.nsnap, self.nsim,
                                              self._simname)

        with File(fpath, 'r') as f:
            grp = f["Subhalo"]
            if kind not in grp.keys():
                raise ValueError(f"Subhalo catalogue key '{kind}' not available. Available keys are: {list(f.keys())}")  # noqa
            out = grp[kind][...]
        return out

    @property
    def coordinates(self):
        out = self._read_subfind_catalogue("SubhaloPos")
        if self.flip_xz:
            out[:, [0, 2]] = out[:, [2, 0]]
        return out

    @property
    def velocities(self):
        out = self._read_subfind_catalogue("SubhaloVel")
        if self.flip_xz:
            out[:, [0, 2]] = out[:, [2, 0]]
        return out

    @property
    def npart(self):
        return self._read_subfind_catalogue("SubhaloLen")

    @property
    def totmass(self):
        return self._read_subfind_catalogue("SubhaloMass") * 1e10

    @property
    def index(self):
        return numpy.arange(self.totmass.size, dtype=numpy.int32)

    @property
    def lagpatch_coordinates(self):
        raise RuntimeError("Lagrangian patch information is not available for "
                           "SUBFIND haloes.")

    @property
    def lagpatch_radius(self):
        raise RuntimeError("Lagrangian patch information is not available for "
                           "SUBFIND haloes.")

    @property
    def SubhaloSpin(self):
        return self._read_subfind_catalogue("SubhaloSpin")

    @property
    def SubhaloVelDisp(self):
        return self._read_subfind_catalogue("SubhaloVelDisp")

    @property
    def SubhaloContamination(self):
        mass_type = self._read_subfind_catalogue("SubhaloMassType")
        return mass_type[:, 5] / (mass_type[:, 1] + mass_type[:, 5])

    @property
    def Central(self):
        return self._read_subfind_catalogue("SubhaloRankInGr") == 0


###############################################################################
#                         Quijote halo catalogue                              #
###############################################################################


class QuijoteCatalogue(BaseCatalogue):
    r"""
    Quijote `z = 0` halo catalogue.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.Paths`, optional
        Paths object.
    snapshot : subclass of py:class:`BaseSnapshot`, optional
        Snapshot object corresponding to the catalogue.
    bounds : dict
        Parameter bounds; keys as parameter names, values as (min, max)
        tuples. Use `dist` for radial distance, `None` for no bound.
    observer_velocity : array, optional
        Observer's velocity in :math:`\mathrm{km} / \mathrm{s}`.
    cache_maxsize : int, optional
        Maximum number of cached arrays.
    """
    def __init__(self, nsim, paths=None, snapshot=None, bounds=None,
                 observer_velocity=None, cache_maxsize=64):
        super().__init__()
        super().init_with_snapshot(
            "quijote", nsim, 4, paths, snapshot, bounds, 1000,
            [500., 500., 500.,], observer_velocity, False, cache_maxsize)

        self._custom_keys = []
        self._bounds = bounds

    def _read_fof_catalogue(self, kind):
        fpath = self.paths.snapshot_catalogue(self.nsnap, self.nsim,
                                              self.simname)

        with File(fpath, 'r') as f:
            if kind not in f.keys():
                raise ValueError(f"FoF catalogue key '{kind}' not available. Available keys are: {list(f.keys())}")  # noqa
            out = f[kind][...]
        return out

    @property
    def coordinates(self):
        return numpy.vstack([self._read_fof_catalogue(key)
                             for key in ["x", "y", "z"]]).T

    @property
    def velocities(self):
        return numpy.vstack([self._read_fof_catalogue(key)
                             for key in ["vx", "vy", "vz"]]).T

    @property
    def npart(self):
        offset = self._read_fof_catalogue("GroupOffset")
        return offset[:, 2] - offset[:, 1]

    @property
    def totmass(self):
        return self._read_fof_catalogue("group_mass")

    @property
    def index(self):
        return self._read_fof_catalogue("index")

    @property
    def lagpatch_coordinates(self):
        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        data = numpy.load(fpath)
        return numpy.vstack([data["x"], data["y"], data["z"]]).T

    @property
    def lagpatch_radius(self):
        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        return numpy.load(fpath)["lagpatch_size"]

    def pick_fiducial_observer(self, n, rmax):
        r"""
        Select a new fiducial observer in the box.

        Parameters
        ----------
        n : int
            Fiducial observer index.
        rmax : float
            Max. distance from the fiducial obs. in :math:`\mathrm{cMpc} / h`.
        """
        self.clear_cache()
        self.observer_location = fiducial_observers(self.box.boxsize, rmax)[n]
        self.observer_velocity = None

        if self._bounds is None:
            bounds = {"dist": (0, rmax)}
        else:
            bounds = {**self._bounds, "dist": (0, rmax)}

        self._make_mask(bounds)


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
