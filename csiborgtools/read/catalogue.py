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
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import cached_property, lru_cache
from gc import collect
from glob import glob
from math import floor
from os.path import dirname, exists, isdir, join

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from h5py import File
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from ..params import paths_glamdring
from ..utils import (cartesian_to_radec, fprint, great_circle_distance,
                     number_counts, periodic_distance_two_points,
                     radec_to_galactic, real2redshift, radec_to_supergalactic)
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
                   "galactic_pos",
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
        """Simulation IC realisation index."""
        if self._nsim is None:
            raise RuntimeError("`nsim` is not set!")
        return self._nsim

    @nsim.setter
    def nsim(self, nsim):
        if not isinstance(nsim, (int, np.integer)):
            raise TypeError("`nsim` must be an integer.")
        self._nsim = int(nsim)

    @property
    def nsnap(self):
        """Catalogue snapshot index."""
        if self._nsnap is None:
            raise RuntimeError("`nsnap` is not set!")
        return self._nsnap

    @nsnap.setter
    def nsnap(self, nsnap):
        if not isinstance(nsnap, (int, np.integer)):
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
        """Paths manager."""
        if self._paths is None:
            return Paths(**paths_glamdring)
        return self._paths

    @property
    def boxsize(self):
        """Box size in `cMpc / h`."""
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
        """
        return self._flip_xz

    @flip_xz.setter
    def flip_xz(self, flip_xz):
        if not isinstance(flip_xz, bool):
            raise TypeError("`flip_xz` must be a boolean.")
        self._flip_xz = flip_xz

    @property
    def cache_maxsize(self):
        """Maximum length of the cache dictionary."""
        if self._cache_maxsize is None:
            raise RuntimeError("`cache_maxsize` is not set!")
        return self._cache_maxsize

    @cache_maxsize.setter
    def cache_maxsize(self, cache_maxsize):
        assert isinstance(cache_maxsize, int)
        self._cache_maxsize = cache_maxsize

    def cache_keys(self):
        """Current keys of the cache dictionary."""
        return list(self._cache.keys())

    def cache_length(self):
        """Current length of the cache dictionary."""
        return len(self._cache)

    @property
    @abstractmethod
    def coordinates(self):
        """Halo coordinates."""
        pass

    @property
    @abstractmethod
    def velocities(self):
        """Halo peculiar velocities."""
        pass

    @property
    @abstractmethod
    def npart(self):
        """Number of particles in a halo."""
        pass

    @property
    @abstractmethod
    def totmass(self):
        """Total particle mass of a halo."""
        pass

    @property
    @abstractmethod
    def index(self):
        """Halo index."""
        pass

    @property
    @abstractmethod
    def lagpatch_coordinates(self):
        """Lagrangian patch coordinates."""
        pass

    @property
    @abstractmethod
    def lagpatch_radius(self):
        """Lagrangian patch radius."""
        pass

    @property
    def observer_location(self):
        if self._observer_location is None:
            raise RuntimeError("`observer_location` is not set!")
        return self._observer_location

    @observer_location.setter
    def observer_location(self, obs_pos):
        assert isinstance(obs_pos, (list, tuple, np.ndarray))
        obs_pos = np.asanyarray(obs_pos)
        assert obs_pos.shape == (3,)
        self._observer_location = obs_pos

    @property
    def observer_velocity(self):
        if self._observer_velocity is None:
            raise RuntimeError("`observer_velocity` is not set!")
        return self._observer_velocity

    @observer_velocity.setter
    def observer_velocity(self, obs_vel):
        if obs_vel is None:
            self._observer_velocity = None
            return

        assert isinstance(obs_vel, (list, tuple, np.ndarray))
        obs_vel = np.asanyarray(obs_vel)
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

        bin_edges = np.log10(bin_edges)
        bin_width = bin_edges[1:] - bin_edges[:-1]

        x = (bin_edges[1:] + bin_edges[:-1]) / 2
        y = counts / bin_width / volume
        yerr = np.sqrt(counts) / bin_width / volume

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
                rad_sep = np.abs(radial_dist[indxs[i]] - radial_query[i])
                mask = rad_sep < radial_tolerance
                dist[i], indxs[i] = dist[i][mask], indxs[i][mask]

        return dist, indxs

    def _make_mask(self, bounds):
        """
        Make an internal mask for the catalogue data.
        """
        self._load_filtered = False

        self._catalogue_length = None
        mask = np.ones(len(self), dtype=bool)
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
        """Clear the cache dictionary."""
        self._cache.clear()
        collect()

    def keys(self):
        """Catalogue keys."""
        return self._properties + self._custom_keys

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
        print(fiducial_observers(self.boxsize, rmax))
        self.observer_location = fiducial_observers(self.boxsize, rmax)[n]
        self.observer_velocity = None

        if self._bounds is None:
            bounds = {"dist": (0, rmax)}
        else:
            bounds = {**self._bounds, "dist": (0, rmax)}

        self._make_mask(bounds)

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
            elif key == "galactic_pos":
                out = self["__spherical_pos"]
                out[:, 1], out[:, 2] = radec_to_galactic(out[:, 1], out[:, 2])
            elif key == "supergalactic_pos":
                out = self["__spherical_pos"]
                out[:, 1], out[:, 2] = radec_to_supergalactic(
                    out[:, 1], out[:, 2])
            elif key == "dist":
                out = np.linalg.norm(
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
                out = np.linalg.norm(out - self.observer_location, axis=1)
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

        if self._load_filtered and not is_internal and isinstance(out, np.ndarray):  # noqa
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
            return np.vstack([z, y, x]).T
        else:
            return np.vstack([x, y, z]).T

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
        data = np.load(fpath)

        if self.flip_xz:
            return np.vstack([data["z"], data["y"], data["x"]]).T
        else:
            return np.vstack([data["x"], data["y"], data["z"]]).T

    @property
    def lagpatch_radius(self):
        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        return np.load(fpath)["lagpatch_size"]


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
                             "GroupNsubs", "Group_M_Crit200",
                             "Group_M_Crit500", "Group_R_Crit200",
                             "Group_R_Crit500"]

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
        return np.arange(nhalo, dtype=np.int32)

    @property
    def lagpatch_coordinates(self):
        if self.nsnap != 99:
            raise RuntimeError("Lagrangian patch information is only "
                               "available for haloes defined at the final "
                               f"snapshot (indexed 99). Chosen {self.nsnap}.")

        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        data = np.load(fpath)

        if self.flip_xz:
            return np.vstack([data["z"], data["y"], data["x"]]).T
        else:
            return np.vstack([data["x"], data["y"], data["z"]]).T

    @property
    def lagpatch_radius(self):
        if self.nsnap != 99:
            raise RuntimeError("Lagrangian patch information is only "
                               "available for haloes defined at the final "
                               f"snapshot (indexed 99). Chosen {self.nsnap}.")

        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        return np.load(fpath)["lagpatch_size"]

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
    def Group_M_Crit500(self):
        return self._read_fof_catalogue("Group_M_Crit500")

    @property
    def Group_R_Crit200(self):
        return self._read_fof_catalogue("Group_R_Crit200")

    @property
    def Group_R_Crit500(self):
        return self._read_fof_catalogue("Group_R_Crit500")

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
        """Simulation IC realisation index."""
        if self._nsim is None:
            raise RuntimeError("`nsim` is not set!")
        return self._nsim

    @nsim.setter
    def nsim(self, nsim):
        if not isinstance(nsim, (int, np.integer)):
            raise TypeError("`nsim` must be an integer.")
        self._nsim = int(nsim)

    @property
    def kind(self):
        """Simulation kind."""
        return self._simname.split("_")[-1]

    @property
    def paths(self):
        """Paths manager."""
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
                        "Group_M_Crit200",
                        "SubhaloSpin",
                        "SubhaloVmax",
                        "SubhaloHalfmassRad",
                        "SubhaloVmaxRad",
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

        if not np.all(first_fof == first_fof[0]):
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

        snap_num, redshift, main_progenitor_mass, group_m200c = [], [], [], []
        main_progenitor_vmax, main_progenitor_spin = [], []
        main_progenitor_vmaxrad, main_progenitor_halfmassrad = [], []
        while True:
            # NOTE: 'Minors' are ignored. This is only relevant if we wanted
            # to find the other subhaloes in the current FoF group.

            # # First off attempt to find the next progenitors of the current
            # # halo. Deal with the main progenitor later.
            # next_prog = tree["TreeNextProgenitor"][n]
            # if next_prog != -1:
            #     minors = []
            #     while True:
            #         minors.append(tree["SubhaloMass"][next_prog])

            #         next_prog = tree["TreeNextProgenitor"][next_prog]

            #         if next_prog == -1:
            #             break
            # else:
            #     # Fiducially set it to zero.
            #     minors = [0]

            # Update data with information from the current main progenitor.
            major = tree["SubhaloMass"][n]
            main_progenitor_mass.append(major)
            group_m200c.append(tree["Group_M_Crit200"][n])
            main_progenitor_vmax.append(tree["SubhaloVmax"][n])
            main_progenitor_spin.append(tree["SubhaloSpin"][n])
            main_progenitor_vmaxrad.append(tree["SubhaloVmaxRad"][n])
            main_progenitor_halfmassrad.append(tree["SubhaloHalfmassRad"][n])
            snap_num.append(tree["SnapNum"][n])
            redshift.append(tree["Redshift"][tree["SnapNum"][n]])

            # Update `n` to the next main progenitor.
            n = tree["TreeMainProgenitor"][n]

            if n == -1:
                break

        # For calculating age of the Universe at each redshift.
        cosmo = FlatLambdaCDM(H0=67.66, Om0=0.3111)

        return {"SnapNum": np.array(snap_num, dtype=np.int32),
                "Age": np.array(cosmo.age(redshift).value),
                "Redshift": np.array(redshift),
                "Group_M_Crit200": np.array(group_m200c) * 1e10,
                "MainProgenitorMass": np.array(main_progenitor_mass) * 1e10,
                "MainProgenitorVmax": np.array(main_progenitor_vmax),
                "MainProgenitorSpin": np.array(main_progenitor_spin),
                "MainProgenitorVmaxRad": np.array(main_progenitor_vmaxrad),
                "MainProgenitorHalfmassRad": np.array(main_progenitor_halfmassrad),      # noqa
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
        super().__init__()
        super().init_with_snapshot(
            f"csiborg2_{kind}", nsim, nsnap, paths, None, bounds,
            676.6, [338.3, 338.3, 338.3], None, flip_xz,
            cache_maxsize)

        self._custom_keys = ["SubhaloSpin", "SubhaloVelDisp", "Central",
                             "SubhaloVmax", "SubhaloVmaxRad",
                             "SubhaloHalfmassRad", "ParentMass"]

    @property
    def kind(self):
        """Simulation kind."""
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

    def _read_fof_catalogue(self, kind):
        fpath = self.paths.snapshot_catalogue(self.nsnap, self.nsim,
                                              self._simname)

        with File(fpath, 'r') as f:
            grp = f["Group"]
            if kind not in grp.keys():
                raise ValueError(f"FoF catalogue key '{kind}' not available. Available keys are: {list(f.keys())}")  # noqa
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
        return np.arange(self.totmass.size, dtype=np.int32)

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
    def SubhaloVmax(self):
        return self._read_subfind_catalogue("SubhaloVmax")

    @property
    def SubhaloVmaxRad(self):
        return self._read_subfind_catalogue("SubhaloVmaxRad")

    @property
    def SubhaloHalfmassRad(self):
        return self._read_subfind_catalogue("SubhaloHalfmassRad")

    @property
    def SubhaloContamination(self):
        mass_type = self._read_subfind_catalogue("SubhaloMassType")
        return mass_type[:, 5] / (mass_type[:, 1] + mass_type[:, 5])

    @property
    def Central(self):
        return self._read_subfind_catalogue("SubhaloRankInGr") == 0

    @property
    def ParentMass(self):
        group_nr = self._read_subfind_catalogue("SubhaloGroupNr")
        fof_mass = self._read_fof_catalogue("GroupMass") * 1e10
        return fof_mass[group_nr]

###############################################################################
#                          CSiBORG3 catalogues                                #
###############################################################################


class CSiBORG3Catalogue(BaseCatalogue):
    r"""
    CSiBORG3 FoF halo catalogue.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    nsnap : int
        Snapshot index.
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
    fpath_override : str, optional
        Path to the catalogue file. If not provided, the default path is used.
    boxsize : float, optional
        Box size in :math:`\mathrm{cMpc} / h` (or units of positions).
    cache_maxsize : int, optional
        Maximum number of cached arrays.
    verbose : bool, optional
        Verbosity flag for reading inputs.
    """
    def __init__(self, nsim, nsnap, paths=None, snapshot=None,
                 bounds=None, observer_velocity=None, flip_xz=True,
                 fpath_override=None, boxsize=681.1, cache_maxsize=64,
                 verbose=True):
        super().__init__()
        super().init_with_snapshot(
            "csiborg3", nsim, nsnap, paths, snapshot, bounds,
            boxsize, [boxsize / 2, boxsize / 2, boxsize / 2],
            observer_velocity, flip_xz, cache_maxsize)
        self.fpath_override = fpath_override
        self._verbose = verbose

        self._custom_keys = ["GroupFirstSub", "GroupContamination",
                             "GroupNsubs", "Group_M_Crit200"]

    def _read_fof_catalogue(self, kind):
        if self.fpath_override is None:
            fpath = self.paths.snapshot_catalogue(
                self.nsnap, self.nsim, self._simname)
        else:
            fpath = self.fpath_override

        files = glob(fpath.replace(".hdf5", ".*.hdf5"))
        if len(files) == 0:
            raise FileNotFoundError(
                f"No files found for snapshot {self.nsnap}.")
        files = sorted(files, key=lambda x: int(x.split(".")[-2]))

        fprint(f"opening {len(files)} blocks for snapshot `{self.nsnap}`.",
               verbose=self._verbose)
        for i, fpath in enumerate(tqdm(files, desc="Reading blocks",
                                       disable=not self._verbose)):
            with File(fpath, 'r') as f:
                try:
                    grp = f["Group"]
                except KeyError:
                    continue

                if kind not in grp.keys():
                    raise ValueError(f"Group catalogue key '{kind}' not available. Available keys are: {list(grp.keys())}")  # noqa
                out = grp[kind][...]

            if i == 0:
                out_all = np.copy(out)
            else:
                out_all = np.concatenate([out_all, out])
        return out_all

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
        return np.arange(nhalo, dtype=np.int32)

    @property
    def lagpatch_coordinates(self):
        raise RuntimeError("Lagrangian patch information is not available for "
                           "CSiBORG3 haloes.")
        if self.nsnap != 99:
            raise RuntimeError("Lagrangian patch information is only "
                               "available for haloes defined at the final "
                               f"snapshot (indexed 99). Chosen {self.nsnap}.")

        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        data = np.load(fpath)

        if self.flip_xz:
            return np.vstack([data["z"], data["y"], data["x"]]).T
        else:
            return np.vstack([data["x"], data["y"], data["z"]]).T

    @property
    def lagpatch_radius(self):
        raise RuntimeError("Lagrangian patch information is not available for "
                           "CSiBORG3 haloes.")
        if self.nsnap != 99:
            raise RuntimeError("Lagrangian patch information is only "
                               "available for haloes defined at the final "
                               f"snapshot (indexed 99). Chosen {self.nsnap}.")

        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        return np.load(fpath)["lagpatch_size"]

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
        return np.vstack([self._read_fof_catalogue(key)
                          for key in ["x", "y", "z"]]).T

    @property
    def velocities(self):
        return np.vstack([self._read_fof_catalogue(key)
                          for key in ["vx", "vy", "vz"]]).T

    @property
    def npart(self):
        offset = self._read_fof_catalogue("GroupOffset")
        return offset[:, 2] - offset[:, 1]

    @property
    def totmass(self):
        return self._read_fof_catalogue("GroupMass")

    @property
    def index(self):
        return self._read_fof_catalogue("index")

    @property
    def lagpatch_coordinates(self):
        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        data = np.load(fpath)
        return np.vstack([data["x"], data["y"], data["z"]]).T

    @property
    def lagpatch_radius(self):
        fpath = self.paths.initial_lagpatch(self.nsim, self.simname)
        return np.load(fpath)["lagpatch_size"]


###############################################################################
#                     External halo catalogues                                #
###############################################################################

class MDPL2Catalogue(BaseCatalogue):
    r"""
    MDPL2 (FoF) halo catalogue at `z = 0`.

    Parameters
    ----------
    paths : py:class`csiborgtools.read.Paths`, optional
        Paths object.
    bounds : dict
        Parameter bounds; keys as parameter names, values as (min, max)
        tuples. Use `dist` for radial distance, `None` for no bound.
    cache_maxsize : int, optional
        Maximum number of cached arrays.
    """

    def __init__(self, paths=None, bounds=None, cache_maxsize=64):
        boxsize = 1000.
        super().__init__()
        x0 = boxsize / 2
        super().init_with_snapshot(
            "MDPL2", 0, 125, paths, None, bounds, boxsize, [x0, x0, x0], None,
            False, cache_maxsize)

        self._custom_keys = []
        self._bounds = bounds

    def _read_fof_catalogue(self, kind):
        fpath = self.paths.external_halo_catalogue(self.simname)

        with File(fpath, 'r') as f:
            if kind == "index":
                return np.arange(len(f["x"]))

            if kind not in f.keys():
                raise ValueError(f"FoF catalogue key '{kind}' not available. Available keys are: {list(f.keys())}")  # noqa
            out = f[kind][...]
        return out

    @property
    def coordinates(self):
        return np.vstack(
            [self._read_fof_catalogue(key) for key in ["x", "y", "z"]]).T

    @property
    def velocities(self):
        return np.vstack(
            [self._read_fof_catalogue(key) for key in ["vx", "vy", "vz"]]).T

    @property
    def totmass(self):
        return self._read_fof_catalogue("mass")

    @property
    def npart(self):
        raise RuntimeError("Number of particles is not available.")

    @property
    def index(self):
        return self._read_fof_catalogue("index")

    @property
    def lagpatch_coordinates(self):
        raise RuntimeError("Lagrangian patch information is not available")

    @property
    def lagpatch_radius(self):
        raise RuntimeError("Lagrangian patch information is not available")


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

    obs = []
    for i in range(nobs):
        x = (2 * i + 1) * radius
        for j in range(nobs):
            y = (2 * j + 1) * radius
            for k in range(nobs):
                z = (2 * k + 1) * radius
                obs.append([x, y, z])

    return obs


###############################################################################
#                           Rockstar halo catalogues                          #
###############################################################################

class BaseRockstar(ABC):
    """Base reader class for simulation data."""

    def __init__(self, basepath, nsnap, use_cache, verbose):
        self.basepath = basepath
        self.nsnap = nsnap
        self.use_cache = use_cache
        self.verbose = verbose

        self._cache = {}

    @property
    def basepath(self):
        if self._basepath is None:
            raise ValueError("Basepath for this catalogue is not set. Likely "
                             "because this is a scaled catalogue.")
        return self._basepath

    @basepath.setter
    def basepath(self, x):
        if x is None:
            self._basepath = None
            return

        if not (isinstance(x, str) and exists(x) and isdir(x)):
            raise TypeError(f"Basepath must be a directory. Received: `{x}`")
        self._basepath = x

    @property
    def nsnap(self):
        if self._nsnap is None:
            raise ValueError("Snapshot number for this catalogue is not set. "
                             "Likely because this is a scaled catalogue.")
        return self._nsnap

    @nsnap.setter
    def nsnap(self, x):
        if x is None:
            self._nsnap = None
            return

        if not isinstance(x, int):
            raise TypeError(f"Snapshot number must be an integer. Received: `{x}`")  # noqa
        self._nsnap = x

    @property
    def use_cache(self):
        return self._use_cache

    @use_cache.setter
    def use_cache(self, x):
        if not isinstance(x, bool):
            raise TypeError(f"Use cache must be a boolean. Received: `{x}`")
        self._use_cache = x

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, x):
        if not isinstance(x, bool):
            raise TypeError(f"Verbose must be a boolean. Received: `{x}`")
        self._verbose = x


class ReaderRockstar(BaseRockstar):
    """
    Reader for Rockstar halo catalogs  that reads the ASCII files because of
    which it first reads in all possible keys.

    Parameters
    ----------
    basepath : str
        Path to the simulation's Rockstar output folder.
    nsnap : int
        Snapshot number.
    use_cache : bool, optional
        Whether to cache the data in memory.
    keep_in_memory : bool, optional
        Whether to keep the entire halo catalogue in memory or not.
    min_Mvir : float, optional
        Minimum virial mass to consider to keep in the halo catalogue.
    verbose : bool, optional
        Verbosity flag.
    """

    def __init__(self, basepath, nsnap, use_cache=True,
                 keep_in_memory=True, min_Mvir=None, verbose=True):
        super().__init__(basepath, nsnap, use_cache, verbose)

        if not isinstance(keep_in_memory, bool):
            raise TypeError("Keep in memory must be a boolean. "
                            f"Received: `{keep_in_memory}`")
        self.keep_in_memory = keep_in_memory

        self._data = None
        self._hid2index = None
        self._time_steps = None
        self._boxsize = None
        self._min_Mvir = min_Mvir
        self._Om = None
        self._Ol = None
        self._particle_mass = None

    @cached_property
    def time_steps(self):
        """Simulation time steps loaded from `time_steps.dat`."""
        if self.is_scaled:
            raise ValueError("Cannot read time steps for a scaled catalogue.")

        if self._time_steps is None:
            fname = join(
                dirname(dirname(self.basepath.rstrip("/"))),
                "write_outputs.dat")
            self._time_steps = np.genfromtxt(fname)

        return self._time_steps

    @cached_property
    def time_steps_redshift(self):
        """Redshifts corresponding to simulation time steps."""
        return 1 / self.time_steps - 1

    @cached_property
    def scale_factor(self):
        """Scale factor of the snapshot."""
        if self.is_scaled:
            return self._scale_factor

        return self.time_steps[self.nsnap]

    @cached_property
    def redshift(self):
        """Redshift of the snapshot."""
        return 1 / self.scale_factor - 1

    @cached_property
    def boxsize(self):
        """Box size of the simulation in `cMpc/h`."""
        if self._boxsize is None:
            fname = join(self.basepath, f"out_{self.nsnap}.list")
            with open(fname, 'r') as file:
                for line in file:
                    if "Box size:" in line:
                        boxsize = float(line.split(":")[1].strip().split()[0])
                        break

            self._boxsize = boxsize

        if self._boxsize is None:
            raise ValueError(f"Box size not found in Rockstar file: {fname}")

        return self._boxsize

    @cached_property
    def Om(self):
        if self._Om is None:
            self._Om = self._extract_cosmology("Om")

        return self._Om

    def Om_z(self, z):
        return self.cosmology.omega_m_z(z)

    @cached_property
    def Ol(self):
        if self._Ol is None:
            self._Ol = self._extract_cosmology("Ol")

        return self._Ol

    def particle_mass(self):
        """Dark matter particle mass in `Msun / h`."""
        fname = join(self.basepath, f"out_{self.nsnap}.list")
        if self._particle_mass is None:
            particle_mass = None
            with open(fname, "r") as f:
                for i, line in enumerate(f):
                    match = re.search(r"#Particle mass:\s*([\d.eE+-]+)", line)
                    if match:
                        particle_mass = float(match.group(1))
                        break

                    if i > 100 and particle_mass is None:
                        raise ValueError("Particle mass not found in "
                                         f"Rockstar file: {fname}")
            self._particle_mass = particle_mass

        return self._particle_mass

    def _extract_cosmology(self, key):
        fname = join(self.basepath, f"out_{self.nsnap}.list")
        with open(fname, 'r') as file:
            for line in file:
                if f"{key} =" in line:
                    match = re.search(rf"{key}\s*=\s*([\d\.]+)", line)
                    if match:
                        value = float(match.group(1))
                        return value

        raise ValueError(f"Key `{key}` not found in Rockstar file: {fname}")

    def _read_catalog(self, nsnap):
        fname = join(self.basepath, f"out_{nsnap}.list")
        if self.verbose:
            print(f"Reading a Rockstar catalogue from `{fname}`.")

        line_start = None

        with open(fname, 'r') as f:
            keys = f.readline().strip("#").strip().split()

            for line_number, line in enumerate(f, start=1):
                if not line.lstrip().startswith("#"):
                    line_start = line_number
                    break

        if line_start is None:
            raise ValueError(f"Invalid Rockstar file: {fname}")

        dtype = [(key, 'int32' if key in ['ID', 'DescID', 'Np'] else 'float32')
                 for key in keys]

        self._data = np.genfromtxt(
            fname, skip_header=line_start, dtype=dtype, names=keys)

        if self._min_Mvir is not None:
            self._data = self._data[self._data["Mvir"] > self._min_Mvir]

    @property
    def data(self):
        if self._data is None:
            self._read_catalog(self.nsnap)
        return self._data

    def _read_key(self, key):
        custom_keys = ("pos", "vel", "c_klypin", "c_profile",
                       "vel_mag", )
        log_keys = ("M200c", "Mvir", "Spin", "c_klypin", "rs_klypin",
                    "vel_mag", )

        if key == "pos":
            return np.vstack(
                [self.data["X"], self.data["Y"], self.data["Z"]]).T
        if key == "vel":
            return np.vstack(
                [self.data["VX"], self.data["VY"], self.data["VZ"]]).T
        elif key == "c_klypin":
            return self["Rvir"] / self["rs_klypin"]
        elif key == "c_profile":
            return self["Rvir"] / self["Rs"]
        elif key == "Mvir_simple":
            return self["Mvir"]
        elif key == "vel_mag":
            return np.linalg.norm(self["vel"], axis=1)
        elif key in self.keys:
            return self.data[key]
        elif "log_" in key and key.replace("log_", "") in log_keys:
            return np.log10(self[key.replace("log_", "")])
        else:
            nsnap = self._nsnap if self._nsnap is not None else "NotFound"

            raise KeyError(f"Key `{key}` not found in Rockstar catalog for "
                           f"snapshot `{nsnap}`. Available keys are "
                           f"`{self.keys}` or `{custom_keys}`")

    def hid2index(self, hid):
        """Convert a halo ID `hid` to its index in the catalogue."""
        if self._hid2index is None:
            hid_ = self["ID"]
            self._hid2index = {int(hid_[i]): i for i in range(len(hid_))}

        if hid not in self._hid2index:
            raise KeyError(f"Halo ID `{hid}` not found in catalogue.")

        return self._hid2index[hid]

    def read_by_hid(self, hid, key):
        """
        Read `key` for a halo with a given halo ID `hid` or a list of halo IDs.
        """
        if isinstance(hid, int):
            hid = [hid]
        elif not isinstance(hid, (list, np.ndarray, tuple)):
            raise TypeError(f"Halo ID must be an integer, list or array. "
                            f"Received: `{hid}`")

        ks = np.asarray([self.hid2index(hid_i) for hid_i in hid])
        return self[key][ks]

    @cached_property
    def keys(self):
        if isinstance(self.data, dict):
            return list(self.data.keys())
        return self.data.dtype.names

    def __len__(self):
        if isinstance(self.data, dict):
            return len(self.data[self.keys[0]])
        return len(self.data)

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]

        x = self._read_key(key)
        if self.use_cache:
            self._cache[key] = x

        return x
