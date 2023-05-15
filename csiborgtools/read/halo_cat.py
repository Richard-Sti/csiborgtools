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
    - CSiBORG: halo and clump catalogue.
    - Quijote: halo catalogue.
"""
from abc import ABC, abstractproperty
from copy import deepcopy
from functools import lru_cache
from itertools import product
from math import floor
from os.path import join

import numpy

from readfof import FoF_catalog
from sklearn.neighbors import NearestNeighbors

from .box_units import CSiBORGBox, QuijoteBox
from .paths import Paths
from .readsim import ParticleReader
from .utils import (add_columns, cartesian_to_radec, cols_to_structured,
                    flip_cols, radec_to_cartesian, real2redshift)


class BaseCatalogue(ABC):
    """
    Base (sub)halo catalogue.
    """
    _data = None
    _paths = None
    _nsim = None

    @property
    def nsim(self):
        """
        The IC realisation index.

        Returns
        -------
        nsim : int
        """
        if self._nsim is None:
            raise RuntimeError("`nsim` is not set!")
        return self._nsim

    @nsim.setter
    def nsim(self, nsim):
        self._nsim = nsim

    @abstractproperty
    def nsnap(self):
        """
        Catalogue's snapshot index.

        Returns
        -------
        nsnap : int
        """
        pass

    @property
    def paths(self):
        """
        Paths manager.

        Returns
        -------
        paths : :py:class:`csiborgtools.read.Paths`
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
        data : structured array
        """
        if self._data is None:
            raise RuntimeError("Catalogue data not loaded!")
        return self._data

    def apply_bounds(self, bounds):
        for key, (xmin, xmax) in bounds.items():
            xmin = -numpy.inf if xmin is None else xmin
            xmax = numpy.inf if xmax is None else xmax
            if key == "dist":
                x = self.radial_distance(in_initial=False)
            else:
                x = self[key]
            self._data = self._data[(x > xmin) & (x <= xmax)]

    @abstractproperty
    def box(self):
        """
        Box object.

        Returns
        -------
        box : instance of :py:class:`csiborgtools.units.BoxUnits`
        """
        pass

    def position(self, in_initial=False, cartesian=True):
        r"""
        Position components. If Cartesian, then in :math:`\mathrm{cMpc}`. If
        spherical, then radius is in :math:`\mathrm{cMpc}`, RA in
        :math:`[0, 360)` degrees and DEC in :math:`[-90, 90]` degrees. Note
        that the position is defined as the minimum of the gravitationl
        potential.

        Parameters
        ----------
        in_initial : bool, optional
            Whether to return the initial snapshot positions.
        cartesian : bool, optional
            Whether to return the Cartesian or spherical position components.
            By default Cartesian.

        Returns
        -------
        pos : 2-dimensional array of shape `(nobjects, 3)`
        """
        if in_initial:
            ps = ["x0", "y0", "z0"]
        else:
            ps = ["x", "y", "z"]
        pos = numpy.vstack([self[p] for p in ps]).T
        if not cartesian:
            pos = cartesian_to_radec(pos)
        return pos

    def velocity(self):
        r"""
        Cartesian velocity components in :math:`\mathrm{km} / \mathrm{s}`.

        Returns
        -------
        vel : 2-dimensional array of shape `(nobjects, 3)`
        """
        return numpy.vstack([self["v{}".format(p)] for p in ("x", "y", "z")]).T

    def redshift_space_position(self, cartesian=True):
        r"""
        Redshift space position components. If Cartesian, then in
        :math:`\mathrm{cMpc}`. If spherical, then radius is in
        :math:`\mathrm{cMpc}`, RA in :math:`[0, 360)` degrees and DEC in
        :math:`[-90, 90]` degrees. Note that the position is defined as the
        minimum of the gravitationl potential.

        Parameters
        ----------
        cartesian : bool, optional
            Whether to return the Cartesian or spherical position components.
            By default Cartesian.

        Returns
        -------
        pos : 2-dimensional array of shape `(nobjects, 3)`
        """
        pos = self.position(cartesian=True)
        vel = self.velocity()
        origin = [0., 0., 0.]
        rsp = real2redshift(pos, vel, origin, self.box, in_box_units=False,
                            make_copy=False)
        if not cartesian:
            rsp = cartesian_to_radec(rsp)
        return rsp

    def radial_distance(self, in_initial=False):
        r"""
        Distance of haloes from the origin.

        Parameters
        ----------
        in_initial : bool, optional
            Whether to calculate in the initial snapshot.

        Returns
        -------
        radial_distance : 1-dimensional array of shape `(nobjects,)`
        """
        pos = self.position(in_initial=in_initial, cartesian=True)
        return numpy.linalg.norm(pos, axis=1)

    def angmomentum(self):
        """
        Cartesian angular momentum components of halos in the box coordinate
        system. Likely in box units.

        Returns
        -------
        angmom : 2-dimensional array of shape `(nobjects, 3)`
        """
        return numpy.vstack([self["L{}".format(p)] for p in ("x", "y", "z")]).T

    @lru_cache(maxsize=2)
    def knn(self, in_initial):
        """
        kNN object fitted on all catalogue objects. Caches the kNN object.

        Parameters
        ----------
        in_initial : bool
            Whether to define the kNN on the initial or final snapshot.

        Returns
        -------
        knn : :py:class:`sklearn.neighbors.NearestNeighbors`
        """
        knn = NearestNeighbors()
        return knn.fit(self.position(in_initial=in_initial))

    def nearest_neighbours(self, X, radius, in_initial, knearest=False,
                           return_mass=False, masss_key=None):
        r"""
        Sorted nearest neigbours within `radius` of `X` in the initial or final
        snapshot. However, if `knearest` is `True` then the `radius` is assumed
        to be the integer number of nearest neighbours to return.

        Parameters
        ----------
        X : 2-dimensional array of shape `(n_queries, 3)`
            Cartesian query position components in :math:`\mathrm{cMpc}`.
        radius : float or int
            Limiting neighbour distance. If `knearest` is `True` then this is
            the number of nearest neighbours to return.
        in_initial : bool
            Whether to define the kNN on the initial or final snapshot.
        knearest : bool, optional
            Whether `radius` is the number of nearest neighbours to return.
        return_mass : bool, optional
            Whether to return the masses of the nearest neighbours.
        masss_key : str, optional
            Key of the mass column in the catalogue. Must be provided if
            `return_mass` is `True`.

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
        if knearest:
            assert isinstance(radius, int)
        if return_mass:
            assert masss_key is not None
        knn = self.knn(in_initial)

        if knearest:
            dist, indxs = knn.kneighbors(X, radius)
        else:
            dist, indxs = knn.radius_neighbors(X, radius, sort_results=True)

        if not return_mass:
            return dist, indxs

        if knearest:
            mass = numpy.copy(dist)
            for i in range(dist.shape[0]):
                mass[i, :] = self[masss_key][indxs[i]]
        else:
            mass = deepcopy(dist)
            for i in range(dist.size):
                mass[i] = self[masss_key][indxs[i]]

        return dist, indxs, mass

    def angular_neighbours(self, X, ang_radius, in_rsp, rad_tolerance=None):
        r"""
        Find nearest neighbours within `ang_radius` of query points `X`.
        Optionally applies radial tolerance, which is expected to be in
        :math:`\mathrm{cMpc}`.

        Parameters
        ----------
        X : 2-dimensional array of shape `(n_queries, 2)` or `(n_queries, 3)`
            Query positions. If 2-dimensional, then RA and DEC in degrees.
            If 3-dimensional, then radial distance in :math:`\mathrm{cMpc}`,
            RA and DEC in degrees.
        in_rsp : bool
            Whether to use redshift space positions of haloes.
        ang_radius : float
            Angular radius in degrees.
        rad_tolerance : float, optional
            Radial tolerance in :math:`\mathrm{cMpc}`.

        Returns
        -------
        dist : array of 1-dimensional arrays of shape `(n_neighbours,)`
            Distance of each neighbour to the query point.
        ind : array of 1-dimensional arrays of shape `(n_neighbours,)`
            Indices of each neighbour in this catalogue.
        """
        assert X.ndim == 2
        # We first get positions of haloes in this catalogue, store their
        # radial distance and normalise them to unit vectors.
        if in_rsp:
            pos = self.redshift_space_position(cartesian=True)
        else:
            pos = self.position(in_initial=False, cartesian=True)
        raddist = numpy.linalg.norm(pos, axis=1)
        pos /= raddist.reshape(-1, 1)
        # We convert RAdec query positions to unit vectors. If no radial
        # distance provided add it.
        if X.shape[1] == 2:
            X = numpy.vstack([numpy.ones_like(X[:, 0]), X[:, 0], X[:, 1]]).T
            radquery = None
        else:
            radquery = X[:, 0]

        X = radec_to_cartesian(X)
        knn = NearestNeighbors(metric="cosine")
        knn.fit(pos)
        # Convert angular radius to cosine difference.
        metric_maxdist = 1 - numpy.cos(numpy.deg2rad(ang_radius))
        dist, ind = knn.radius_neighbors(X, radius=metric_maxdist,
                                         sort_results=True)
        # And the cosine difference to angular distance.
        for i in range(X.shape[0]):
            dist[i] = numpy.rad2deg(numpy.arccos(1 - dist[i]))

        # Apply the radial tolerance
        if rad_tolerance is not None:
            assert radquery is not None
            for i in range(X.shape[0]):
                mask = numpy.abs(raddist[ind[i]] - radquery) < rad_tolerance
                dist[i] = dist[i][mask]
                ind[i] = ind[i][mask]
        return dist, ind

    @property
    def keys(self):
        """
        Catalogue keys.

        Returns
        -------
        keys : list of strings
        """
        return self.data.dtype.names

    def __getitem__(self, key):
        if key not in self.keys:
            raise KeyError(f"Key '{key}' not in catalogue.")
        return self.data[key]

    def __len__(self):
        return self.data.size


###############################################################################
#                       CSiBORG  base catalogue                               #
###############################################################################


class BaseCSiBORG(BaseCatalogue):
    """
    Base CSiBORG catalogue class.
    """

    @property
    def nsnap(self):
        return max(self.paths.get_snapshots(self.nsim))

    @property
    def box(self):
        """
        CSiBORG box object handling unit conversion.

        Returns
        -------
        box : instance of :py:class:`csiborgtools.units.BaseBox`
        """
        return CSiBORGBox(self.nsnap, self.nsim, self.paths)


###############################################################################
#                        CSiBORG clumps catalogue                             #
###############################################################################


class ClumpsCatalogue(BaseCSiBORG):
    r"""
    Clumps catalogue, defined in the final snapshot.

    Parameters
    ----------
    sim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.Paths`
        Paths object.
    bounds : dict
        Parameter bounds to apply to the catalogue. The keys are the parameter
        names and the items are a len-2 tuple of (min, max) values. In case of
        no minimum or maximum, use `None`. For radial distance from the origin
        use `dist`.
    load_fitted : bool, optional
        Whether to load fitted quantities.
    rawdata : bool, optional
        Whether to return the raw data. In this case applies no cuts and
        transformations.
    """

    def __init__(self, nsim, paths, bounds={"dist", (0, 155.5 / 0.705)},
                 load_fitted=True, rawdata=False):
        self.nsim = nsim
        self.paths = paths
        # Read in the clumps from the final snapshot
        partreader = ParticleReader(self.paths)
        cols = ["index", "parent", "x", "y", "z", "mass_cl"]
        self._data = partreader.read_clumps(self.nsnap, self.nsim, cols=cols)
        # Overwrite the parent with the ultimate parent
        mmain = numpy.load(self.paths.mmain_path(self.nsnap, self.nsim))
        self._data["parent"] = mmain["ultimate_parent"]

        if load_fitted:
            fits = numpy.load(paths.structfit_path(self.nsnap, nsim, "clumps"))
            cols = [col for col in fits.dtype.names if col != "index"]
            X = [fits[col] for col in cols]
            self._data = add_columns(self._data, X, cols)

        # If the raw data is not required, then start applying transformations
        # and cuts.
        if not rawdata:
            flip_cols(self._data, "x", "z")
            for p in ("x", "y", "z"):
                self._data[p] -= 0.5
            names = ["x", "y", "z", "mass_cl", "totpartmass", "rho0", "r200c",
                     "r500c", "m200c", "m500c", "r200m", "m200m",
                     "vx", "vy", "vz"]
            self._data = self.box.convert_from_box(self._data, names)
            self.apply_bounds(bounds)

    @property
    def ismain(self):
        """
        Whether the clump is a main halo.

        Returns
        -------
        ismain : 1-dimensional array
        """
        return self["index"] == self["parent"]


###############################################################################
#                        CSiBORG halo catalogue                               #
###############################################################################


class HaloCatalogue(BaseCSiBORG):
    r"""
    Halo catalogue, i.e. parent halos with summed substructure, defined in the
    final snapshot.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.Paths`
        Paths object.
    bounds : dict
        Parameter bounds to apply to the catalogue. The keys are the parameter
        names and the items are a len-2 tuple of (min, max) values. In case of
        no minimum or maximum, use `None`. For radial distance from the origin
        use `dist`.
    with_lagpatch : bool, optional
        Whether to only load halos with a resolved Lagrangian patch.
    load_fitted : bool, optional
        Whether to load fitted quantities.
    load_initial : bool, optional
        Whether to load initial positions.
    rawdata : bool, optional
        Whether to return the raw data. In this case applies no cuts and
        transformations.
    """
    _clumps_cat = None

    def __init__(self, nsim, paths, bounds={"dist", (0, 155.5 / 0.705)},
                 with_lagpatch=True, load_fitted=True, load_initial=True,
                 load_clumps_cat=False, rawdata=False):
        self.nsim = nsim
        self.paths = paths
        # Read in the mmain catalogue of summed substructure
        mmain = numpy.load(self.paths.mmain_path(self.nsnap, self.nsim))
        self._data = mmain["mmain"]
        # We will also need the clumps catalogue
        if load_clumps_cat:
            self._clumps_cat = ClumpsCatalogue(nsim, paths, rawdata=True,
                                               load_fitted=False)
        if load_fitted:
            fits = numpy.load(paths.structfit_path(self.nsnap, nsim, "halos"))
            cols = [col for col in fits.dtype.names if col != "index"]
            X = [fits[col] for col in cols]
            self._data = add_columns(self._data, X, cols)

        if load_initial:
            fits = numpy.load(paths.initmatch_path(nsim, "fit"))
            X, cols = [], []
            for col in fits.dtype.names:
                if col == "index":
                    continue
                if col in ['x', 'y', 'z']:
                    cols.append(col + "0")
                else:
                    cols.append(col)
                X.append(fits[col])

            self._data = add_columns(self._data, X, cols)

        if not rawdata:
            if with_lagpatch:
                self._data = self._data[numpy.isfinite(self['lagpatch'])]
            # Flip positions and convert from code units to cMpc. Convert M too
            flip_cols(self._data, "x", "z")
            for p in ("x", "y", "z"):
                self._data[p] -= 0.5
            names = ["x", "y", "z", "M", "totpartmass", "rho0", "r200c",
                     "r500c", "m200c", "m500c", "r200m", "m200m",
                     "vx", "vy", "vz"]
            self._data = self.box.convert_from_box(self._data, names)

            if load_initial:
                names = ["x0", "y0", "z0", "lagpatch"]
                self._data = self.box.convert_from_box(self._data, names)

            self.apply_bounds(bounds)

    @property
    def clumps_cat(self):
        """
        The raw clumps catalogue.

        Returns
        -------
        clumps_cat : :py:class:`csiborgtools.read.ClumpsCatalogue`
        """
        if self._clumps_cat is None:
            raise ValueError("`clumps_cat` is not loaded.")
        return self._clumps_cat


###############################################################################
#                         Quijote halo catalogue                              #
###############################################################################


class QuijoteHaloCatalogue(BaseCatalogue):
    """
    Quijote halo catalogue.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.Paths`
        Paths object.
    nsnap : int
        Snapshot index.
    origin : len-3 tuple, optional
        Where to place the origin of the box. By default the centre of the box.
        In units of :math:`cMpc`.
    bounds : dict
        Parameter bounds to apply to the catalogue. The keys are the parameter
        names and the items are a len-2 tuple of (min, max) values. In case of
        no minimum or maximum, use `None`. For radial distance from the origin
        use `dist`.
    **kwargs : dict
        Keyword arguments for backward compatibility.
    """
    _nsnap = None

    def __init__(self, nsim, paths, nsnap,
                 origin=[500 / 0.6711, 500 / 0.6711, 500 / 0.6711],
                 bounds=None, **kwargs):
        self.paths = paths
        self.nsnap = nsnap
        fpath = join(self.paths.quijote_dir, "halos", str(nsim))
        fof = FoF_catalog(fpath, self.nsnap, long_ids=False, swap=False,
                          SFR=False, read_IDs=False)

        cols = [("x", numpy.float32), ("y", numpy.float32),
                ("z", numpy.float32), ("vx", numpy.float32),
                ("vy", numpy.float32), ("vz", numpy.float32),
                ("group_mass", numpy.float32), ("npart", numpy.int32)]
        data = cols_to_structured(fof.GroupLen.size, cols)

        pos = fof.GroupPos / 1e3 / self.box.h
        for i in range(3):
            pos[:, i] -= origin[i]
        vel = fof.GroupVel * (1 + self.redshift)
        for i, p in enumerate(["x", "y", "z"]):
            data[p] = pos[:, i]
            data["v" + p] = vel[:, i]
        data["group_mass"] = fof.GroupMass * 1e10 / self.box.h
        data["npart"] = fof.GroupLen

        self._data = data
        self.apply_bounds(bounds)

    @property
    def nsnap(self):
        """
        Snapshot number.

        Returns
        -------
        nsnap : int
        """
        return self._nsnap

    @nsnap.setter
    def nsnap(self, nsnap):
        assert nsnap in [0, 1, 2, 3, 4]
        self._nsnap = nsnap

    @property
    def redshift(self):
        """
        Redshift of the snapshot.

        Returns
        -------
        redshift : float
        """
        z_dict = {4: 0.0, 3: 0.5, 2: 1.0, 1: 2.0, 0: 3.0}
        return z_dict[self.nsnap]

    @property
    def box(self):
        """
        Quijote box object.

        Returns
        -------
        box : instance of :py:class:`csiborgtools.units.BaseBox`
        """
        return QuijoteBox(self.nsnap)


###############################################################################
#                     Utility functions for halo catalogues                   #
###############################################################################


def fiducial_observers(boxwidth, radius):
    """
    Positions of fiducial observers in a box, such that that the box is
    subdivided among them into spherical regions.

    Parameters
    ----------
    boxwidth : float
        Box width.
    radius : float
        Radius of the spherical regions.

    Returns
    -------
    origins : list of len-3 lists
        Positions of the observers.
    """
    nobs = floor(boxwidth / (2 * radius))  # Number of observers per dimension

    origins = list(product([1, 3, 5], repeat=nobs))
    for i in range(len(origins)):
        origins[i] = list(origins[i])
        for j in range(nobs):
            origins[i][j] *= radius
    return origins
