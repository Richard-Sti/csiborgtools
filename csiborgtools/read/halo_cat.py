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
"""CSiBORG halo catalogue."""
from abc import ABC
import numpy
from sklearn.neighbors import NearestNeighbors
from .readsim import (ParticleReader, read_initcm)
from .paths import CSiBORGPaths
from ..utils import (flip_cols, add_columns)
from ..units import (BoxUnits, cartesian_to_radec)


class BaseCatalogue(ABC):
    """
    Base (sub)halo catalogue
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
        assert isinstance(nsim, int)
        self._nsim = nsim

    @property
    def paths(self):
        """
        CSiBORG paths manager.

        Returns
        -------
        paths : :py:class:`csiborgtools.read.CSiBORGPaths`
        """
        if self._paths is None:
            raise RuntimeError("`paths` is not set!")
        return self._paths

    @paths.setter
    def paths(self, paths):
        # assert isinstance(paths, CSiBORGPaths)  # REMOVE
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

    @property
    def nsnap(self):
        """
        Catalogue's snapshot index corresponding to the maximum simulation
        snapshot index.

        Returns
        -------
        nsnap : int
        """
        return max(self.paths.get_snapshots(self.nsim))

    @property
    def box(self):
        """
        CSiBORG box object handling unit conversion.

        Returns
        -------
        box : :py:class:`csiborgtools.units.BoxUnits`
        """
        return BoxUnits(self.nsnap, self.nsim, self.paths)

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
            ps = ['x0', 'y0', 'z0']
        else:
            ps = ['x', 'y', 'z']
        pos = [self[p] for p in ps]
        if cartesian:
            return numpy.vstack(pos).T
        else:
            return numpy.vstack([cartesian_to_radec(*pos)]).T

    def velocity(self):
        """
        Cartesian velocity components in box units.

        Returns
        -------
        vel : 2-dimensional array of shape `(nobjects, 3)`
        """
        return numpy.vstack([self["v{}".format(p)] for p in ("x", "y", "z")]).T

    def angmomentum(self):
        """
        Cartesian angular momentum components of halos in the box coordinate
        system. Likely in box units.

        Returns
        -------
        angmom : 2-dimensional array of shape `(nobjects, 3)`
        """
        return numpy.vstack([self["L{}".format(p)] for p in ("x", "y", "z")]).T

    def knn(self, in_initial):
        """
        kNN object fitted on all catalogue objects.

        Parameters
        ----------
        in_initial : bool
            Whether to define the kNN on the initial or final snapshot.

        Returns
        -------
        knn : :py:class:`sklearn.neighbors.NearestNeighbors`
        """
        knn = NearestNeighbors()
        return knn.fit(self.positions(in_initial))

    def radius_neigbours(self, X, radius, in_initial):
        r"""
        Sorted nearest neigbours within `radius` of `X` in the initial
        or final snapshot.

        Parameters
        ----------
        X : 2-dimensional array of shape `(n_queries, 3)`
            Cartesian query position components in :math:`\mathrm{cMpc}`.
        radius : float
            Limiting neighbour distance.
        in_initial : bool
            Whether to define the kNN on the initial or final snapshot.

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
        knn = self.knn(in_initial)
        return knn.radius_neighbors(X, radius, sort_results=True)

    @property
    def keys(self):
        """Catalogue keys."""
        return self.data.dtype.names

    def __getitem__(self, key):
        initpars = ["x0", "y0", "z0"]
        if key in initpars and key not in self.keys:
            raise RuntimeError("Initial positions are not set!")
        return self.data[key]

    def __len__(self):
        return self.data.size


class ClumpsCatalogue(BaseCatalogue):
    r"""
    Clumps catalogue, defined in the final snapshot.

    TODO:
        Add fitted quantities.
        Add threshold on number of particles

    Parameters
    ----------
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.CSiBORGPaths`
        CSiBORG paths object.
    maxdist : float, optional
        The maximum comoving distance of a halo. By default
        :math:`155.5 / 0.705 ~ \mathrm{Mpc}` with assumed :math:`h = 0.705`,
        which corresponds to the high-resolution region.
    """
    def __init__(self, nsim, paths, maxdist=155.5 / 0.705):
        self.nsim = nsim
        self.paths = paths
        # Read in the clumps from the final snapshot
        partreader = ParticleReader(self.paths)
        cols=["index", "parent", 'x', 'y', 'z', "mass_cl"]
        data = partreader.read_clumps(self.nsnap, self.nsim, cols=cols)
        # Overwrite the parent with the ultimate parent
        mmain = numpy.load(self.paths.mmain_path(self.nsnap, self.nsim))
        data["parent"] = mmain["ultimate_parent"]

        # Flip positions and convert from code units to cMpc. Convert M too
        flip_cols(data, "x", "z")
        for p in ('x', 'y', 'z'):
            data[p] -= 0.5
        data = self.box.convert_from_boxunits(data, ['x', 'y', 'z', "mass_cl"])

        mask = numpy.sqrt(data['x']**2 + data['y']**2 + data['z']**2) < maxdist
        self._data = data[mask]

    @property
    def ismain(self):
        """
        Whether the clump is a main halo.

        Returns
        -------
        ismain : 1-dimensional array
        """
        return self["index"] == self["parent"]

    def _set_data(self, min_mass, max_dist, load_init):
        """
        TODO: old later remove.
        Loads the data, merges with mmain, does various coordinate transforms.
        """
        # Load the processed data
        data = numpy.load(self.paths.hcat_path(self.nsim))

        # Load the mmain file and add it to the data
        # TODO: read the mmain here
#        mmain = read_mmain(self.nsim, self.paths.mmain_dir)
#        data = self.merge_mmain_to_clumps(data, mmain)
        flip_cols(data, "peak_x", "peak_z")

        # Cut on number of particles and finite m200. Do not change! Hardcoded
        data = data[(data["npart"] > 100) & numpy.isfinite(data["m200"])]

        # Now also load the initial positions
        if load_init:

            initcm = read_initcm(self.nsim,
                                 self.paths.initmatch_path(self.nsim, "cm"))
            if initcm is not None:
                data = self.merge_initmatch_to_clumps(data, initcm)
                flip_cols(data, "x0", "z0")

        # Unit conversion
        convert_cols = ["m200", "m500", "totpartmass", "mass_mmain",
                        "r200", "r500", "Rs", "rho0",
                        "peak_x", "peak_y", "peak_z"]
        data = self.box.convert_from_boxunits(data, convert_cols)

        # And do the unit transform
        if load_init and initcm is not None:
            data = self.box.convert_from_boxunits(
                data, ["x0", "y0", "z0", "lagpatch"])

        # Convert all that is not an integer to float32
        names = list(data.dtype.names)
        formats = []
        for name in names:
            if data[name].dtype.char in numpy.typecodes["AllInteger"]:
                formats.append(numpy.int32)
            else:
                formats.append(numpy.float32)
        dtype = numpy.dtype({"names": names, "formats": formats})

        # Apply cuts on distance and total particle mass if any
        data = data[data["dist"] < max_dist] if max_dist is not None else data
        data = (data[data["totpartmass"] > min_mass]
                if min_mass is not None else data)

        self._data = data.astype(dtype)

    def merge_mmain_to_clumps(self, clumps, mmain):
        """
        TODO: old, later remove.
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

    def merge_initmatch_to_clumps(self, clumps, initcat):
        """
        TODO: old, later remove.
        Merge columns from the `init_cm` files to the `clump` file.

        Parameters
        ----------
        clumps : structured array
            Clumps structured array.
        initcat : structured array
            Catalog with the clumps initial centre of mass at z = 70.

        Returns
        -------
        out : structured array
        """
        # There are more initcat clumps, so check which ones have z = 0
        # and then downsample
        mask = numpy.isin(initcat["ID"], clumps["index"])
        initcat = initcat[mask]
        # Now the index ordering should match
        if not numpy.alltrue(initcat["ID"] == clumps["index"]):
            raise ValueError(
                "Ordering of `initcat` and `clumps` is inconsistent.")

        X = numpy.full((clumps.size, 4), numpy.nan)
        for i, p in enumerate(['x', 'y', 'z', "lagpatch"]):
            X[:, i] = initcat[p]
        return add_columns(clumps, X, ["x0", "y0", "z0", "lagpatch"])


class HaloCatalogue(BaseCatalogue):
    """
    Halo catalogue, i.e. parent halos with summed substructure, defined in the
    final snapshot.

    TODO:
        Add the fitted quantities
        Add threshold on number of particles

    Parameters
    ----------
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.CSiBORGPaths`
        CSiBORG paths object.
    maxdist : float, optional
        The maximum comoving distance of a halo. By default
        :math:`155.5 / 0.705 ~ \mathrm{Mpc}` with assumed :math:`h = 0.705`,
        which corresponds to the high-resolution region.
    """
    def __init__(self, nsim, paths, maxdist=155.5 / 0.705):
        self.nsim = nsim
        self.paths = paths
        # Read in the mmain catalogue of summed substructure
        mmain = numpy.load(self.paths.mmain_path(self.nsnap, self.nsim))
        data = mmain["mmain"]
        # Flip positions and convert from code units to cMpc. Convert M too
        flip_cols(data, "x", "z")
        for p in ('x', 'y', 'z'):
            data[p] -= 0.5
        data = self.box.convert_from_boxunits(data, ['x', 'y', 'z', 'M'])

        mask = numpy.sqrt(data['x']**2 + data['y']**2 + data['z']**2) < maxdist
        self._data = data[mask]


###############################################################################
#                           Useful functions                                  #
###############################################################################


def concatenate_clumps(clumps):
    """
    Concatenate an array of clumps to a single array containing all particles.

    Parameters
    ----------
    clumps : list of structured arrays

    Returns
    -------
    particles : structured array
    """
    # Count how large array will be needed
    N = 0
    for clump, __ in clumps:
        N += clump.size
    # Infer dtype of positions
    if clumps[0][0]['x'].dtype.char in numpy.typecodes["AllInteger"]:
        posdtype = numpy.int32
    else:
        posdtype = numpy.float32

    # Pre-allocate array
    dtype = {"names": ['x', 'y', 'z', 'M'],
             "formats": [posdtype] * 3 + [numpy.float32]}
    particles = numpy.full(N, numpy.nan, dtype)

    # Fill it one clump by another
    start = 0
    for clump, __ in clumps:
        end = start + clump.size
        for p in ('x', 'y', 'z', 'M'):
            particles[p][start:end] = clump[p]
        start = end

    return particles
