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
from sklearn.neighbors import NearestNeighbors
from .readsim import (CSiBORGPaths, read_mmain, read_initcm)
from ..utils import (flip_cols, add_columns)
from ..units import (BoxUnits, cartesian_to_radec)


class HaloCatalogue:
    r"""
    Processed halo catalogue, the data should be calculated in `run_fit_halos`.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.CSiBORGPaths`
        CSiBORG paths object.
    min_mass : float, optional
        The minimum :math:`M_{rm tot} / M_\odot` mass. By default no threshold.
    max_dist : float, optional
        The maximum comoving distance of a halo. By default no upper limit.
    load_init : bool, optional
        Whether to load the initial snapshot information. By default False.
    """
    _paths = None
    _nsim = None
    _data = None
    _selmask = None

    def __init__(self, nsim, paths, min_mass=None, max_dist=None,
                 load_init=False):
        assert isinstance(paths, CSiBORGPaths)
        self._nsim = nsim
        self._paths = paths
        self._set_data(min_mass, max_dist, load_init)

    @property
    def nsim(self):
        return self._nsim

    @property
    def paths(self):
        return self._paths

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

    @property
    def data(self):
        """
        Halo catalogue.

        Returns
        -------
        cat : structured array
        """
        return self._data

    def _set_data(self, min_mass, max_dist, load_init):
        """
        Loads the data, merges with mmain, does various coordinate transforms.
        """
        # Load the processed data
        fname = "ramses_out_{}_{}.npy".format(
            str(self.nsim).zfill(5), str(self.nsnap).zfill(5))
        data = numpy.load(join(self.paths.dumpdir, fname))

        # Load the mmain file and add it to the data
        mmain = read_mmain(self.nsim, self.paths.mmain_path)
        data = self.merge_mmain_to_clumps(data, mmain)
        flip_cols(data, "peak_x", "peak_z")

        # Cut on number of particles and finite m200. Do not change! Hardcoded
        data = data[(data["npart"] > 100) & numpy.isfinite(data["m200"])]

        # Now also load the initial positions
        if load_init:
            initcm = read_initcm(self.nsim, self.paths.initmatch_path)
            if initcm is not None:
                data = self.merge_initmatch_to_clumps(data, initcm)
                flip_cols(data, "x0", "z0")

#        # Calculate redshift
#        pos = [data["peak_{}".format(p)] - 0.5 for p in ("x", "y", "z")]
#        vel = [data["v{}".format(p)] for p in ("x", "y", "z")]
#        zpec = self.box.box2pecredshift(*vel, *pos)
#        zobs = self.box.box2obsredshift(*vel, *pos)
#        zcosmo = self.box.box2cosmoredshift(
#            sum(pos[i]**2 for i in range(3))**0.5)
#        data = add_columns(data, [zpec, zobs, zcosmo],
#                           ["zpec", "zobs", "zcosmo"])

        # Unit conversion
        convert_cols = ["m200", "m500", "totpartmass", "mass_mmain",
                        "r200", "r500", "Rs", "rho0",
                        "peak_x", "peak_y", "peak_z"]
        data = self.box.convert_from_boxunits(data, convert_cols)

        # Now calculate spherical coordinates
        d, ra, dec = cartesian_to_radec(
            data["peak_x"], data["peak_y"], data["peak_z"])
        data = add_columns(data, [d, ra, dec], ["dist", "ra", "dec"])

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

    def positions(self, in_initial=False):
        r"""
        Cartesian position components of halos in :math:`\mathrm{cMpc}`.

        Parameters
        ----------
        in_initial : bool, optional
            Whether to define the kNN on the initial or final snapshot.

        Returns
        -------
        pos : 2-dimensional array of shape `(nhalos, 3)`
        """
        if in_initial:
            ps = ["x0", "y0", "z0"]
        else:
            ps = ["peak_x", "peak_y", "peak_z"]
        return numpy.vstack([self[p] for p in ps]).T

    def velocities(self):
        """
        Cartesian velocity components of halos. Likely in box units.

        Returns
        -------
        vel : 2-dimensional array of shape `(nhalos, 3)`
        """
        return numpy.vstack([self["v{}".format(p)] for p in ("x", "y", "z")]).T

    def angmomentum(self):
        """
        Cartesian angular momentum components of halos in the box coordinate
        system. Likely in box units.

        Returns
        -------
        angmom : 2-dimensional array of shape `(nhalos, 3)`
        """
        return numpy.vstack([self["L{}".format(p)] for p in ("x", "y", "z")]).T

    def knn(self, in_initial):
        """
        kNN object of all halo positions.

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
        return self._data[key]

    def __len__(self):
        return self.data.size


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
