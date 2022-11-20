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
from .readsim import (get_sim_path, read_mmain)
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

    def __init__(self, n_sim, n_snap, minimum_m500=None,
                 dumpdir="/mnt/extraspace/rstiskalek/csiborg/",
                 mmain_path="/mnt/zfsusers/hdesmond/Mmain"):
        self._box = BoxUnits(n_snap, get_sim_path(n_sim))
        minimum_m500 = 0 if minimum_m500 is None else minimum_m500
        self._set_data(n_sim, n_snap, dumpdir, mmain_path, minimum_m500)
        self._nsim = n_sim
        self._nsnap = n_snap

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

        # Cut on mass
        data = data[data["m500"] > minimum_m500]

        # Now calculate spherical coordinates
        d, ra, dec = cartesian_to_radec(data)
        data = add_columns(data, [d, ra, dec], ["dist", "ra", "dec"])
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
    def keys(self):
        """Catalogue keys."""
        return self.data.dtype.names

    def __getitem__(self, key):
        return self._data[key]
