# Copyright (C) 2023 Richard Stiskalek
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
"""2PCF reader."""
from os.path import join
from glob import glob
import numpy
import joblib


class TPCFReader:
    """
    Shortcut object to read in the 2PCF data.
    """
    def read(self, run, folder):
        """
        Read the auto- or cross-correlation kNN-CDF data. Infers the type from
        the data files.

        Parameters
        ----------
        run : str
            Run ID to read in.
        folder : str
            Path to the folder where the auto-2PCF is stored.

        Returns
        -------
        rp : 1-dimensional array of shape `(neval, )`
            Projected separations where the 2PCF is evaluated.
        out : 2-dimensional array of shape `(len(files), len(rp))`
            Array of 2PCFs.
        """
        run += ".p"
        files = [f for f in glob(join(folder, "*")) if run in f]
        if len(files) == 0:
            raise RuntimeError("No files found for run `{}`.".format(run[:-2]))

        for i, file in enumerate(files):
            data = joblib.load(file)
            if i == 0:  # Initialise the array
                rp = data["rp"]
                out = numpy.full((len(files), rp.size), numpy.nan,
                                 dtype=numpy.float32)
            out[i, ...] = data["wp"]

        return rp, out
