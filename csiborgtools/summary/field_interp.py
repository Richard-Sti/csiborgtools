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
import numpy
from tqdm import tqdm


###############################################################################
#             Read in the field values at the galaxy positions                #
###############################################################################


def read_interpolated_field(survey_name, kind, galaxy_index, paths, MAS, grid,
                            in_rsp, rand_data=False, verbose=True):
    """
    Read in the interpolated field at the galaxy positions, and reorder the
    data to match the galaxy index.

    Parameters
    ----------
    survey_name : str
        Survey name.
    kind : str
        Field kind.
    galaxy_index : 1-dimensional array
        Galaxy indices to read in.
    paths : py:class:`csiborgtools.read.Paths`
        Paths manager.
    MAS : str
        Mass assignment scheme.
    grid : int
        Grid size.
    in_rsp : bool
        Whether to read in the field in redshift space.
    rand_data : bool, optional
        Whether to read in the random field data instead of the galaxy field.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    3-dimensional array of shape (nsims, len(galaxy_index), nsmooth)
    """
    nsims = paths.get_ics("csiborg")
    for i, nsim in enumerate(tqdm(nsims,
                                  desc="Reading fields",
                                  disable=not verbose)):
        fpath = paths.field_interpolated(
            survey_name, kind, MAS, grid, nsim, in_rsp=in_rsp)
        data = numpy.load(fpath)
        out_ = data["val"] if not rand_data else data["rand_val"]

        if i == 0:
            out = numpy.empty((len(nsims), *out_.shape), dtype=out_.dtype)
            indxs = data["indxs"]

        out[i] = out_

    # Reorder the data to match the survey index.
    ind2pos = {v: k for k, v in enumerate(indxs)}
    ks = numpy.empty(len(galaxy_index), dtype=numpy.int64)

    for i, k in enumerate(galaxy_index):
        j = ind2pos.get(k, None)
        if j is None:
            raise ValueError(f"There is no galaxy with index {k} in the "
                             "interpolated field.")
        ks[i] = j

    return out[:, ks, :]
