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


def read_interpolated_field(survey, simname, kind, MAS, grid, paths,
                            verbose=True):
    """
    Read in the interpolated field at the galaxy positions, and reorder the
    data to match the galaxy index.

    Parameters
    ----------
    survey : Survey
        Survey object.
    simname : str
        Simulation name.
    kind : str
        Field kind.
    MAS : str
        Mass assignment scheme.
    grid : int
        Grid size.
    paths : py:class:`csiborgtools.read.Paths`
        Paths manager.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    val : 3-dimensional array of shape (nsims, num_gal, nsmooth)
        Scalar field values at the galaxy positions.
    smooth_scales : 1-dimensional array
        Smoothing scales.
    """
    nsims = paths.get_ics(simname)

    for i, nsim in enumerate(tqdm(nsims,
                                  desc="Reading fields",
                                  disable=not verbose)):
        fpath = paths.field_interpolated(survey.name, simname, nsim, kind, MAS,
                                         grid)
        data = numpy.load(fpath)
        out_ = data["val"]

        if i == 0:
            out = numpy.empty((len(nsims), *out_.shape), dtype=out_.dtype)
            smooth_scales = data["smooth_scales"]

        out[i] = out_

    if survey.selection_mask is not None:
        out = out[:, survey.selection_mask, :]

    return out, smooth_scales
