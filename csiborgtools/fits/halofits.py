# Copyright (C) 2022 Richard Stiskalek, Deaglan Bartlett
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
Tools for fitting the halos and distributing the jobs.
"""


import numpy
from ..io import nparts_to_start_ind


def distribute_halos(Njobs, clump_indxs):
    """
    Evenly distribute clump indices to CPUs.

    Parameters
    ----------
    Njobs : int
        Number of jobs.
    clump_indxs : 1-dimensional array
        Array of clump indices.

    Returns
    -------
    start : 1-dimensional array
        The starting index of each CPU.
    """
    # Make sure these are unique IDs
    if clump_indxs.size > numpy.unique((clump_indxs)):
        raise ValueError("`clump_indxs` constains duplicate indices.")
    Ntotal = clump_indxs.size
    Njobs_per_cpu = numpy.ones(Njobs, dtype=int) * Ntotal // Njobs
    # Split the remainder Ntotal % Njobs among the CPU
    Njobs_per_cpu[:Ntotal % Njobs] += 1
    start = nparts_to_start_ind(Njobs_per_cpu)
    return start
