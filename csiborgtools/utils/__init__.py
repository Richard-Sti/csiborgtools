# Copyright (C) 2022 Richard Stiskalek
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
from datetime import datetime
from .recarray_manip import (cols_to_structured, add_columns, rm_columns,  # noqa
                             list_to_ndarray, array_to_structured,  # noqa
                             flip_cols, extract_from_structured)  # noqa


def now(tz=None):
    """Shortcut to `datetime.datetime.now`."""
    return datetime.now(tz=tz)


def split_jobs(njobs, ncpu):
    """
    Split `njobs` amongst `ncpu`.

    Parameters
    ----------
    njobs : int
        Number of jobs.
    ncpu : int
        Number of CPUs.

    Returns
    -------
    jobs : list of lists of integers
        Outer list of each CPU and inner lists for CPU's jobs.
    """
    njobs_per_cpu, njobs_remainder = divmod(njobs, ncpu)
    jobs = numpy.arange(njobs_per_cpu * ncpu).reshape((njobs_per_cpu, ncpu)).T
    jobs = jobs.tolist()
    for i in range(njobs_remainder):
        jobs[i].append(njobs_per_cpu * ncpu + i)

    return jobs
