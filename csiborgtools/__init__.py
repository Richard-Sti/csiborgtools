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
from csiborgtools import clustering, field, match, read, summary                 # noqa

from .utils import (center_of_mass, delta2ncells, number_counts,                 # noqa
                    periodic_distance, periodic_distance_two_points,             # noqa
                    binned_statistic, cosine_similarity, fprint)                 # noqa


# Arguments to csiborgtools.read.Paths.
paths_glamdring = {"srcdir": "/mnt/extraspace/hdesmond/",
                   "postdir": "/mnt/extraspace/rstiskalek/CSiBORG/",
                   "borg_dir": "/users/hdesmond/BORG_final/",
                   "quijote_dir": "/mnt/extraspace/rstiskalek/Quijote",
                   }


neighbour_kwargs = {"rmax_radial": 155 / 0.705,
                    "nbins_radial": 50,
                    "rmax_neighbour": 100.,
                    "nbins_neighbour": 150,
                    "paths_kind": paths_glamdring}


###############################################################################
#                             Surveys                                         #
###############################################################################

class SDSS:
    @staticmethod
    def steps(cls):
        return [(lambda x: cls[x], ("IN_DR7_LSS",)),
                (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
                (lambda x: cls[x] < 155, ("DIST", ))
                ]

    def __call__(self):
        return read.SDSS(h=1, sel_steps=self.steps)
