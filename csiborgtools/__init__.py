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
from csiborgtools import clustering, field, halo, match, read, summary          # noqa

from .utils import (center_of_mass, delta2ncells, number_counts,                # noqa
                    periodic_distance, periodic_distance_two_points,            # noqa
                    binned_statistic, cosine_similarity, fprint,                # noqa
                    hms_to_degrees, dms_to_degrees, great_circle_distance)      # noqa
from .params import paths_glamdring, simname2boxsize                            # noqa


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

    def __call__(self, fpath=None, apply_selection=True):
        if fpath is None:
            fpath = "/mnt/extraspace/rstiskalek/catalogs/nsa_v1_0_1.fits"
        sel_steps = self.steps if apply_selection else None
        return read.SDSS(fpath, h=1, sel_steps=sel_steps)


class SDSSxALFALFA:
    @staticmethod
    def steps(cls):
        return [(lambda x: cls[x], ("IN_DR7_LSS",)),
                (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
                (lambda x: cls[x] < 155, ("DIST", ))
                ]

    def __call__(self, fpath=None, apply_selection=True):
        if fpath is None:
            fpath = "/mnt/extraspace/rstiskalek/catalogs/5asfullmatch.fits"
        sel_steps = self.steps if apply_selection else None
        survey = read.SDSS(fpath, h=1, sel_steps=sel_steps)
        survey.name = "SDSSxALFALFA"
        return survey


###############################################################################
#                              Clusters                                       #
###############################################################################

clusters = {"Virgo": read.ObservedCluster(RA=hms_to_degrees(12, 27),
                                          dec=dms_to_degrees(12, 43),
                                          dist=16.5 * 0.7,
                                          name="Virgo"),
            "Fornax": read.ObservedCluster(RA=hms_to_degrees(3, 38),
                                           dec=dms_to_degrees(-35, 27),
                                           dist=19 * 0.7,
                                           name="Fornax"),
            }
