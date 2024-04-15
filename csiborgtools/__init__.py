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
from csiborgtools import clustering, field, flow, halo, match, read, summary    # noqa

from .clusters import clusters                                                  # noqa
from .utils import (center_of_mass, delta2ncells, number_counts,                # noqa
                    periodic_distance, periodic_distance_two_points,            # noqa
                    binned_statistic, cosine_similarity, fprint,                # noqa
                    hms_to_degrees, dms_to_degrees, great_circle_distance,      # noqa
                    radec_to_cartesian, cartesian_to_radec,                     # noqa
                    thin_samples_by_acl, numpyro_gof, radec_to_galactic)        # noqa
from .params import (paths_glamdring, simname2boxsize, simname2Omega_m,         # noqa
                     snap2redshift)                                             # noqa


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
