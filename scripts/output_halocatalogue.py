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
"""
Quick script to output either halo positions and masses or positions of
galaxies in a survey as an ASCII file.
"""
from os.path import join

import csiborgtools
import numpy
from tqdm import tqdm

DIR_OUT = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/ascii_positions"


def process_simulation(simname):
    """Watch out about the distinction between real and redshift space."""
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    if "csiborg2" in simname:
        nsims = paths.get_ics(simname)
        kind = simname.split("_")[-1]

        for nsim in tqdm(nsims, desc="Looping over simulations"):
            cat = csiborgtools.read.CSiBORG2Catalogue(nsim, 99, kind, paths)
            pos = cat["cartesian_pos"]
            mass = cat["totmass"]
            # Stack positions and masses
            x = numpy.hstack([pos, mass.reshape(-1, 1)])

            # Save to a file
            fname = join(DIR_OUT, f"halos_real_{simname}_{nsim}.txt")
            numpy.savetxt(fname, x)
    else:
        raise RuntimeError("Simulation not implemented..")


def process_survey(survey_name, boxsize):
    """Watch out about the distance definition."""
    if survey_name == "SDSS":
        survey = csiborgtools.SDSS()()
        dist, ra, dec = survey["DIST"], survey["RA"], survey["DEC"]
    elif survey_name == "SDSSxALFALFA":
        survey = csiborgtools.SDSSxALFALFA()()
        dist, ra, dec = survey["DIST"], survey["RA_1"], survey["DEC_1"]
    else:
        raise RuntimeError("Survey not implemented..")

    # Convert to Cartesian coordinates
    X = numpy.vstack([dist, ra, dec]).T
    X = csiborgtools.radec_to_cartesian(X)

    # Center the coordinates in the box
    X += boxsize / 2

    fname = join(DIR_OUT, f"survey_{survey_name}.txt")
    numpy.savetxt(fname, X)


if __name__ == "__main__":
    # process_simulation("csiborg2_main")

    boxsize = 676.6
    for survey in ["SDSS", "SDSSxALFALFA"]:
        process_survey(survey, boxsize)
