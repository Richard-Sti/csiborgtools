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
import numpy as np
from tqdm import tqdm

DIR_OUT = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/ascii_positions"


def write_simulation(simname, high_resolution_only=True):
    """"
    Write the positions, velocities and IDs of the particles in a simulation
    to an ASCII file. The output is `X Y Z VX VY VZ ID`.
    """
    if not high_resolution_only:
        raise RuntimeError("Writing low-res particles is not implemented.")

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    if "csiborg2" in simname:
        nsims = paths.get_ics(simname)
        kind = simname.split("_")[-1]

        for nsim in tqdm(nsims, desc="Simulations"):
            reader = csiborgtools.read.CSiBORG2Snapshot(nsim, 99, kind, paths,
                                                        flip_xz=False)
            x = np.hstack(
                [reader.coordinates(high_resolution_only=True),
                 reader.velocities(high_resolution_only=True),
                 reader.particle_ids(high_resolution_only=True).reshape(-1, 1)]
                 )
            # Store positions and velocities with 6 decimal places and IDs as
            # integers.
            fmt_string = "%1.6f %1.6f %1.6f %1.6f %1.6f %1.6f %d"
            fname = join(DIR_OUT, f"high_res_particles_{simname}_{nsim}.txt")
            np.savetxt(fname, x, fmt=fmt_string)
    else:
        raise RuntimeError("Simulation not implemented..")


def write_halos(simname):
    """
    Watch out about the distinction between real and redshift space. The output
    is `X Y Z MASS`.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    if "csiborg2" in simname:
        nsims = paths.get_ics(simname)
        kind = simname.split("_")[-1]

        for nsim in tqdm(nsims, desc="Looping over simulations"):
            cat = csiborgtools.read.CSiBORG2Catalogue(nsim, 99, kind, paths)
            pos = cat["cartesian_pos"]
            mass = cat["totmass"]
            # Stack positions and masses
            x = np.hstack([pos, mass.reshape(-1, 1)])

            # Save to a file
            fname = join(DIR_OUT, f"halos_real_{simname}_{nsim}.txt")
            np.savetxt(fname, x)
    else:
        raise RuntimeError("Simulation not implemented..")


def write_survey(survey_name, boxsize):
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
    X = np.vstack([dist, ra, dec]).T
    X = csiborgtools.radec_to_cartesian(X)

    # Center the coordinates in the box
    X += boxsize / 2

    fname = join(DIR_OUT, f"survey_{survey_name}.txt")
    np.savetxt(fname, X)


if __name__ == "__main__":
    write_simulation("csiborg2_main")

    # write_halos("csiborg2_main")

    # boxsize = 676.6
    # for survey in ["SDSS", "SDSSxALFALFA"]:
    #     write_survey(survey, boxsize)
