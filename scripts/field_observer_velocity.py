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

"""
from argparse import ArgumentParser
from datetime import datetime
from os.path import join
from warnings import warn

import csiborgtools
import numpy as np
from astropy.coordinates import SkyCoord
from mpi4py import MPI
from taskmaster import work_delegation

from utils import get_nsims

FDIR = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity/observer"  # noqa


def t():
    return datetime.now()


def read_velocity_field(args, nsim):
    if args.simname == "csiborg1":
        reader = csiborgtools.read.CSiBORG1Field(nsim)
        return reader.velocity_field("SPH", 1024)
    elif "csiborg2" in args.simname:
        kind = args.simname.split("_")[-1]
        reader = csiborgtools.read.CSiBORG2Field(nsim, kind)
        return reader.velocity_field("SPH", 1024)
    elif args.simname == "Carrick2015":
        folder = "/mnt/extraspace/rstiskalek/catalogs"
        warn(f"Using local paths from `{folder}`.", RuntimeWarning)
        fpath = join(folder, "twompp_velocity_carrick2015.npy")
        field = np.load(fpath).astype(np.float32)

        # Because the Carrick+2015 data is in the following form:
        # "The velocities are predicted peculiar velocities in the CMB
        # frame in Galactic Cartesian coordinates, generated from the
        # \(\delta_g^*\) field with \(\beta^* = 0.43\) and an external
        # dipole \(V_\mathrm{ext} = [89,-131,17]\) (Carrick et al Table 3)
        # has already been added.""
        field[0] -= 89
        field[1] -= -131
        field[2] -= 17
        field /= 0.43
        return field
    else:
        raise ValueError(f"Unknown simname: `{args.simname}`.")


def main(smooth_scales, nsim, args):
    velocity_field = read_velocity_field(args, nsim)
    boxsize = csiborgtools.simname2boxsize(args.simname)

    if smooth_scales is None:
        smooth_scales = [0]
    smooth_scales = np.asanyarray(smooth_scales) / boxsize

    vobs = csiborgtools.field.observer_peculiar_velocity(
        velocity_field, smooth_scales=smooth_scales, observer=None,
        verbose=False)

    # For Carrick+2015 the velocity vector is in the Galactic frame, so we
    # need to convert it to RA/dec
    if args.simname == "Carrick2015":
        coord = SkyCoord(vobs, unit='kpc', frame='galactic',
                         representation_type='cartesian').transform_to("icrs")
        vobs = coord.cartesian.xyz.value.T

    fname = join(FDIR, f"{args.simname}_{nsim}_observer_velocity.npz")
    print(f"Saving to `{fname}`.")
    np.savez(fname, vobs=vobs, smooth_scales=smooth_scales * boxsize)


###############################################################################
#                       Main & command line interface                         #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, help="Simulation name.",
                        choices=["csiborg1", "csiborg2_main", "csiborg2_varysmall", "csiborg2_random", "Carrick2015"])  # noqa
    args = parser.parse_args()
    args.nsims = [-1]
    comm = MPI.COMM_WORLD

    smooth_scales = [0, 0.5, 1.0, 2.0, 4.0, 40.0]
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(args, paths)

    def main_(nsim):
        main(smooth_scales, nsim, args)

    work_delegation(main_, nsims, comm, master_verbose=True)

    comm.Barrier()

    if comm.Get_rank() == 0:
        print("All finished.", flush=True)
