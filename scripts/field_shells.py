
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
"""
NOTE: This script is pretty dodgy.

A script to calculate the mean and standard deviation of a field at different
distances from the center of the box such that at each distance the field is
evaluated at uniformly-spaced points on a sphere.

The script is not parallelized in any way but it should not take very long, the
main bottleneck is reading the data from disk.
"""
from argparse import ArgumentParser
from os.path import join

import csiborgtools
import numpy
from tqdm import tqdm


def main(args):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    boxsize = csiborgtools.simname2boxsize(args.simname)
    distances = numpy.linspace(0, boxsize / 2, 101)[1:]
    nsims = paths.get_ics(args.simname)
    folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/field_shells"

    mus = numpy.zeros((len(nsims), len(distances)))
    stds = numpy.zeros((len(nsims), len(distances)))
    for i, nsim in enumerate(tqdm(nsims, desc="Simulations")):
        # Get the correct field loader
        if args.simname == "csiborg1":
            reader = csiborgtools.read.CSiBORG1Field(nsim, paths)
        elif "csiborg2" in args.simname:
            kind = args.simname.split("_")[-1]
            reader = csiborgtools.read.CSiBORG2Field(nsim, kind, paths)
        elif args.simname == "borg2":
            reader = csiborgtools.read.BORG2Field(nsim, paths)
        else:
            raise ValueError(f"Unknown simname: `{args.simname}`.")

        # Get the field
        if args.field == "density":
            field = reader.density_field(args.MAS, args.grid)
        elif args.field == "overdensity":
            if args.simname == "borg2":
                field = reader.overdensity_field()
            else:
                field = reader.density_field(args.MAS, args.grid)
                csiborgtools.field.overdensity_field(field, make_copy=False)
        elif args.field == "radvel":
            field = reader.radial_velocity_field(args.MAS, args.grid)
        else:
            raise ValueError(f"Unknown field: `{args.field}`.")

        # Evaluate this field at different distances
        vals = [csiborgtools.field.field_at_distance(field, distance, boxsize)
                for distance in distances]

        # Calculate the mean and standard deviation
        mus[i, :] = [numpy.mean(val) for val in vals]
        stds[i, :] = [numpy.std(val) for val in vals]

    # Finally save the output
    fname = f"{args.simname}_{args.field}_{args.MAS}_{args.grid}.npz"
    fname = join(folder, fname)
    numpy.savez(fname, mean=mus, std=stds, distances=distances)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--field", type=str, help="Field type.",
                        choices=["density", "overdensity", "radvel"])
    parser.add_argument("--simname", type=str, help="Simulation name.",
                        choices=["csiborg1", "csiborg2_main", "csiborg2_varysmall", "csiborg2_random", "borg2"])  # noqa
    parser.add_argument("--MAS", type=str, help="Mass assignment scheme.",
                        choices=["NGP", "CIC", "TSC", "PCS", "SPH"])
    parser.add_argument("--grid", type=int, help="Grid size.")
    args = parser.parse_args()

    main(args)
