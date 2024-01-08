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
"""MPI script to calculate the various fields."""
from argparse import ArgumentParser
from datetime import datetime

import numpy
from mpi4py import MPI
from taskmaster import work_delegation

import csiborgtools
from utils import get_nsims


###############################################################################
#                            Density field                                    #
###############################################################################


def density_field(nsim, parser_args):
    """Calculate the density field."""
    if parser_args.MAS == "SPH":
        raise NotImplementedError("SPH is not implemented here. Use cosmotool")

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, parser_args.simname))

    # Read in the particle coordinates and masses
    if parser_args.simname == "csiborg1":
        snapshot = csiborgtools.read.CSIBORG1Snapshot(nsim, nsnap, paths)
    elif "csiborg2" in parser_args.simname:
        kind = parser_args.simname.split("_")[-1]
        snapshot = csiborgtools.read.CSIBORG2Snapshot(nsim, nsnap, paths, kind)
    elif parser_args.simname == "quijote":
        snapshot = csiborgtools.read.QuijoteSnapshot(nsim, nsnap, paths)
    else:
        raise RuntimeError(f"Unknown simulation name `{parser_args.simname}`.")

    pos = snapshot.coordinates()
    mass = snapshot.masses()

    # Run the field generator
    boxsize = csiborgtools.simname2boxsize(parser_args.simname)
    gen = csiborgtools.field.DensityField(boxsize, parser_args.MAS)
    field = gen(pos, mass, parser_args.grid)

    fout = paths.field("density", parser_args.MAS, parser_args.grid,
                       nsim, parser_args.simname)

    print(f"{datetime.now()}: saving output to `{fout}`.")
    numpy.save(fout, field)
    return field


###############################################################################
#                            Velocity field                                   #
###############################################################################


def velocity_field(nsim, parser_args):
    """Calculate the velocity field."""
    if parser_args.MAS == "SPH":
        raise NotImplementedError("SPH is not implemented here. Use cosmotool")

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, parser_args.simname))

    if parser_args.simname == "csiborg1":
        snapshot = csiborgtools.read.CSIBORG1Snapshot(nsim, nsnap, paths)
    elif "csiborg2" in parser_args.simname:
        kind = parser_args.simname.split("_")[-1]
        snapshot = csiborgtools.read.CSIBORG2Snapshot(nsim, nsnap, paths, kind)
    elif parser_args.simname == "quijote":
        snapshot = csiborgtools.read.QuijoteSnapshot(nsim, nsnap, paths)
    else:
        raise RuntimeError(f"Unknown simulation name `{parser_args.simname}`.")

    pos = snapshot.coordinates()
    vel = snapshot.velocities()
    mass = snapshot.masses()

    boxsize = csiborgtools.simname2boxsize(parser_args.simname)
    gen = csiborgtools.field.VelocityField(boxsize, parser_args.MAS)
    field = gen(pos, vel, mass, parser_args.grid)

    fout = paths.field("velocity", parser_args.MAS, parser_args.grid,
                       nsim, parser_args.simname)
    print(f"{datetime.now()}: saving output to `{fout}`.")
    numpy.save(fout, field)
    return field


###############################################################################
#                          Radial velocity field                              #
###############################################################################


def radvel_field(nsim, parser_args):
    """Calculate the radial velocity field."""
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    if parser_args.simname == "csiborg1":
        field = csiborgtools.read.CSiBORG1Field(nsim, paths)
    elif "csiborg2" in parser_args.simname:
        kind = parser_args.simname.split("_")[-1]
        field = csiborgtools.read.CSiBORG2Field(nsim, kind, paths)
    elif parser_args.simname == "quijote":
        field = csiborgtools.read.QuijoteField(nsim, paths)
    else:
        raise RuntimeError(f"Unknown simulation name `{parser_args.simname}`.")

    vel = field.velocity_field(parser_args.MAS, parser_args.grid)

    observer_velocity = csiborgtools.field.observer_peculiar_velocity(vel)
    radvel = csiborgtools.field.radial_velocity(vel, observer_velocity)

    fout = paths.field("radvel", parser_args.MAS, parser_args.grid,
                       nsim, parser_args.simname)
    print(f"{datetime.now()}: saving output to `{fout}`.")
    numpy.save(fout, radvel)
    return field


def observer_peculiar_velocity(nsim, parser_args):
    """
    Calculate the peculiar velocity of an observer in the centre of the box
    for several smoothing scales.
    """
    boxsize = csiborgtools.simname2boxsize(parser_args.simname)
    # NOTE these values are hard-coded.
    smooth_scales = numpy.array([0., 2.0, 4.0, 8.0, 16.])
    smooth_scales /= boxsize

    if parser_args.simname == "csiborg1":
        field = csiborgtools.read.CSiBORG1Field(nsim, paths)
    elif "csiborg2" in parser_args.simname:
        kind = parser_args.simname.split("_")[-1]
        field = csiborgtools.read.CSiBORG2Field(nsim, paths, kind)
    elif parser_args.simname == "quijote":
        field = csiborgtools.read.QuijoteField(nsim, paths)
    else:
        raise RuntimeError(f"Unknown simulation name `{parser_args.simname}`.")

    vel = field.velocity_field(parser_args.MAS, parser_args.grid)

    observer_vp = csiborgtools.field.observer_peculiar_velocity(
        vel, smooth_scales)

    fout = paths.observer_peculiar_velocity(parser_args.MAS, parser_args.grid,
                                            nsim, parser_args.simname)
    print(f"Saving to ... `{fout}`")
    numpy.savez(fout, smooth_scales=smooth_scales, observer_vp=observer_vp)
    return observer_vp


###############################################################################
#                          Command line interface                             #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. `-1` for all simulations.")
    parser.add_argument("--simname", type=str, help="Simulation name.")
    parser.add_argument("--kind", type=str,
                        choices=["density", "velocity", "radvel", "observer_vp"],  # noqa
                        help="What derived field to calculate?")
    parser.add_argument("--MAS", type=str,
                        choices=["NGP", "CIC", "TSC", "PCS", "SPH"],
                        help="Mass assignment scheme.")
    parser.add_argument("--grid", type=int, help="Grid resolution.")
    parser_args = parser.parse_args()

    comm = MPI.COMM_WORLD
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(parser_args, paths)

    def main(nsim):
        if parser_args.kind == "density":
            density_field(nsim, parser_args)
        elif parser_args.kind == "velocity":
            velocity_field(nsim, parser_args)
        elif parser_args.kind == "radvel":
            radvel_field(nsim, parser_args)
        elif parser_args.kind == "observer_vp":
            observer_peculiar_velocity(nsim, parser_args)
        else:
            raise RuntimeError(f"Field {parser_args.kind} is not implemented.")

    work_delegation(main, nsims, comm, master_verbose=True)


# def potential_field(nsim, parser_args, to_save=True):
#     """
#     Calculate the potential field in the CSiBORG simulation.
#     """
#     paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
#     nsnap = max(paths.get_snapshots(nsim, "csiborg"))
#     box = csiborgtools.read.CSiBORG1Box(nsnap, nsim, paths)
#
#     if not parser_args.in_rsp:
#         rho = numpy.load(paths.field(
#             "density", parser_args.MAS, parser_args.grid, nsim,
# in_rsp=False))
#         density_gen = csiborgtools.field.DensityField(box, parser_args.MAS)
#         rho = density_gen.overdensity_field(rho)
#
#         gen = csiborgtools.field.PotentialField(box, parser_args.MAS)
#         field = gen(rho)
#     else:
#         field = numpy.load(paths.field(
#             "potential", parser_args.MAS, parser_args.grid, nsim, False))
#         radvel_field = numpy.load(paths.field(
#             "radvel", parser_args.MAS, parser_args.grid, nsim, False))
#
#         field = csiborgtools.field.field2rsp(field, radvel_field, box,
#                                              parser_args.MAS)
#
#     if to_save:
#         fout = paths.field(parser_args.kind, parser_args.MAS,
# parser_args.grid,
#                            nsim, parser_args.in_rsp)
#         print(f"{datetime.now()}: saving output to `{fout}`.")
#         numpy.save(fout, field)
#     return field
#
#
# #############################################################################
# #                        Environment classification                         #
# #############################################################################
#
#
# def environment_field(nsim, parser_args, to_save=True):
#     """
#     Calculate the environmental classification in the CSiBORG simulation.
#     """
#     paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
#     nsnap = max(paths.get_snapshots(nsim, "csiborg"))
#     box = csiborgtools.read.CSiBORG1Box(nsnap, nsim, paths)
#
#     rho = numpy.load(paths.field(
#         "density", parser_args.MAS, parser_args.grid, nsim, in_rsp=False))
#     density_gen = csiborgtools.field.DensityField(box, parser_args.MAS)
#     rho = density_gen.overdensity_field(rho)
#
#     if parser_args.smooth_scale > 0.0:
#         rho = csiborgtools.field.smoothen_field(
#             rho, parser_args.smooth_scale, box.box2mpc(1.))
#
#     gen = csiborgtools.field.TidalTensorField(box, parser_args.MAS)
#     field = gen(rho)
#
#     del rho
#     collect()
#
#     if parser_args.in_rsp:
#         radvel_field = numpy.load(paths.field(
#             "radvel", parser_args.MAS, parser_args.grid, nsim, False))
#         args = (radvel_field, box, parser_args.MAS)
#
#         field.T00 = csiborgtools.field.field2rsp(field.T00, *args)
#         field.T11 = csiborgtools.field.field2rsp(field.T11, *args)
#         field.T22 = csiborgtools.field.field2rsp(field.T22, *args)
#         field.T01 = csiborgtools.field.field2rsp(field.T01, *args)
#         field.T02 = csiborgtools.field.field2rsp(field.T02, *args)
#         field.T12 = csiborgtools.field.field2rsp(field.T12, *args)
#
#         del radvel_field
#         collect()
#
#     eigvals = gen.tensor_field_eigvals(field)
#
#     del field
#     collect()
#
#     env = gen.eigvals_to_environment(eigvals)
#
#     if to_save:
#         fout = paths.field("environment", parser_args.MAS, parser_args.grid,
#                            nsim, parser_args.in_rsp,
# parser_args.smooth_scale)
#         print(f"{datetime.now()}: saving output to `{fout}`.")
#         numpy.save(fout, env)
#     return env
