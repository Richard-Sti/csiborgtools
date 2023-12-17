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
MPI script to calculate density field-derived fields in the CSiBORG
simulations' final snapshot.
"""
from argparse import ArgumentParser
from datetime import datetime
from distutils.util import strtobool
from gc import collect

import numpy
from mpi4py import MPI
from taskmaster import work_delegation

import csiborgtools
from utils import get_nsims


from h5py import File


# EVERYTHING HERE NEEDS TO BE UPDATED.

###############################################################################
#                            Density field                                    #
###############################################################################

# TODO do RSP separately | routines that read the particles are also old.


def density_field(nsim, parser_args):
    """
    Calculate the density field in the CSiBORG simulation.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, parser_args.simname))

    boxsize = csiborgtools.simname2boxsize(parser_args.simname)

    fname = paths.snapshot(nsnap, nsim, parser_args.simname)

    snapshot = File(fname, 'r')

    gen = csiborgtools.field.DensityField(boxsize, parser_args.MAS)

    if parser_args.simname == "csiborg1":
        pos = snapshot["Coordinates"]
        mass = snapshot["Masses"]

        field = gen(pos, mass, parser_args.grid, verbose=parser_args.verbose)
    elif "csiborg2" in parser_args.simname:
        # First do the high-resolution particles
        pos = snapshot["PartType1"]["Coordinates"]
        mass = numpy.ones(len(pos), dtype=pos.dtype)
        mass *= snapshot.header.attrs["MassTable"][1]
        field = gen(pos, mass, parser_args.grid, verbose=parser_args.verbose)

        # And then the low-resolution particles
        pos = snapshot["PartType5"]["Coordinates"]
        mass = snapshot["PartType5"]["Masses"]
        field += gen(pos, mass, parser_args.grid, verbose=parser_args.verbose)

        # Convert to `Msun / h`
        field *= 1e10
    else:
        raise ValueError(f"Unknown simname: {parser_args.simname}")

    # Normalize so that the field is in units of `h^2 Msun / kpc^3`
    field /= (boxsize * 1000 / parser_args.grid)**3


    # TODO clear out these paths.
    fout = paths.field(parser_args.kind, parser_args.MAS, parser_args.grid,
                        nsim, parser_args.in_rsp)
    print(f"{datetime.now()}: saving output to `{fout}`.")
    # Save as .hdf5?
    numpy.save(fout, field)
    return field


###############################################################################
#                            Velocity field                                   #
###############################################################################


def velocity_field(nsim, parser_args, to_save=True):
    """
    Calculate the velocity field in a CSiBORG simulation.
    """
    if parser_args.in_rsp:
        raise NotImplementedError("Velocity field in RSP is not implemented.")

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, "csiborg"))
    box = csiborgtools.read.CSiBORG1Box(nsnap, nsim, paths)
    fname = paths.processed_output(nsim, "csiborg", "halo_catalogue")

    snap = csiborgtools.read.read_h5(fname)["snapshot_final"]
    pos = snap["pos"]
    vel = snap["vel"]
    mass = snap["mass"]

    gen = csiborgtools.field.VelocityField(box, parser_args.MAS)
    field = gen(pos, vel, mass, parser_args.grid, verbose=parser_args.verbose)

    if to_save:
        fout = paths.field("velocity", parser_args.MAS, parser_args.grid,
                           nsim, in_rsp=False)
        print(f"{datetime.now()}: saving output to `{fout}`.")
        numpy.save(fout, field)
    return field


###############################################################################
#                          Radial velocity field                              #
###############################################################################


def radvel_field(nsim, parser_args, to_save=True):
    """
    Calculate the radial velocity field in the CSiBORG simulation.
    """
    if parser_args.in_rsp:
        raise NotImplementedError("Radial vel. field in RSP not implemented.")

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, "csiborg"))
    box = csiborgtools.read.CSiBORG1Box(nsnap, nsim, paths)

    vel = numpy.load(paths.field("velocity", parser_args.MAS, parser_args.grid,
                                 nsim, parser_args.in_rsp))
    observer_velocity = csiborgtools.field.observer_vobs(vel)

    gen = csiborgtools.field.VelocityField(box, parser_args.MAS)
    field = gen.radial_velocity(vel, observer_velocity)

    if to_save:
        fout = paths.field("radvel", parser_args.MAS, parser_args.grid,
                           nsim, parser_args.in_rsp)
        print(f"{datetime.now()}: saving output to `{fout}`.")
        numpy.save(fout, field)
    return field

###############################################################################
#                          Potential field                                    #
###############################################################################


def potential_field(nsim, parser_args, to_save=True):
    """
    Calculate the potential field in the CSiBORG simulation.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, "csiborg"))
    box = csiborgtools.read.CSiBORG1Box(nsnap, nsim, paths)

    if not parser_args.in_rsp:
        rho = numpy.load(paths.field(
            "density", parser_args.MAS, parser_args.grid, nsim, in_rsp=False))
        density_gen = csiborgtools.field.DensityField(box, parser_args.MAS)
        rho = density_gen.overdensity_field(rho)

        gen = csiborgtools.field.PotentialField(box, parser_args.MAS)
        field = gen(rho)
    else:
        field = numpy.load(paths.field(
            "potential", parser_args.MAS, parser_args.grid, nsim, False))
        radvel_field = numpy.load(paths.field(
            "radvel", parser_args.MAS, parser_args.grid, nsim, False))

        field = csiborgtools.field.field2rsp(field, radvel_field, box,
                                             parser_args.MAS)

    if to_save:
        fout = paths.field(parser_args.kind, parser_args.MAS, parser_args.grid,
                           nsim, parser_args.in_rsp)
        print(f"{datetime.now()}: saving output to `{fout}`.")
        numpy.save(fout, field)
    return field


###############################################################################
#                        Environment classification                           #
###############################################################################


def environment_field(nsim, parser_args, to_save=True):
    """
    Calculate the environmental classification in the CSiBORG simulation.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, "csiborg"))
    box = csiborgtools.read.CSiBORG1Box(nsnap, nsim, paths)

    rho = numpy.load(paths.field(
        "density", parser_args.MAS, parser_args.grid, nsim, in_rsp=False))
    density_gen = csiborgtools.field.DensityField(box, parser_args.MAS)
    rho = density_gen.overdensity_field(rho)

    if parser_args.smooth_scale > 0.0:
        rho = csiborgtools.field.smoothen_field(
            rho, parser_args.smooth_scale, box.box2mpc(1.))

    gen = csiborgtools.field.TidalTensorField(box, parser_args.MAS)
    field = gen(rho)

    del rho
    collect()

    if parser_args.in_rsp:
        radvel_field = numpy.load(paths.field(
            "radvel", parser_args.MAS, parser_args.grid, nsim, False))
        args = (radvel_field, box, parser_args.MAS)

        field.T00 = csiborgtools.field.field2rsp(field.T00, *args)
        field.T11 = csiborgtools.field.field2rsp(field.T11, *args)
        field.T22 = csiborgtools.field.field2rsp(field.T22, *args)
        field.T01 = csiborgtools.field.field2rsp(field.T01, *args)
        field.T02 = csiborgtools.field.field2rsp(field.T02, *args)
        field.T12 = csiborgtools.field.field2rsp(field.T12, *args)

        del radvel_field
        collect()

    eigvals = gen.tensor_field_eigvals(field)

    del field
    collect()

    env = gen.eigvals_to_environment(eigvals)

    if to_save:
        fout = paths.field("environment", parser_args.MAS, parser_args.grid,
                           nsim, parser_args.in_rsp, parser_args.smooth_scale)
        print(f"{datetime.now()}: saving output to `{fout}`.")
        numpy.save(fout, env)
    return env


def observer_peculiar_velocity(nsim, parser_args):
    """
    Calculate the peculiar velocity of an observer in the centre of the box
    for several smoothing scales.
    """
    pos = numpy.array([0.5, 0.5, 0.5]).reshape(-1, 3)
    boxsize = 677.7
    smooth_scales = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    observer_vp = numpy.full((len(smooth_scales), 3), numpy.nan,
                             dtype=numpy.float32)

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    field_path = paths.field("velocity", parser_args.MAS, parser_args.grid,
                             nsim, in_rsp=False)
    field0 = numpy.load(field_path)

    for j, smooth_scale in enumerate(tqdm(smooth_scales,
                                          desc="Smoothing the fields",
                                          disable=not parser_args.verbose)):
        if smooth_scale > 0:
            field = [None, None, None]
            for k in range(3):
                field[k] = csiborgtools.field.smoothen_field(
                    field0[k], smooth_scale, boxsize)
        else:
            field = field0

        v = csiborgtools.field.evaluate_cartesian(
            field[0], field[1], field[2], pos=pos)
        observer_vp[j, 0] = v[0][0]
        observer_vp[j, 1] = v[1][0]
        observer_vp[j, 2] = v[2][0]

    fout = paths.observer_peculiar_velocity(parser_args.MAS, parser_args.grid,
                                            nsim)
    if parser_args.verbose:
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
    parser.add_argument("--simname", type=str, default="csiborg",
                        choices=["csiborg", "csiborg2"],
                        help="Verbosity flag for reading in particles.")
    parser.add_argument("--kind", type=str,
                        choices=["density", "rspdensity", "velocity", "radvel",
                                 "potential", "environment"],
                        help="What derived field to calculate?")
    parser.add_argument("--MAS", type=str,
                        choices=["NGP", "CIC", "TSC", "PCS"])
    parser.add_argument("--grid", type=int, help="Grid resolution.")
    # parser.add_argument("--in_rsp", type=lambda x: bool(strtobool(x)),
    #                     help="Calculate in RSP?")
    parser.add_argument("--smooth_scale", type=float, default=0.0,
                        help="Smoothing scale in Mpc / h. Only used for the environment field.")  # noqa
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        help="Verbosity flag for reading in particles.")
    parser_args = parser.parse_args()
    comm = MPI.COMM_WORLD
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(parser_args, paths)

    def main(nsim):
        if parser_args.kind == "density" or parser_args.kind == "rspdensity":
            density_field(nsim, parser_args)
        elif parser_args.kind == "velocity":
            velocity_field(nsim, parser_args)
        elif parser_args.kind == "radvel":
            radvel_field(nsim, parser_args)
        elif parser_args.kind == "potential":
            potential_field(nsim, parser_args)
        elif parser_args.kind == "environment":
            environment_field(nsim, parser_args)
        else:
            raise RuntimeError(f"Field {parser_args.kind} is not implemented.")

    work_delegation(main, nsims, comm, master_verbose=True)


def density_field(nsim, parser_args, to_save=True):
    """
    Calculate the density field in the CSiBORG simulation.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, "csiborg"))
    box = csiborgtools.read.CSiBORG1Box(nsnap, nsim, paths)
    fname = paths.processed_output(nsim, "csiborg", "halo_catalogue")

    if not parser_args.in_rsp:
        # TODO I removed this function
        snap = csiborgtools.read.read_h5(fname)["snapshot_final"]
        pos = snap["pos"]
        mass = snap["mass"]

        gen = csiborgtools.field.DensityField(box, parser_args.MAS)
        field = gen(pos, mass, parser_args.grid, verbose=parser_args.verbose)
    else:
        field = numpy.load(paths.field(
            "density", parser_args.MAS, parser_args.grid, nsim, False))
        radvel_field = numpy.load(paths.field(
            "radvel", parser_args.MAS, parser_args.grid, nsim, False))

        if parser_args.verbose:
            print(f"{datetime.now()}: converting density field to RSP.",
                  flush=True)

        field = csiborgtools.field.field2rsp(field, radvel_field, box,
                                             parser_args.MAS)

    if to_save:
        fout = paths.field(parser_args.kind, parser_args.MAS, parser_args.grid,
                           nsim, parser_args.in_rsp)
        print(f"{datetime.now()}: saving output to `{fout}`.")
        numpy.save(fout, field)
    return field