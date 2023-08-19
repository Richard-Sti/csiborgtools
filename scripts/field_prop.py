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

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools

from taskmaster import work_delegation

from utils import get_nsims

###############################################################################
#                            Density field                                    #
###############################################################################


def density_field(nsim, parser_args, to_save=True):
    """
    Calculate the density field in the CSiBORG simulation.

    Parameters
    ----------
    nsim : int
        Simulation index.
    parser_args : argparse.Namespace
        Parsed arguments.
    to_save : bool, optional
        Whether to save the output to disk.

    Returns
    -------
    field : 3-dimensional array
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, "csiborg"))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)
    parts = csiborgtools.read.read_h5(paths.particles(nsim, "csiborg"))
    parts = parts["particles"]
    gen = csiborgtools.field.DensityField(box, parser_args.MAS)

    if parser_args.kind == "density":
        field = gen(parts, parser_args.grid, in_rsp=False,
                    verbose=parser_args.verbose)
        if parser_args.in_rsp:
            field = csiborgtools.field.field2rsp(*field, parts=parts, box=box,
                                                 verbose=parser_args.verbose)
    else:
        field = gen(parts, parser_args.grid, in_rsp=parser_args.in_rsp,
                    verbose=parser_args.verbose)

    if to_save:
        fout = paths.field(parser_args.kind, parser_args.MAS, parser_args.grid,
                           nsim, parser_args.in_rsp)
        print(f"{datetime.now()}: saving output to `{fout}`.")
        numpy.save(fout, field)
    return field


###############################################################################
#                            Velocity field                                   #
###############################################################################


def velocity_field(nsim, parser_args, to_save=True):
    """
    Calculate the velocity field in a CSiBORG simulation.

    Parameters
    ----------
    nsim : int
        Simulation index.
    parser_args : argparse.Namespace
        Parsed arguments.
    to_save : bool, optional
        Whether to save the output to disk.

    Returns
    -------
    velfield : 4-dimensional array
    """
    if parser_args.in_rsp:
        raise NotImplementedError("Velocity field in RSP is not implemented.")

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, "csiborg"))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)
    parts = csiborgtools.read.read_h5(paths.particles(nsim, "csiborg"))
    parts = parts["particles"]

    gen = csiborgtools.field.VelocityField(box, parser_args.MAS)
    field = gen(parts, parser_args.grid, verbose=parser_args.verbose)

    if to_save:
        fout = paths.field("velocity", parser_args.MAS, parser_args.grid,
                           nsim, in_rsp=False)
        print(f"{datetime.now()}: saving output to `{fout}`.")
        numpy.save(fout, field)
    return field


###############################################################################
#                          Potential field                                    #
###############################################################################


def potential_field(nsim, parser_args, to_save=True):
    """
    Calculate the potential field in the CSiBORG simulation.

    Parameters
    ----------
    nsim : int
        Simulation index.
    parser_args : argparse.Namespace
        Parsed arguments.
    to_save : bool, optional
        Whether to save the output to disk.

    Returns
    -------
    potential : 3-dimensional array
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, "csiborg"))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)

    # Load the real space overdensity field
    density_gen = csiborgtools.field.DensityField(box, parser_args.MAS)
    rho = numpy.load(paths.field("density", parser_args.MAS, parser_args.grid,
                                 nsim, in_rsp=False))
    rho = density_gen.overdensity_field(rho)
    # Calculate the real space potentiel field
    gen = csiborgtools.field.PotentialField(box, parser_args.MAS)
    field = gen(rho)

    if parser_args.in_rsp:
        parts = csiborgtools.read.read_h5(paths.particles(nsim, "csiborg"))
        parts = parts["particles"]
        field = csiborgtools.field.field2rsp(*field, parts=parts, box=box,
                                             verbose=parser_args.verbose)
    if to_save:
        fout = paths.field(parser_args.kind, parser_args.MAS, parser_args.grid,
                           nsim, parser_args.in_rsp)
        print(f"{datetime.now()}: saving output to `{fout}`.")
        numpy.save(fout, field)
    return field


###############################################################################
#                          Radial velocity field                              #
###############################################################################


def radvel_field(nsim, parser_args, to_save=True):
    """
    Calculate the radial velocity field in the CSiBORG simulation.

    Parameters
    ----------
    nsim : int
        Simulation index.
    parser_args : argparse.Namespace
        Parsed arguments.
    to_save : bool, optional
        Whether to save the output to disk.

    Returns
    -------
    radvel : 3-dimensional array
    """
    if parser_args.in_rsp:
        raise NotImplementedError("Radial vel. field in RSP not implemented.")

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, "csiborg"))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)

    vel = numpy.load(paths.field("velocity", parser_args.MAS, parser_args.grid,
                                 nsim, parser_args.in_rsp))
    gen = csiborgtools.field.VelocityField(box, parser_args.MAS)
    field = gen.radial_velocity(vel)
    if to_save:
        fout = paths.field("radvel", parser_args.MAS, parser_args.grid,
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

    Parameters
    ----------
    nsim : int
        Simulation index.
    parser_args : argparse.Namespace
        Parsed arguments.
    to_save : bool, optional
        Whether to save the output to disk.

    Returns
    -------
    env : 3-dimensional array
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, "csiborg"))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)
    density_gen = csiborgtools.field.DensityField(box, parser_args.MAS)
    gen = csiborgtools.field.TidalTensorField(box, parser_args.MAS)

    # Load the real space overdensity field
    if parser_args.verbose:
        print(f"{datetime.now()}: loading density field.")
    rho = numpy.load(paths.field("density", parser_args.MAS, parser_args.grid,
                                 nsim, in_rsp=False))

    rho = density_gen.overdensity_field(rho)
    # Calculate the real space tidal tensor field, delete overdensity.
    if parser_args.verbose:
        print(f"{datetime.now()}: calculating tidal tensor field.")
    tensor_field = gen(rho)
    del rho
    collect()

    # Optionally drag the field to RSP.
    if parser_args.in_rsp:
        parts = csiborgtools.read.read_h5(paths.particles(nsim, "csiborg"))
        parts = parts["particles"]
        fields = (tensor_field.T00, tensor_field.T11, tensor_field.T22,
                  tensor_field.T01, tensor_field.T02, tensor_field.T12)

        T00, T11, T22, T01, T02, T12 = csiborgtools.field.field2rsp(
            *fields, parts=parts, box=box, verbose=parser_args.verbose)
        tensor_field.T00[...] = T00
        tensor_field.T11[...] = T11
        tensor_field.T22[...] = T22
        tensor_field.T01[...] = T01
        tensor_field.T02[...] = T02
        tensor_field.T12[...] = T12
        del T00, T11, T22, T01, T02, T12
        collect()

    # Calculate the eigenvalues of the tidal tensor field, delete tensor field.
    if parser_args.verbose:
        print(f"{datetime.now()}: calculating eigenvalues.")
    eigvals = gen.tensor_field_eigvals(tensor_field)
    del tensor_field
    collect()

    # Classify the environment based on the eigenvalues.
    if parser_args.verbose:
        print(f"{datetime.now()}: classifying environment.")
    env = gen.eigvals_to_environment(eigvals)
    del eigvals
    collect()

    if to_save:
        fout = paths.field("environment", parser_args.MAS, parser_args.grid,
                           nsim, parser_args.in_rsp)
        print(f"{datetime.now()}: saving output to `{fout}`.")
        numpy.save(fout, env)
    return env


###############################################################################
#                          Command line interface                             #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. `-1` for all simulations.")
    parser.add_argument("--kind", type=str,
                        choices=["density", "rspdensity", "velocity", "radvel",
                                 "potential", "environment"],
                        help="What derived field to calculate?")
    parser.add_argument("--MAS", type=str,
                        choices=["NGP", "CIC", "TSC", "PCS"])
    parser.add_argument("--grid", type=int, help="Grid resolution.")
    parser.add_argument("--in_rsp", type=lambda x: bool(strtobool(x)),
                        help="Calculate in RSP?")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        help="Verbosity flag for reading in particles.")
    parser.add_argument("--simname", type=str, default="csiborg",
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
