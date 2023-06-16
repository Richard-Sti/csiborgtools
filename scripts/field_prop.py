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


def density_field(nsim, parser_args):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)
    parts = csiborgtools.read.read_h5(paths.particles(nsim))["particles"]

    gen = csiborgtools.field.DensityField(box, parser_args.MAS)
    field = gen(parts, parser_args.grid, in_rsp=parser_args.in_rsp,
                verbose=parser_args.verbose)

    fout = paths.field("density", parser_args.MAS, parser_args.grid,
                       nsim, parser_args.in_rsp)
    print(f"{datetime.now()}: saving output to `{fout}`.")
    numpy.save(fout, field)


###############################################################################
#                            Velocity field                                   #
###############################################################################


def velocity_field(nsim, parser_args):
    if parser_args.in_rsp:
        raise NotImplementedError("Velocity field in RSP is not implemented.")
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    mpart = 1.1641532e-10  # Particle mass in CSiBORG simulations.
    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)
    parts = csiborgtools.read.read_h5(paths.particles(nsim))["particles"]

    gen = csiborgtools.field.VelocityField(box, parser_args.MAS)
    field = gen(parts, parser_args.grid, mpart, verbose=parser_args.verbose)

    fout = paths.field("velocity", parser_args.MAS, parser_args.grid,
                       nsim, in_rsp=False)
    print(f"{datetime.now()}: saving output to `{fout}`.")
    numpy.save(fout, field)


###############################################################################
#                          Potential field                                    #
###############################################################################


def potential_field(nsim, parser_args):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim))
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
        parts = csiborgtools.read.read_h5(paths.particles(nsim))["particles"]
        field = csiborgtools.field.field2rsp(field, parts=parts, box=box,
                                             verbose=parser_args.verbose)
    fout = paths.field(parser_args.kind, parser_args.MAS, parser_args.grid,
                       nsim, parser_args.in_rsp)
    print(f"{datetime.now()}: saving output to `{fout}`.")
    numpy.save(fout, field)


###############################################################################
#                          Radial velocity field                              #
###############################################################################


def radvel_field(nsim, parser_args):
    if parser_args.in_rsp:
        raise NotImplementedError("Radial vel. field in RSP not implemented.")
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)

    vel = numpy.load(paths.field("velocity", parser_args.MAS, parser_args.grid,
                                 nsim, parser_args.in_rsp))
    gen = csiborgtools.field.VelocityField(box, parser_args.MAS)
    field = gen.radial_velocity(vel)

    fout = paths.field("radvel", parser_args.MAS, parser_args.grid,
                       nsim, parser_args.in_rsp)
    print(f"{datetime.now()}: saving output to `{fout}`.")
    numpy.save(fout, field)


###############################################################################
#                        Environment classification                           #
###############################################################################


def environment_field(nsim, parser_args):
    if parser_args.in_rsp:
        raise NotImplementedError("Env. field in RSP not implemented.")
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim))
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

    fout = paths.field("environment", parser_args.MAS, parser_args.grid,
                       nsim, parser_args.in_rsp)
    print(f"{datetime.now()}: saving output to `{fout}`.")
    numpy.save(fout, env)


###############################################################################
#                          Command line interface                             #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. `-1` for all simulations.")
    parser.add_argument("--kind", type=str,
                        choices=["density", "velocity", "radvel", "potential",
                                 "environment"],
                        help="What derived field to calculate?")
    parser.add_argument("--MAS", type=str,
                        choices=["NGP", "CIC", "TSC", "PCS"])
    parser.add_argument("--grid", type=int, help="Grid resolution.")
    parser.add_argument("--in_rsp", type=lambda x: bool(strtobool(x)),
                        help="Calculate in RSP?")
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        help="Verbosity flag for reading in particles.")
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
        elif parser_args.kind == "potential":
            potential_field(nsim, parser_args)
        elif parser_args.kind == "environment":
            environment_field(nsim, parser_args)
        else:
            raise RuntimeError(f"Field {parser_args.kind} is not implemented.")

    work_delegation(main, nsims, comm, master_verbose=True)
