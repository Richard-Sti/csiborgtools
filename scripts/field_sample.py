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
Sample a CSiBORG field at galaxy positions and save the result to disk.
"""
from argparse import ArgumentParser
from distutils.util import strtobool
from os.path import join

import numpy
from astropy.cosmology import FlatLambdaCDM
from h5py import File
from mpi4py import MPI
from taskmaster import work_delegation
from tqdm import tqdm

import csiborgtools
from utils import get_nsims

MPC2BOX = 1 / 677.7


def steps(cls, survey_name):
    """Make a list of selection criteria to apply to a survey."""
    if survey_name == "SDSS":
        return [
            # (lambda x: cls[x], ("IN_DR7_LSS",)),
            # (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
            (lambda x: cls[x] < 155.5, ("DIST", ))
            ]
    else:
        raise NotImplementedError(f"Survey `{survey_name}` not implemented.")


def open_galaxy_positions(survey_name, comm):
    """
    Load the survey galaxy positions and indices, broadcasting them to all
    ranks.
    """
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        if survey_name == "SDSS":
            survey = csiborgtools.SDSS()()
            pos = numpy.vstack([survey["DIST_UNCORRECTED"],
                                survey["RA"],
                                survey["DEC"]],
                               ).T
            pos = pos.astype(numpy.float32)
            indxs = survey["INDEX"]
        if survey_name == "SDSSxALFALFA":
            survey = csiborgtools.SDSSxALFALFA()()
            pos = numpy.vstack([survey["DIST_UNCORRECTED"],
                                survey["RA_1"],
                                survey["DEC_1"]],
                               ).T
            pos = pos.astype(numpy.float32)
            indxs = survey["INDEX"]
        elif survey_name == "GW170817":
            samples = File("/mnt/extraspace/rstiskalek/GWLSS/H1L1V1-EXTRACT_POSTERIOR_GW170817-1187008600-400.hdf", 'r')["samples"]  # noqa
            cosmo = FlatLambdaCDM(H0=100, Om0=0.3175)
            pos = numpy.vstack([
                cosmo.comoving_distance(samples["redshift"][:]).value,
                samples["ra"][:] * 180 / numpy.pi,
                samples["dec"][:] * 180 / numpy.pi],
                               ).T
            indxs = numpy.arange(pos.shape[0])
        else:
            raise NotImplementedError(f"Survey `{survey_name}` not "
                                      "implemented.")
    else:
        pos = None
        indxs = None

    comm.Barrier()

    if size > 1:
        pos = comm.bcast(pos, root=0)
        indxs = comm.bcast(indxs, root=0)

    return pos, indxs


def evaluate_field(field, pos, nrand, smooth_scales=None, seed=42,
                   verbose=True):
    """
    Evaluate the field at the given sky positions. Additionally, evaluate the
    field at `nrand` random positions.
    """
    if smooth_scales is None:
        smooth_scales = [0.]

    nsample = pos.shape[0]
    nsmooth = len(smooth_scales)

    val = numpy.full((nsample, nsmooth), numpy.nan, dtype=field.dtype)
    if nrand > 0:
        rand_val = numpy.full((nsample, nsmooth, nrand), numpy.nan,
                              dtype=field.dtype)
    else:
        rand_val = None

    for i, scale in enumerate(tqdm(smooth_scales, desc="Smoothing",
                                   disable=not verbose)):
        if scale > 0:
            field_smoothed = csiborgtools.field.smoothen_field(
                field, scale * MPC2BOX, boxsize=1, make_copy=True)
        else:
            field_smoothed = field

        val[:, i] = csiborgtools.field.evaluate_sky(
            field_smoothed, pos=pos, mpc2box=MPC2BOX)

        if nrand == 0:
            continue

        for j in range(nrand):
            gen = numpy.random.default_rng(seed + j)
            pos_rand = numpy.vstack([
                gen.permutation(pos[:, 0]),
                gen.uniform(0, 360, nsample),
                90 - numpy.rad2deg(numpy.arccos(gen.uniform(-1, 1, nsample))),
                ]).T

            rand_val[:, i, j] = csiborgtools.field.evaluate_sky(
                field_smoothed, pos=pos_rand, mpc2box=MPC2BOX)

    return val, rand_val, smooth_scales


def main(nsim, parser_args, pos, indxs, paths, verbose):
    """Load the field, interpolate it and save it to disk."""
    fpath_field = paths.field(parser_args.kind, parser_args.MAS,
                              parser_args.grid, nsim, parser_args.in_rsp)
    field = numpy.load(fpath_field)

    val, rand_val, smooth_scales = evaluate_field(
        field, pos, nrand=parser_args.nrand,
        smooth_scales=parser_args.smooth_scales, verbose=verbose)

    if parser_args.survey == "GW170817":
        kind = parser_args.kind
        kind = kind + "_rsp" if parser_args.in_rsp else kind

        fout = join(
            "/mnt/extraspace/rstiskalek/GWLSS/",
            f"{kind}_{parser_args.MAS}_{parser_args.grid}_{nsim}_H1L1V1-EXTRACT_POSTERIOR_GW170817-1187008600-400.npz")  # noqa
    else:
        fout = paths.field_interpolated(parser_args.survey, parser_args.kind,
                                        parser_args.MAS, parser_args.grid,
                                        nsim, parser_args.in_rsp)
    if verbose:
        print(f"Saving to ... `{fout}`.")
    numpy.savez(fout, val=val, rand_val=rand_val, indxs=indxs,
                smooth_scales=smooth_scales)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. If `-1` processes all.")
    parser.add_argument("--survey", type=str, required=True,
                        choices=["SDSS", "SDSSxALFALFA", "GW170817"],
                        help="Galaxy survey")
    parser.add_argument("--smooth_scales", type=float, nargs="+", default=None,
                        help="Smoothing scales in Mpc / h.")
    parser.add_argument("--kind", type=str,
                        choices=["density", "rspdensity", "velocity", "radvel",
                                 "potential"],
                        help="What field to interpolate.")
    parser.add_argument("--MAS", type=str,
                        choices=["NGP", "CIC", "TSC", "PCS"],
                        help="Mass assignment scheme.")
    parser.add_argument("--grid", type=int, help="Grid resolution.")
    parser.add_argument("--in_rsp", type=lambda x: bool(strtobool(x)),
                        help="Field in RSP?")
    parser.add_argument("--nrand", type=int, required=True,
                        help="Number of rand. positions to evaluate the field")
    parser.add_argument("--simname", type=str, default="csiborg",
                        choices=["csiborg"], help="Simulation name")
    args = parser.parse_args()

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(args, paths)

    pos, indxs = open_galaxy_positions(args.survey, MPI.COMM_WORLD)

    if MPI.COMM_WORLD.Get_rank() == 0 and args.survey != "GW170817":
        fout = f"/mnt/extraspace/rstiskalek/CSiBORG/ascii_positions/{args.survey}_positions.npz"  # noqa
        pos = csiborgtools.utils.radec_to_cartesian(pos) + 677.7 / 2
        print(f"Saving to ... `{fout}`.")
        numpy.savez(fout, pos=pos, indxs=indxs)

    def _main(nsim):
        main(nsim, args, pos, indxs, paths,
             verbose=MPI.COMM_WORLD.Get_size() == 1)

    work_delegation(_main, nsims, MPI.COMM_WORLD)
