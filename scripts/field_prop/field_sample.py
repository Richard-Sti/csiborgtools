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
Script to sample a CSiBORG field at galaxy positions and save the result.
Supports additional smoothing of the field as well.
"""
import sys
from argparse import ArgumentParser
from os.path import join

import csiborgtools
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from h5py import File
from mpi4py import MPI
from numba import jit
from scipy.integrate import simpson
from taskmaster import work_delegation                                          # noqa

sys.path.append("../")
from utils import get_nsims                                                     # noqa


@jit(nopython=True, fastmath=True, boundscheck=False)
def scatter_along_radial_direction(pos, scatter, boxsize):
    """
    Scatter galaxy positions along the radial direction. Enforces that the
    radial position is always on the same side of the box and that the galaxy
    is still inside the box.

    Parameters
    ----------
    pos : 2-dimensional array
        Galaxy positions in the form of (distance, RA, DEC).
    scatter : float
        Scatter to add to the radial positions of galaxies in same units as
        `distance` (Mpc / h).
    boxsize : float
        Box size in `Mpc / h`.
    """
    pos_new = np.copy(pos)

    for i in range(len(pos)):
        r0, ra, dec = pos[i]
        # Convert to radians
        ra *= np.pi / 180
        dec *= np.pi / 180

        # Convert to normalized Cartesian coordinates
        xnorm = np.cos(dec) * np.cos(ra)
        ynorm = np.cos(dec) * np.sin(ra)
        znorm = np.sin(dec)

        while True:
            rnew = np.random.normal(r0, scatter)
            if rnew < 0:
                continue

            xnew = rnew * xnorm + boxsize / 2
            ynew = rnew * ynorm + boxsize / 2
            znew = rnew * znorm + boxsize / 2

            if 0 <= xnew < boxsize and 0 <= ynew < boxsize and 0 <= znew < boxsize:  # noqa
                pos_new[i, 0] = rnew
                break

    return pos_new


def open_galaxy_positions(survey_name, comm, scatter=None):
    """
    Load the survey's galaxy positions , broadcasting them to all ranks.

    Parameters
    ----------
    survey_name : str
        Name of the survey.
    comm : mpi4py.MPI.Comm
        MPI communicator.
    scatter : float
        Scatter to add to the radial positions of galaxies, supportted only in
        TNG300-1.

    Returns
    -------
    pos : 2-dimensional array
        Galaxy positions in the form of (distance, RA, DEC).
    """
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        if survey_name == "SDSS":
            survey = csiborgtools.SDSS()()
            pos = np.vstack([survey["DIST"], survey["RA"], survey["DEC"]]).T
            pos = pos.astype(np.float32)
            # Assume a distance error of 300 km / s, so about 3 Mpc / h
            edist = np.ones(len(pos)) * 3  # Mpc / h
        elif survey_name == "SDSSxALFALFA":
            survey = csiborgtools.SDSSxALFALFA()()
            pos = np.vstack([
                survey["DIST"], survey["RA_1"], survey["DEC_1"]],).T
            pos = pos.astype(np.float32)
            # Assume a distance error of 300 km / s, so about 3 Mpc / h
            edist = np.ones(len(pos)) * 3  # Mpc / h
        elif survey_name == "GW170817":
            samples = File("/mnt/extraspace/rstiskalek/GWLSS/H1L1V1-EXTRACT_POSTERIOR_GW170817-1187008600-400.hdf", 'r')["samples"]  # noqa
            cosmo = FlatLambdaCDM(H0=100, Om0=0.3175)
            pos = np.vstack([
                cosmo.comoving_distance(samples["redshift"][:]).value,
                samples["ra"][:] * 180 / np.pi,
                samples["dec"][:] * 180 / np.pi],
                               ).T
            edist = None
        elif survey_name == "TNG300-1":
            with File("/mnt/extraspace/rstiskalek/TNG300-1/postprocessing/subhalo_catalogue_099.hdf5", 'r') as f:  # noqa
                pos = np.vstack([f["SubhaloPos"][:, 0],
                                 f["SubhaloPos"][:, 1],
                                 f["SubhaloPos"][:, 2]]).T
                boxsize = csiborgtools.simname2boxsize("TNG300-1")
                pos -= boxsize / 2
                pos = csiborgtools.cartesian_to_radec(pos)
                if scatter is not None:
                    if scatter < 0:
                        raise ValueError("Scatter must be positive.")
                    if scatter > 0:
                        pos = scatter_along_radial_direction(pos, scatter,
                                                             boxsize)
            edist = None

        else:
            raise NotImplementedError(f"Survey `{survey_name}` not "
                                      "implemented.")
    else:
        pos = None
        edist = None

    comm.Barrier()

    if size > 1:
        pos = comm.bcast(pos, root=0)
        edist = comm.bcast(edist, root=0)

    return pos, edist


def match_to_no_selection(val, parser_args):
    """
    Match the shape of the evaluated field to the shape of the survey without
    any masking. Missing values are filled with `np.nan`.

    Parameters
    ----------
    val : n-dimensional array
        Evaluated field.
    parser_args : argparse.Namespace
        Command line arguments.

    Returns
    -------
    n-dimensional array
    """
    if parser_args.survey == "SDSS":
        survey = csiborgtools.SDSS()()
    elif parser_args.survey == "SDSSxALFALFA":
        survey = csiborgtools.SDSSxALFALFA()()
    else:
        raise NotImplementedError(
            f"Survey `{parser_args.survey}` not implemented for matching to no selection.")  # noqa

    return csiborgtools.read.match_array_to_no_masking(val, survey)


def normal_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * np.sqrt(2 * np.pi))


def main(nsim, parser_args, pos, edist, boxsize, r_grid, verbose):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    # Get the appropriate field loader
    if parser_args.simname == "csiborg1":
        freader = csiborgtools.read.CSiBORG1Field(nsim)
    elif "csiborg2" in parser_args.simname:
        kind = parser_args.simname.split("_")[-1]
        freader = csiborgtools.read.CSiBORG2Field(nsim, kind)
    elif "manticore_2MPP_N128_DES_V1" == parser_args.simname:
        freader = csiborgtools.read.CSiBORG2XField(nsim, 1)
    elif parser_args.simname == "TNG300-1":
        freader = csiborgtools.read.TNG300_1Field()
    else:
        raise NotImplementedError(f"Simulation `{parser_args.simname}` is not supported.")  # noqa

    # Get the appropriate field
    if parser_args.kind == "density":
        field = freader.density_field(
            MAS=parser_args.MAS, grid=parser_args.grid)
    else:
        raise NotImplementedError(f"Field `{parser_args.kind}` is not supported.")  # noqa

    # Convert to Cartesian positions in box units.
    box_pos = csiborgtools.radec_to_cartesian(pos) / boxsize + 0.5
    box_smooth = [s / boxsize for s in parser_args.smooth_scales]

    val = csiborgtools.field.evaluate_cartesian_cic(
        field, pos=box_pos, smooth_scales=box_smooth, verbose=verbose)

    if edist is not None:
        rdist, density_interp = csiborgtools.field.evaluate_los(
            field, sky_pos=pos, boxsize=boxsize, rdist=r_grid,
            smooth_scales=parser_args.smooth_scales, verbose=verbose,
            interpolation_method="linear")

        if density_interp.ndim == 2:
            density_interp = density_interp[:, :, None]

        mean_dist = pos[:, 0]

        # Evaluate the normal PDF such that we have `(n_query, len(rdist))`
        pgauss = normal_pdf(rdist[None, :], mean_dist[:, None], edist[:, None])
        # Multiply by the Malmquist bias
        pgauss *= rdist[None, :]**2
        # Multiply by the inhomogeneous Malmquist, so that the shape becomes
        # `(n_query, len(rdist), len(smooth_scales))`
        pgauss = pgauss[:, :, None] * density_interp

        # Now calculate the normalisation, by integrating over the radial
        # distance, the shape becomes `(n_query, len(smooth_scales))`
        pgauss_norm = simpson(pgauss, x=rdist, axis=1)

        # Now multiply `pgauss` by the density, because that's what we want to
        # average out, and divide by the normalisation.
        pgauss = density_interp * pgauss / pgauss_norm[:, None, :]

        # Do the integral along the LOS to calculate the mean density
        val_with_edist = simpson(pgauss, x=rdist, axis=1)
    else:
        val_with_edist = np.nan

    if parser_args.survey == "GW170817":
        fout = join(
            "/mnt/extraspace/rstiskalek/GWLSS/",
            f"{parser_args.kind}_{parser_args.MAS}_{parser_args.grid}_{nsim}_H1L1V1-EXTRACT_POSTERIOR_GW170817-1187008600-400.npz")  # noqa
    else:
        if parser_args.simname == "TNG300-1":
            scatter = parser_args.scatter
        else:
            scatter = None

        fout = paths.field_interpolated(
            parser_args.survey, parser_args.simname, nsim, parser_args.kind,
            parser_args.MAS, parser_args.grid, scatter)

        # The survey above had some cuts, however for compatibility we want
        # the same shape as the `uncut` survey
        if parser_args.survey != "TNG300-1":
            val = match_to_no_selection(val, parser_args)

    if verbose:
        print(f"Saving to ... `{fout}`.")

    np.savez(fout, val=val, val_with_edist=val_with_edist,
             smooth_scales=parser_args.smooth_scales)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. If `-1` processes all.")
    parser.add_argument("--simname", type=str, default="csiborg1",
                        choices=["csiborg1", "csiborg2_main", "csiborg2_random", "csiborg2_varysmall", "manticore_2MPP_N128_DES_V1", "TNG300-1"],  # noqa
                        help="Simulation name")
    parser.add_argument("--survey", type=str, required=True,
                        choices=["SDSS", "SDSSxALFALFA", "GW170817", "TNG300-1"],  # noqa
                        help="Galaxy survey")
    parser.add_argument("--smooth_scales", type=float, nargs="+", default=None,
                        help="Smoothing scales in Mpc / h.")
    parser.add_argument("--kind", type=str,
                        choices=["density", "rspdensity", "velocity", "radvel",
                                 "potential"],
                        help="What field to interpolate.")
    parser.add_argument("--MAS", type=str,
                        choices=["NGP", "CIC", "TSC", "PCS", "SPH"],
                        help="Mass assignment scheme.")
    parser.add_argument("--grid", type=int, help="Grid resolution.")
    parser.add_argument("--scatter", type=float, default=None,
                        help="Scatter to add to the radial positions of galaxies, supportted only in TNG300-1.")  # noqa
    args = parser.parse_args()

    if args.simname == "TNG300-1" and args.survey != "TNG300-1":
        raise ValueError("TNG300-1 simulation is only supported for TNG300-1 survey.")  # noqa

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    if args.simname == "TNG300-1":
        nsims = [0]
    else:
        nsims = get_nsims(args, paths)

    boxsize = csiborgtools.simname2boxsize(args.simname)

    pos, edist = open_galaxy_positions(
        args.survey, MPI.COMM_WORLD, args.scatter)

    if edist is not None:
        r_grid = np.arange(0, 200, 0.1)
        print(f"Setting up the radial distance grid from {r_grid.min()} to "
              f"{r_grid.max()} in steps of {r_grid[1] - r_grid[0]} Mpc / h, "
              f"with a total of {len(r_grid)} steps.")
    else:
        r_grid = None

    print(f"Smoothing scales = {args.smooth_scales} Mpc / h")

    def _main(nsim):
        main(nsim, args, pos, edist, boxsize, r_grid,
             verbose=MPI.COMM_WORLD.Get_size() == 1)

    work_delegation(_main, nsims, MPI.COMM_WORLD)
