# Copyright (C) 2024 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""
MPI script to interpolate the density and velocity fields along the line of
sight.
"""
from argparse import ArgumentParser
from datetime import datetime
from gc import collect
from os import makedirs, remove, rmdir
from os.path import exists, join
from warnings import warn

import csiborgtools
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from h5py import File
from mpi4py import MPI
from taskmaster import work_delegation

from utils import get_nsims

###############################################################################
#                             I/O functions                                   #
###############################################################################


def get_los(catalogue_name, simname, comm):
    """
    Get the line of sight RA/dec coordinates for the given catalogue.

    Parameters
    ----------
    catalogue_name : str
        Catalogue name.
    simname : str
        Simulation name.
    comm : mpi4py.MPI.Comm
        MPI communicator.

    Returns
    -------
    pos : 2-dimensional array
        RA/dec coordinates of the line of sight.
    """
    if comm.Get_rank() == 0:
        folder = "/mnt/extraspace/rstiskalek/catalogs"

        if catalogue_name == "LOSS" or catalogue_name == "Foundation":
            fpath = join(folder, "PV_compilation_Supranta2019.hdf5")
            with File(fpath, 'r') as f:
                grp = f[catalogue_name]
                RA = grp["RA"][:]
                dec = grp["DEC"][:]
        elif catalogue_name == "A2":
            fpath = join(folder, "A2.h5")
            with File(fpath, 'r') as f:
                RA = f["RA"][:]
                dec = f["DEC"][:]
        elif "csiborg1" in catalogue_name:
            nsim = int(catalogue_name.split("_")[-1])
            cat = csiborgtools.read.CSiBORG1Catalogue(
                nsim, bounds={"totmass": (1e13, None)})

            seed = 42
            gen = np.random.default_rng(seed)
            mask = gen.choice(len(cat), size=100, replace=False)

            sph_pos = cat["spherical_pos"]
            RA = sph_pos[mask, 1]
            dec = sph_pos[mask, 2]
        else:
            raise ValueError(f"Unknown field name: `{catalogue_name}`.")

        # The Carrick+2015 is in galactic coordinates, so we need to convert
        # the RA/dec to galactic coordinates.
        if simname == "Carrick2015":
            c = SkyCoord(ra=RA*u.degree, dec=dec*u.degree, frame='icrs')
            pos = np.vstack((c.galactic.l, c.galactic.b)).T
        else:
            pos = np.vstack((RA, dec)).T

    else:
        pos = None

    return comm.bcast(pos, root=0)


def get_field(simname, nsim, kind, MAS, grid):
    """
    Get the field from the simulation.

    Parameters
    ----------
    simname : str
        Simulation name.
    nsim : int
        IC realisation index.
    kind : str
        Field kind. Either `density` or `velocity`.
    MAS : str
        Mass assignment scheme.
    grid : int
        Grid resolution.

    Returns
    -------
    field : n-dimensional array
    """
    # Open the field reader.
    if simname == "csiborg1":
        field_reader = csiborgtools.read.CSiBORG1Field(nsim)
    elif simname == "Carrick2015":
        folder = "/mnt/extraspace/rstiskalek/catalogs"
        warn(f"Using local paths from `{folder}`.", RuntimeWarning)
        if kind == "density":
            fpath = join(folder, "twompp_density_carrick2015.npy")
            return np.load(fpath).astype(np.float32)
        elif kind == "velocity":
            fpath = join(folder, "twompp_velocity_carrick2015.npy")
            return np.load(fpath).astype(np.float32)
        else:
            raise ValueError(f"Unknown field kind: `{kind}`.")
    else:
        raise ValueError(f"Unknown simulation name: `{simname}`.")

    # Read in the field.
    if kind == "density":
        field = field_reader.density_field(MAS, grid)
    elif kind == "velocity":
        field = field_reader.velocity_field(MAS, grid)
    else:
        raise ValueError(f"Unknown field kind: `{kind}`.")

    return field


def combine_from_simulations(catalogue_name, simname, nsims, outfolder,
                             dumpfolder):
    """
    Combine the results from individual simulations into a single file.

    Parameters
    ----------
    catalogue_name : str
        Catalogue name.
    simname : str
        Simulation name.
    nsims : list
        List of IC realisations.
    outfolder : str
        Output folder.
    dumpfolder : str
        Dumping folder where the temporary files are stored.

    Returns
    -------
    None
    """
    fname_out = join(outfolder, f"los_{catalogue_name}_{simname}.hdf5")
    print(f"Combining results from invidivual simulations to `{fname_out}`.")

    if exists(fname_out):
        remove(fname_out)

    for nsim in nsims:
        fname = join(dumpfolder, f"los_{simname}_{nsim}.hdf5")

        with File(fname, 'r') as f, File(fname_out, 'a') as f_out:
            f_out.create_dataset(f"rdist_{nsim}", data=f["rdist"][:])
            f_out.create_dataset(f"density_{nsim}", data=f["density"][:])
            f_out.create_dataset(f"velocity_{nsim}", data=f["velocity"][:])

        # Remove the temporary file.
        remove(fname)

    # Remove the dumping folder.
    rmdir(dumpfolder)

    print("Finished combining results.")


###############################################################################
#                       Main interpolating function                           #
###############################################################################


def interpolate_field(pos, simname, nsim, MAS, grid, dump_folder, rmax,
                      dr, smooth_scales):
    """
    Interpolate the density and velocity fields along the line of sight.

    Parameters
    ----------
    pos : 2-dimensional array
        RA/dec coordinates of the line of sight.
    simname : str
        Simulation name.
    nsim : int
        IC realisation index.
    MAS : str
        Mass assignment scheme.
    grid : int
        Grid resolution.
    dump_folder : str
        Folder where the temporary files are stored.
    rmax : float
        Maximum distance along the line of sight.
    dr : float
        Distance spacing along the line of sight.
    smooth_scales : list
        Smoothing scales.

    Returns
    -------
    None
    """
    boxsize = csiborgtools.simname2boxsize(simname)
    fname_out = join(dump_folder, f"los_{simname}_{nsim}.hdf5")

    # First do the density field.
    density = get_field(simname, nsim, "density", MAS, grid)

    rdist, finterp = csiborgtools.field.evaluate_los(
        density, sky_pos=pos, boxsize=boxsize, rmax=rmax, dr=dr,
        smooth_scales=smooth_scales, verbose=False)

    print(f"Writing temporary file `{fname_out}`.")
    with File(fname_out, 'w') as f:
        f.create_dataset("rdist", data=rdist)
        f.create_dataset("density", data=finterp)

    del density, rdist, finterp
    collect()

    velocity = get_field(simname, nsim, "velocity", MAS, grid)
    rdist, finterp = csiborgtools.field.evaluate_los(
        velocity[0], velocity[1], velocity[2],
        sky_pos=pos, boxsize=boxsize, rmax=rmax, dr=dr,
        smooth_scales=smooth_scales, verbose=False)

    with File(fname_out, 'a') as f:
        f.create_dataset("velocity", data=finterp)


###############################################################################
#                           Command line interface                            #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--catalogue", type=str, help="Catalogue name.")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. `-1` for all simulations.")
    parser.add_argument("--simname", type=str, help="Simulation name.")
    parser.add_argument("--MAS", type=str,
                        choices=["NGP", "CIC", "TSC", "PCS", "SPH"],
                        help="Mass assignment scheme.")
    parser.add_argument("--grid", type=int, help="Grid resolution.")
    args = parser.parse_args()

    rmax = 200
    dr = 0.5
    smooth_scales = [0, 2, 4, 6]

    comm = MPI.COMM_WORLD
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(args, paths)

    out_folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/field_los"
    # Create the dumping folder.
    if comm.Get_rank() == 0:
        dump_folder = join(out_folder,
                           f"temp_{str(datetime.now())}".replace(" ", "_"))
        print(f"Creating folder `{dump_folder}`.")
        makedirs(dump_folder)
    else:
        dump_folder = None
    dump_folder = comm.bcast(dump_folder, root=0)

    # Get the line of sight sky coordinates.
    pos = get_los(args.catalogue, args.simname, comm)

    def main(nsim):
        interpolate_field(pos, args.simname, nsim, args.MAS, args.grid,
                          dump_folder, rmax, dr, smooth_scales)

    work_delegation(main, nsims, comm, master_verbose=True)
    comm.Barrier()

    if comm.Get_rank() == 0:
        combine_from_simulations(args.catalogue, args.simname, nsims,
                                 out_folder, dump_folder)
        print("All finished!")
