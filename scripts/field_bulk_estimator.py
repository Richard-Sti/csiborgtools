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
A script to calculate the bulk flow in Quijote to compare the volume average
definition to various estimators that rely on radial velocities.
"""
from argparse import ArgumentParser
from datetime import datetime
from os import remove
from os.path import join

import csiborgtools
import numpy as np
from mpi4py import MPI
from taskmaster import work_delegation  # noqa
from tqdm import trange

###############################################################################
#                Read in information about the simulation                     #
###############################################################################


def t():
    return datetime.now()


def get_reader(simname, paths, nsim):
    """Get the appropriate snaspshot reader for the simulation."""
    # We only want Quijote because it has all particles of the same mass.
    if simname == "quijote":
        # We want the z = 0 snapshots
        reader = csiborgtools.read.QuijoteSnapshot(nsim, 4, paths)
    else:
        raise ValueError(f"Unknown simname: `{simname}`.")

    return reader


def get_particles(reader, verbose=True):
    """
    Get the distance of particles from the center of the box and their masses.
    """
    if verbose:
        print(f"{t()},: reading coordinates and calculating radial distance.")
    pos = reader.coordinates().astype(np.float64)
    vel = reader.velocities().astype(np.float64)
    return pos, vel


###############################################################################
#                       Main & command line interface                         #
###############################################################################


def main(simname, nsim, folder, Rmax):
    observers = csiborgtools.read.fiducial_observers(boxsize, Rmax)
    distances = np.linspace(0, Rmax, 101)[1:]

    reader = get_reader(simname, paths, nsim)
    pos, vel = get_particles(reader, verbose=False)
    mass = np.ones(len(pos))  # Quijote has equal masses

    bf_volume = np.full((len(observers), len(distances), 3), np.nan)
    bf_peery = np.full_like(bf_volume, np.nan)
    bf_const = np.full_like(bf_volume, np.nan)

    for i in trange(len(observers), desc=f"Observers {nsim}`"):
        # Subtract the observer position.
        pos_current = pos - observers[i]
        # Get the distance of each particle from the observer and sort it.
        rdist = np.linalg.norm(pos_current, axis=1)
        indxs = np.argsort(rdist)

        pos_current = pos_current[indxs]
        vel_current = vel[indxs]
        rdist = rdist[indxs]

        # Volume average
        bf_volume[i, ...] = csiborgtools.field.particles_enclosed_momentum(
            rdist, mass, vel_current, distances)
        bf_volume[i, ...] /= csiborgtools.field.particles_enclosed_mass(
            rdist, mass, distances)[:, np.newaxis]

        # Peery 2018 1 / r^2 weighted
        bf_peery[i, ...] = csiborgtools.field.bulkflow_peery2018(
            rdist, mass, pos_current, vel_current, distances, "1/r^2",
            verbose=False)

        # Constant weight
        bf_const[i, ...] = csiborgtools.field.bulkflow_peery2018(
            rdist, mass, pos_current, vel_current, distances, "constant",
            verbose=False)

    # Finally save the output
    fname = join(folder, f"bf_estimators_{simname}_{nsim}.npz")
    print(f"Saving to `{fname}`.")
    np.savez(fname, bf_volume=bf_volume, bf_peery=bf_peery, bf_const=bf_const,
             distances=distances)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, help="Simulation name.",
                        choices=["quijote"])  # noqa
    args = parser.parse_args()
    Rmax = 150
    folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/field_shells"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    boxsize = csiborgtools.simname2boxsize(args.simname)

    def main_wrapper(nsim):
        main(args.simname, nsim, folder, Rmax)

    nsims = list(paths.get_ics(args.simname))
    if rank == 0:
        print(f"Running with {len(nsims)} Quijote simulations.")

    work_delegation(main_wrapper, nsims, comm, master_verbose=True)

    comm.Barrier()

    # Collect the results
    if rank == 0:
        for i, nsim in enumerate(nsims):
            fname = join(folder, f"bf_estimators_{args.simname}_{nsim}.npz")
            data = np.load(fname)

            if i == 0:
                bf_volume = np.empty((len(nsims), *data["bf_volume"].shape))
                bf_peery = np.empty_like(bf_volume)
                bf_const = np.empty_like(bf_volume)

                distances = data["distances"]

            bf_volume[i, ...] = data["bf_volume"]
            bf_peery[i, ...] = data["bf_peery"]
            bf_const[i, ...] = data["bf_const"]

            # Remove file from this rank
            remove(fname)

        # Save the results
        fname = join(folder, f"bf_estimators_{args.simname}.npz")
        print(f"Saving final results to `{fname}`.")
        np.savez(fname, bf_volume=bf_volume, bf_peery=bf_peery,
                 bf_const=bf_const, distances=distances)
