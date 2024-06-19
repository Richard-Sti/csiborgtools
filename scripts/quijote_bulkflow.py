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
A script to calculate the bulk flow in Quijote simulations from either
particles or FoF haloes and to also save the resulting smaller halo catalogues.
"""
from datetime import datetime
from os.path import join

import csiborgtools
import numpy as np
from mpi4py import MPI
from taskmaster import work_delegation  # noqa
from warnings import catch_warnings, simplefilter
from h5py import File
from sklearn.neighbors import NearestNeighbors


###############################################################################
#                Read in information about the simulation                     #
###############################################################################


def t():
    return datetime.now()


def get_data(nsim, verbose=True):
    if verbose:
        print(f"{t()}: reading particles of simulation `{nsim}`.")
    reader = csiborgtools.read.QuijoteSnapshot(nsim, 4, paths)
    part_pos = reader.coordinates().astype(np.float64)
    part_vel = reader.velocities().astype(np.float64)

    if verbose:
        print(f"{t()}: reading haloes of simulation `{nsim}`.")
    reader = csiborgtools.read.QuijoteCatalogue(nsim)
    halo_pos = reader.coordinates
    halo_vel = reader.velocities
    halo_mass = reader.totmass

    return part_pos, part_vel, halo_pos, halo_vel, halo_mass


def volume_bulk_flow(rdist, mass, vel, distances):
    out = csiborgtools.field.particles_enclosed_momentum(
        rdist, mass, vel, distances)
    with catch_warnings():
        simplefilter("ignore", category=RuntimeWarning)
        out /= csiborgtools.field.particles_enclosed_mass(
            rdist, mass, distances)[:, np.newaxis]

    return out


###############################################################################
#                       Main & command line interface                         #
###############################################################################


def main(nsim, folder, fname_basis, Rmax, subtract_observer_velocity,
         verbose=True):
    boxsize = csiborgtools.simname2boxsize("quijote")
    observers = csiborgtools.read.fiducial_observers(boxsize, Rmax)
    distances = np.linspace(0, Rmax, 101)[1:]
    part_pos, part_vel, halo_pos, halo_vel, halo_mass = get_data(nsim, verbose)

    if verbose:
        print(f"{t()}: Fitting the particle and halo trees of simulation `{nsim}`.")  # noqa
    part_tree = NearestNeighbors().fit(part_pos)
    halo_tree = NearestNeighbors().fit(halo_pos)

    samples = {}
    bf_volume_part = np.full((len(observers), len(distances), 3), np.nan)
    bf_volume_halo = np.full_like(bf_volume_part, np.nan)
    bf_volume_halo_uniform = np.full_like(bf_volume_part, np.nan)
    bf_vrad_weighted_part = np.full_like(bf_volume_part, np.nan)
    bf_vrad_weighted_halo_uniform = np.full_like(bf_volume_part, np.nan)
    bf_vrad_weighted_halo = np.full_like(bf_volume_part, np.nan)

    for i in range(len(observers)):
        print(f"{t()}: Calculating bulk flow for observer {i + 1} of simulation {nsim}.")  # noqa

        # Select particles within Rmax of the observer
        rdist_part, indxs = part_tree.radius_neighbors(
            np.asarray(observers[i]).reshape(1, -1), Rmax,
            return_distance=True, sort_results=True)
        rdist_part, indxs = rdist_part[0], indxs[0]

        part_pos_current = part_pos[indxs] - observers[i]
        part_vel_current = part_vel[indxs]
        # Quijote particle masses are all equal
        part_mass = np.ones_like(rdist_part)

        # Select haloes within Rmax of the observer
        rdist_halo, indxs = halo_tree.radius_neighbors(
            np.asarray(observers[i]).reshape(1, -1), Rmax,
            return_distance=True, sort_results=True)
        rdist_halo, indxs = rdist_halo[0], indxs[0]

        halo_pos_current = halo_pos[indxs] - observers[i]
        halo_vel_current = halo_vel[indxs]
        halo_mass_current = halo_mass[indxs]

        # Subtract the observer velocity
        if subtract_observer_velocity:
            rscale = 0.5
            weights = np.exp(-0.5 * (rdist_part / rscale)**2)
            obs_vel_x = np.average(part_vel_current[:, 0], weights=weights)
            obs_vel_y = np.average(part_vel_current[:, 1], weights=weights)
            obs_vel_z = np.average(part_vel_current[:, 2], weights=weights)

            part_vel_current[:, 0] -= obs_vel_x
            part_vel_current[:, 1] -= obs_vel_y
            part_vel_current[:, 2] -= obs_vel_z

            halo_vel_current[:, 0] -= obs_vel_x
            halo_vel_current[:, 1] -= obs_vel_y
            halo_vel_current[:, 2] -= obs_vel_z

        # Calculate the volume average bulk flows
        bf_volume_part[i, ...] = volume_bulk_flow(
            rdist_part, part_mass, part_vel_current, distances)
        bf_volume_halo[i, ...] = volume_bulk_flow(
            rdist_halo, halo_mass_current, halo_vel_current, distances)
        bf_volume_halo_uniform[i, ...] = volume_bulk_flow(
            rdist_halo, np.ones_like(halo_mass_current), halo_vel_current,
            distances)
        bf_vrad_weighted_part[i, ...] = csiborgtools.field.bulkflow_peery2018(
            rdist_part, part_mass, part_pos_current, part_vel_current,
            distances, weights="1/r^2", verbose=False)

        # Calculate the bulk flow from projected velocities w. 1/r^2 weights
        bf_vrad_weighted_halo_uniform[i, ...] = csiborgtools.field.bulkflow_peery2018(  # noqa
            rdist_halo, np.ones_like(halo_mass_current), halo_pos_current,
            halo_vel_current, distances, weights="1/r^2", verbose=False)
        bf_vrad_weighted_halo[i, ...] = csiborgtools.field.bulkflow_peery2018(
            rdist_halo, halo_mass_current, halo_pos_current,
            halo_vel_current, distances, weights="1/r^2", verbose=False)

        # Store the haloes around this observer
        samples[i] = {
            "halo_pos": halo_pos_current,
            "halo_vel": halo_vel_current,
            "halo_mass": halo_mass_current}

    # Finally save the output
    fname = join(folder, f"{fname_basis}_{nsim}.hdf5")
    if verbose:
        print(f"Saving to `{fname}`.")
    with File(fname, 'w') as f:
        f["distances"] = distances
        f["bf_volume_part"] = bf_volume_part
        f["bf_volume_halo"] = bf_volume_halo
        f["bf_vrad_weighted_part"] = bf_vrad_weighted_part
        f["bf_volume_halo_uniform"] = bf_volume_halo_uniform
        f["bf_vrad_weighted_halo_uniform"] = bf_vrad_weighted_halo_uniform
        f["bf_vrad_weighted_halo"] = bf_vrad_weighted_halo

        for i in range(len(observers)):
            g = f.create_group(f"obs_{str(i)}")
            g["halo_pos"] = samples[i]["halo_pos"]
            g["halo_vel"] = samples[i]["halo_vel"]
            g["halo_mass"] = samples[i]["halo_mass"]


if __name__ == "__main__":
    Rmax = 150
    subtract_observer_velocity = False
    folder = "/mnt/extraspace/rstiskalek/quijote/BulkFlow_fiducial"
    fname_basis = "BF_nsim"
    if subtract_observer_velocity:
        fname_basis += "_subtracted_observer_velocity"

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = list(paths.get_ics("quijote"))

    def main_wrapper(nsim):
        main(nsim, folder, fname_basis, Rmax, subtract_observer_velocity,
             verbose=rank == 0)

    if rank == 0:
        print(f"Running with {len(nsims)} Quijote simulations.")

    comm.Barrier()
    work_delegation(main_wrapper, nsims, comm, master_verbose=True)
