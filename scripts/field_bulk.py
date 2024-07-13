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
A script to calculate the enclosed mass or bulk flow at different distances
from the center of the box directly from the particles. Note that the velocity
of an observer is not being subtracted from the bulk flow.

The script is not parallelized in any way but it should not take very long, the
main bottleneck is reading the data from disk.
"""
from argparse import ArgumentParser
from datetime import datetime
from gc import collect
from os.path import join

import numpy as np
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, SkyCoord
from tqdm import tqdm

import csiborgtools
from csiborgtools import fprint
from csiborgtools.field import (field_enclosed_mass, particles_enclosed_mass,
                                particles_enclosed_momentum)

###############################################################################
#                Read in information about the simulation                     #
###############################################################################


def t():
    return datetime.now()


def get_reader(simname, paths, nsim):
    """Get the appropriate snapshot reader for the simulation."""
    if simname == "csiborg1":
        nsnap = max(paths.get_snapshots(nsim, simname))
        reader = csiborgtools.read.CSiBORG1Snapshot(nsim, nsnap, paths,
                                                    flip_xz=True)
    elif "csiborg2" in simname:
        kind = simname.split("_")[-1]
        reader = csiborgtools.read.CSiBORG2Snapshot(nsim, 99, kind, paths,
                                                    flip_xz=True)
    else:
        raise ValueError(f"Unknown simname: `{simname}`.")

    return reader


def get_particles(reader, boxsize, get_velocity=True, verbose=True):
    """Get the snapshot particles."""
    fprint("reading coordinates and calculating radial distance.", verbose)
    pos = reader.coordinates()
    dtype = pos.dtype
    pos -= boxsize / 2
    dist = np.linalg.norm(pos, axis=1).astype(dtype)
    collect()

    if get_velocity:
        fprint("reading velocities.", verbose)
        vel = reader.velocities().astype(dtype)
        vrad = np.sum(pos, vel, axis=1) / dist

    del pos
    collect()

    fprint("reading masses.")
    mass = reader.masses()

    fprint("sorting arrays.")
    indxs = np.argsort(dist)
    dist = dist[indxs]
    mass = mass[indxs]
    if get_velocity:
        vel = vel[indxs]

    del indxs
    collect()

    if get_velocity:
        return dist, mass, vel, vrad

    return dist, mass


###############################################################################
#                                 Main                                        #
###############################################################################


def main_borg(args, folder):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    boxsize = csiborgtools.simname2boxsize(args.simname)
    nsims = paths.get_ics(args.simname)
    distances = np.linspace(0, boxsize / 2, 101)

    cumulative_mass = np.zeros((len(nsims), len(distances)))
    cumulative_volume = np.zeros((len(nsims), len(distances)))
    for i, nsim in enumerate(tqdm(nsims, desc="Simulations")):
        if args.simname == "borg1":
            reader = csiborgtools.read.BORG1Field(nsim)
            field = reader.density_field()
        elif args.simname == "borg2" or args.simname == "borg2_all":
            reader = csiborgtools.read.BORG2Field(nsim)
            field = reader.density_field()
        else:
            raise ValueError(f"Unknown simname: `{args.simname}`.")

        cumulative_mass[i, :], cumulative_volume[i, :] = field_enclosed_mass(
            field, distances, boxsize)

    # Finally save the output
    fname = f"enclosed_mass_{args.simname}.npz"
    fname = join(folder, fname)
    np.savez(fname, enclosed_mass=cumulative_mass, distances=distances,
             enclosed_volume=cumulative_volume)


def main_csiborg(args, folder):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    boxsize = csiborgtools.simname2boxsize(args.simname)
    nsims = paths.get_ics(args.simname)
    distances = np.linspace(0, boxsize / 2, 501)[1:]

    # Initialize arrays to store the results
    cumulative_mass = np.zeros((len(nsims), len(distances)))
    mass135 = np.zeros(len(nsims))
    masstot = np.zeros(len(nsims))
    cumulative_vel_mono = np.zeros((len(nsims), len(distances)))
    cumulative_velocity = np.zeros((len(nsims), len(distances), 3))

    for i, nsim in enumerate(tqdm(nsims, desc="Simulations")):
        reader = get_reader(args.simname, paths, nsim)
        rdist, mass, vel, vrad = get_particles(reader, boxsize,  verbose=True)

        # Calculate masses
        cumulative_mass[i, :] = particles_enclosed_mass(rdist, mass, distances)
        mass135[i] = particles_enclosed_mass(rdist, mass, [135])[0]
        masstot[i] = np.sum(mass)

        # Calculate monopole momentum
        cumulative_vel_mono[i] = particles_enclosed_mass(
            rdist, vrad * mass, distances)

        # Calculate velocities
        cumulative_velocity[i, ...] = particles_enclosed_momentum(
            rdist, mass, vel, distances)

        # Normalize the momentum to get velocity out of it.
        for j in range(3):
            cumulative_velocity[i, :, j] /= cumulative_mass[i, :]
        cumulative_vel_mono[i, ...] /= cumulative_mass[i, ...]

    # Finally save the output
    fname = f"enclosed_mass_{args.simname}.npz"
    fname = join(folder, fname)
    np.savez(fname, enclosed_mass=cumulative_mass, mass135=mass135,
             masstot=masstot, distances=distances,
             cumulative_velocity=cumulative_velocity,
             cumulative_velocity_mono=cumulative_vel_mono)


def main_from_field(args, folder):
    """Bulk flows in 3D fields"""
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    boxsize = csiborgtools.simname2boxsize(args.simname)
    nsims = paths.get_ics(args.simname)
    distances = np.linspace(0, boxsize / 2, 101)[1:]

    cumulative_mass = np.zeros((len(nsims), len(distances)))
    cumulative_volume = np.zeros((len(nsims), len(distances)))
    cumulative_vel_mono = np.zeros((len(nsims), len(distances)))
    cumulative_vel_x = np.zeros((len(nsims), len(distances)))
    cumulative_vel_y = np.zeros_like(cumulative_vel_x)
    cumulative_vel_z = np.zeros_like(cumulative_vel_x)
    for i, nsim in enumerate(tqdm(nsims, desc="Simulations")):
        if args.simname == "csiborg2X":
            reader = csiborgtools.read.CSiBORG2XField(nsim, paths)
        elif args.simname == "Carrick2015":
            reader = csiborgtools.read.Carrick2015Field(paths)
        elif args.simname == "Lilow2024":
            reader = csiborgtools.read.Lilow2024Field(paths)
        else:
            raise ValueError(f"Unknown simname: `{args.simname}`.")

        density_field = reader.density_field()
        cumulative_mass[i, :], cumulative_volume[i, :] = field_enclosed_mass(
            density_field, distances, boxsize, verbose=False)
        del density_field
        collect()

        velocity_field = reader.velocity_field()
        radial_velocity_field = csiborgtools.field.radial_velocity(
            velocity_field, [0., 0., 0.])

        cumulative_vel_mono[i, :], __ = field_enclosed_mass(
            radial_velocity_field, distances, boxsize, verbose=False)
        del radial_velocity_field
        collect()

        cumulative_vel_x[i, :], __ = field_enclosed_mass(
            velocity_field[0], distances, boxsize, verbose=False)
        cumulative_vel_y[i, :], __ = field_enclosed_mass(
            velocity_field[1], distances, boxsize, verbose=False)
        cumulative_vel_z[i, :], __ = field_enclosed_mass(
            velocity_field[2], distances, boxsize, verbose=False)

        del velocity_field
        collect()

    if args.simname in ["Carrick2015", "Lilow2024"]:
        # Carrick+2015 and Lilow+2024 box is in galactic coordinates, so we
        # need to convert the bulk flow vector to RA/dec Cartesian
        # representation.
        galactic_cartesian = CartesianRepresentation(
            cumulative_vel_x, cumulative_vel_y, cumulative_vel_z,
            unit=u.km/u.s)
        galactic_coord = SkyCoord(galactic_cartesian, frame='galactic')
        icrs_cartesian = galactic_coord.icrs.cartesian

        cumulative_vel_x = icrs_cartesian.x.to(u.km/u.s).value
        cumulative_vel_y = icrs_cartesian.y.to(u.km/u.s).value
        cumulative_vel_z = icrs_cartesian.z.to(u.km/u.s).value

    cumulative_vel = np.stack(
        [cumulative_vel_x, cumulative_vel_y, cumulative_vel_z], axis=-1)
    cumulative_vel /= cumulative_volume[..., None]
    cumulative_vel_mono /= cumulative_volume

    # Finally save the output
    fname = f"enclosed_mass_{args.simname}.npz"
    fname = join(folder, fname)
    print(f"Saving to `{fname}`.")
    np.savez(fname, enclosed_mass=cumulative_mass, distances=distances,
             cumulative_velocity_mono=cumulative_vel_mono,
             cumulative_velocity=cumulative_vel,
             enclosed_volume=cumulative_volume)


###############################################################################
#                        Command line interface                               #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, help="Simulation name.",
                        choices=["csiborg1", "csiborg2_main", "csiborg2_varysmall", "csiborg2_random",  # noqa
                                 "borg1", "borg2", "borg2_all", "csiborg2X", "Carrick2015",             # noqa
                                 "Lilow2024"])                                                          # noqa
    args = parser.parse_args()

    folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/field_shells"
    if args.simname in ["csiborg2X", "Carrick2015", "Lilow2024"]:
        main_from_field(args, folder)
    elif "csiborg" in args.simname:
        main_csiborg(args, folder)
    elif "borg" in args.simname:
        main_borg(args, folder)
    else:
        raise ValueError(f"Unknown simname: `{args.simname}`.")
