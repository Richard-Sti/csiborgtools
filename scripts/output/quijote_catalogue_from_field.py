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
"""Quick script to write halo catalogues as a HDF5 file."""
from os import remove
from os.path import exists, join

import numpy as np
from h5py import File

import csiborgtools
from csiborgtools import fprint

###############################################################################
#                    Evaluating fields at a fixed radius.                     #
###############################################################################


def load_fields(nsim, MAS, grid, paths):
    # reader = csiborgtools.read.QuijoteField(nsim, paths)
    # velocity_field = reader.velocity_field(MAS, grid)

    # reader = csiborgtools.read.CSiBORG2Field(nsim, "random", paths)
    # velocity_field = reader.velocity_field(MAS, grid)

    folder = "/mnt/extraspace/rstiskalek/catalogs"
    fpath = join(folder, "twompp_velocity_carrick2015.npy")
    field = np.load(fpath).astype(np.float32)
    field[0] -= 89
    field[1] -= -131
    field[2] -= 17
    # field /= 0.43
    return field

    # return velocity_field


def uniform_points_at_radius(R, npoints, seed):
    gen = np.random.RandomState(seed)
    phi = gen.uniform(0, 2*np.pi, npoints)
    theta = np.arccos(gen.uniform(-1, 1, npoints))

    return R * np.vstack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)]).T


def main_field(velocity_field, radius, observers, npoints, fname, boxsize):
    if exists(fname):
        print(f"Removing existing file `{fname}`.", flush=True)
        remove(fname)

    points = uniform_points_at_radius(radius, npoints, 42)
    for i, observer in enumerate(observers):
        current_points = (points + observer)
        current_points_box_units = current_points / boxsize
        vel = np.vstack([csiborgtools.field.evaluate_cartesian_cic(
            velocity_field[i], pos=current_points_box_units,
            smooth_scales=None,) for i in range(3)]).T

        vel_observer = np.vstack([csiborgtools.field.evaluate_cartesian_cic(
            velocity_field[i],
            pos=np.asarray(observer).reshape(1, 3) / boxsize,
            smooth_scales=None) for i in range(3)]).T[0]

        with File(fname, 'a') as f:
            grp = f.create_group(f"obs_{i}")
            grp.create_dataset("pos", data=current_points)
            grp.create_dataset("vel", data=vel)
            grp.create_dataset("observer", data=observer)
            grp.create_dataset("vel_observer", data=vel_observer)

    print(f"Written to `{fname}`.", flush=True)


###############################################################################
#                  Selecting particles in a thin radial shell.                #
###############################################################################


def load_particles(nsim, paths):
    reader = csiborgtools.read.QuijoteSnapshot(nsim, 4, paths)
    fprint("reading particles positions...")
    pos = reader.coordinates()
    fprint("reading particles velocities...")
    vel = reader.velocities()
    fprint("finished reading the snapshot.")
    return pos, vel


def main_particles(pos, vel, rmin, rmax, observers, fname):
    if exists(fname):
        print(f"Removing existing file `{fname}`.", flush=True)
        remove(fname)

    for i, observer in enumerate(observers):
        r = np.linalg.norm(pos - observer, axis=1)
        mask = (r > rmin) & (r < rmax)
        print(f"Observer {i}: {mask.sum()} particles in the shell.")
        with File(fname, 'a') as f:
            grp = f.create_group(f"obs_{i}")
            grp.create_dataset("pos", data=pos[mask])
            grp.create_dataset("vel", data=vel[mask])
            grp.create_dataset("observer", data=observer)

    print(f"Written to `{fname}`.", flush=True)


###############################################################################
#                               Interface                                     #
###############################################################################


if __name__ == "__main__":
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    fdir = "/mnt/extraspace/rstiskalek/BBF/Quijote_field_points"

    # grid = 512
    # MAS = "PCS"
    # boxsize = csiborgtools.simname2boxsize("quijote")
    # rmax = 200
    # radius_evaluate = 150
    # npoints = 1000

    # fiducial_observers = csiborgtools.read.fiducial_observers(boxsize, rmax)
    # print(f"There are {len(fiducial_observers)} observers.")
    # # fiducial_observers = [[boxsize/2, boxsize/2, boxsize/2]]
    # nsims = [0]

    # for nsim in nsims:
    #     fname = join(fdir, f"field_points_{nsim}.h5")
    #     velocity_field = load_fields(nsim, MAS, grid, paths)
    #     main_field(velocity_field, radius_evaluate, fiducial_observers,
    #                npoints, fname, boxsize)

    rmin = 100
    rmax = 100.5
    boxsize = csiborgtools.simname2boxsize("quijote")
    fiducial_observers = csiborgtools.read.fiducial_observers(boxsize, 200)
    print(f"There are {len(fiducial_observers)} observers.")
    nsims = [0]

    for nsim in nsims:
        fname = join(fdir, f"particles_points_{nsim}.h5")
        pos, vel = load_particles(nsim, paths)
        main_particles(pos, vel, rmin, rmax, fiducial_observers, fname)

    # grid = None
    # MAS = None
    # boxsize = csiborgtools.simname2boxsize("Carrick2015")
    # radius_evaluate = 75
    # npoints = 1000

    # fiducial_observers = [[boxsize/2, boxsize/2, boxsize/2]]
    # nsims = [0]

    # for nsim in nsims:
    #     fname = join(fdir, f"Carrick2015_field_points_{nsim}.h5")
    #     velocity_field = load_fields(nsim, MAS, grid, paths)
    #     main_field(velocity_field, radius_evaluate, fiducial_observers,
    #                npoints, fname, boxsize)
