
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
A script to calculate the mean and standard deviation of a field at different
distances from the center of the box such that at each distance the field is
evaluated at uniformly-spaced points on a sphere.

The script is not parallelized in any way but it should not take very long, the
main bottleneck is reading the data from disk.
"""
from argparse import ArgumentParser
from os.path import join
from gc import collect

import csiborgtools
import numpy
from tqdm import tqdm
from numba import jit

from datetime import datetime


###############################################################################
#                Read in information about the simulation                     #
###############################################################################


def t():
    return datetime.now()


def get_reader(simname, paths, nsim):
    """
    Get the appropriate snaspshot reader for the simulation.

    Parameters
    ----------
    simname : str
        Name of the simulation.
    paths : csiborgtools.read.Paths
        Paths object.
    nsim : int
        Simulation index.

    Returns
    -------
    reader : instance of csiborgtools.read.BaseSnapshot
        Snapshot reader.
    """
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


def get_particles(reader, boxsize, verbose=True):
    """
    Get the distance of particles from the center of the box and their masses.

    Parameters
    ----------
    reader : instance of csiborgtools.read.BaseSnapshot
        Snapshot reader.
    boxsize : float
        Box size in Mpc / h.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    dist : 1-dimensional array
        Distance of particles from the center of the box.
    mass : 1-dimensional array
        Mass of particles.
    """
    if verbose:
        print(f"{t()},: reading coordinates and calculating radial distance.")
    pos = reader.coordinates()
    pos -= boxsize / 2
    dist = numpy.linalg.norm(pos, axis=1)
    del pos
    collect()

    if verbose:
        print(f"{t()}: reading masses.")
    mass = reader.masses()

    if verbose:
        print(f"{t()}: sorting coordinates and masses.")
    indxs = numpy.argsort(dist)
    dist = dist[indxs]
    mass = mass[indxs]
    del indxs
    collect()

    return dist, mass


###############################################################################
#                Calculate the enclosed mass at each distance                 #
###############################################################################


@jit(nopython=True, boundscheck=False)
def _enclosed_mass(rdist, mass, rmax, start_index):
    enclosed_mass = 0.

    for i in range(start_index, len(rdist)):
        if rdist[i] <= rmax:
            enclosed_mass += mass[i]
        else:
            break

    return enclosed_mass, i


def enclosed_mass(rdist, mass, distances):
    """
    Calculate the enclosed mass at each distance.

    Parameters
    ----------
    rdist : 1-dimensional array
        Distance of particles from the center of the box.
    mass : 1-dimensional array
        Mass of particles.
    distances : 1-dimensional array
        Distances at which to calculate the enclosed mass.

    Returns
    -------
    enclosed_mass : 1-dimensional array
        Enclosed mass at each distance.
    """
    enclosed_mass = numpy.full_like(distances, 0.)
    start_index = 0
    for i, dist in enumerate(distances):
        if i > 0:
            enclosed_mass[i] += enclosed_mass[i - 1]

        m, start_index = _enclosed_mass(rdist, mass, dist, start_index)
        enclosed_mass[i] += m

    return enclosed_mass


###############################################################################
#                       Main & command line interface                         #
###############################################################################


def main(args):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    boxsize = csiborgtools.simname2boxsize(args.simname)
    distances = numpy.linspace(0, boxsize / 2, 101)
    nsims = paths.get_ics(args.simname)
    folder = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/field_shells"

    cumulative_mass = numpy.zeros((len(nsims), len(distances)))
    mass135 = numpy.zeros(len(nsims))
    masstot = numpy.zeros(len(nsims))
    for i, nsim in enumerate(tqdm(nsims, desc="Simulations")):
        reader = get_reader(args.simname, paths, nsim)
        rdist, mass = get_particles(reader, nsim, boxsize)

        # Calculate the cnlosed mass
        cumulative_mass[i, :] = enclosed_mass(rdist, mass, distances)
        mass135[i] = enclosed_mass(rdist, mass, [135])[0]
        masstot[i] = numpy.sum(mass)

    # Finally save the output
    fname = f"enclosed_mass_{args.simname}.npz"
    fname = join(folder, fname)
    numpy.savez(fname, enclosed_mass=cumulative_mass, mass135=mass135,
                masstot=masstot, distances=distances)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, help="Simulation name.",
                        choices=["csiborg1", "csiborg2_main", "csiborg2_varysmall", "csiborg2_random"])  # noqa
    args = parser.parse_args()

    main(args)
