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
r"""
Script to process simulation files and create a single HDF5 file, in which
particles are sorted by the particle halo IDs.
"""
from argparse import ArgumentParser
from gc import collect

import h5py
import numpy
from mpi4py import MPI

import csiborgtools
from csiborgtools import fprint
from numba import jit
from taskmaster import work_delegation
from tqdm import trange
from utils import get_nsims


@jit(nopython=True, boundscheck=False)
def minmax_halo(hid, halo_ids, start_loop=0):
    """
    Find the start and end index of a halo in a sorted array of halo IDs.
    This is much faster than using `numpy.where` and then `numpy.min` and
    `numpy.max`.
    """
    start = None
    end = None

    for i in range(start_loop, halo_ids.size):
        n = halo_ids[i]
        if n == hid:
            if start is None:
                start = i
            end = i
        elif n > hid:
            break
    return start, end


def process_snapshot(nsim, simname, halo_finder, verbose):
    """
    Read in the snapshot particles, sort them by their halo ID and dump
    into a HDF5 file. Stores the first and last index of each halo in the
    particle array for fast slicing of the array to acces particles of a single
    halo.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, simname))

    if simname == "csiborg":
        partreader = csiborgtools.read.CSiBORGReader(paths)
        box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)
    else:
        partreader = csiborgtools.read.QuijoteReader(paths)
        box = None

    fname = paths.processed_output(nsim, simname, halo_finder)
    # We first read in the halo IDs of the particles and infer the sorting.
    # Right away we dump the halo IDs to a HDF5 file and clear up memory.
    fprint(f"loading PIDs of IC {nsim}.", verbose)
    hids = partreader.read_halo_id(nsnap, nsim, halo_finder, verbose)
    fprint("sorting PIDs of IC {nsim}.")
    sort_indxs = numpy.argsort(hids).astype(numpy.uint64)

    # Dump halo IDs
    fprint("Loading and dumping halo IDs", verbose)
    with h5py.File(fname, "w") as f:
        group = f.create_group("snapshot_final")
        group.attrs["header"] = "Snapshot data at z = 0."
        dset = group.create_dataset("halo_ids", data=hids[sort_indxs])
        dset.attrs["header"] = f"{halo_finder} particles' halo IDs"
        f.close()
    del hids
    collect()

    # Dump particle positions
    fprint("Loading and dumping particle positions", verbose)
    pos = partreader.read_snapshot(nsnap, nsim, "pos")[sort_indxs]
    with h5py.File(fname, "r+") as f:
        dset = f["snapshot_final"].create_dataset("pos", data=pos)
        dset.attrs["header"] = "DM particle positions in box units."
        f.close()
    del pos
    collect()

    # Dump velocities
    fprint("Loading and dumping particle velocities", verbose)
    vel = partreader.read_snapshot(nsnap, nsim, "vel")[sort_indxs]
    vel = box.box2vel(vel) if simname == "csiborg" else vel
    with h5py.File(fname, "r+") as f:
        dset = f["snapshot_final"].create_dataset("vel", data=vel)
        dset.attrs["header"] = "DM particle velocity in km / s."
        f.close()
    del vel
    collect()

    # Dump masses
    fprint("Loading and dumping particle masses", verbose)
    mass = partreader.read_snapshot(nsnap, nsim, "mass")[sort_indxs]
    mass = box.box2solarmass(mass) if simname == "csiborg" else mass
    with h5py.File(fname, "r+") as f:
        dset = f["snapshot_final"].create_dataset("mass", data=mass)
        dset.attrs["header"] = "DM particle mass in Msun / h."
        f.close()
    del mass
    collect()

    # Dump particle IDs
    fprint("Loading and dumping particle IDs", verbose)
    pid = partreader.read_snapshot(nsnap, nsim, "pid")[sort_indxs]
    with h5py.File(fname, "r+") as f:
        dset = f["snapshot_final"].create_dataset("pid", data=pid)
        dset.attrs["header"] = "DM particle ID."
        f.close()
    del pid
    collect()

    del sort_indxs
    collect()

    fprint(f"creating a halo map for {nsim}.")
    with h5py.File(fname, "r") as f:
        part_hids = f["snapshot_final"]["halo_ids"][:]
    # We loop over the unique halo IDs.
    unique_halo_ids = numpy.unique(part_hids)
    halo_map = numpy.full((unique_halo_ids.size, 3), numpy.nan,
                          dtype=numpy.uint64)
    start_loop, niters = 0, unique_halo_ids.size
    for i in trange(niters, disable=not verbose):
        hid = unique_halo_ids[i]
        k0, kf = minmax_halo(hid, part_hids, start_loop=start_loop)
        halo_map[i, :] = hid, k0, kf
        start_loop = kf

    # Dump the halo mapping.
    with h5py.File(fname, "r+") as f:
        dset = f["snapshot_final"].create_dataset("halo_map", data=halo_map)
        dset.attrs["header"] = """
        Halo to particle mapping. Columns are HID, start index, end index.
        """
        f.close()

    del part_hids
    collect()

    # Add the halo finder catalogue
    with h5py.File(fname, "r+") as f:
        group = f.create_group("halofinder_catalogue")
        group.attrs["header"] = f"Original {halo_finder} halo catalogue."
        cat = partreader.read_catalogue(nsnap, nsim, halo_finder)

        hid2pos = {hid: i for i, hid in enumerate(unique_halo_ids)}

        for key in cat.dtype.names:
            x = numpy.full(unique_halo_ids.size, numpy.nan,
                           dtype=cat[key].dtype)
            for i in range(len(cat)):
                j = hid2pos[cat["index"][i]]
                x[j] = cat[key][i]
            group.create_dataset(key, data=x)

    # Lastly create the halo catalogue
    with h5py.File(fname, "r+") as f:
        group = f.create_group("halo_catalogue")
        group.attrs["header"] = f"{halo_finder} halo catalogue."
        group.create_dataset("hid", data=unique_halo_ids)


def add_initial_snapshot(nsim, simname, halo_finder, verbose):
    """
    Sort the initial snapshot particles according to their final snapshot and
    add them to the final snapshot's HDF5 file.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    if simname == "csiborg":
        partreader = csiborgtools.read.CSiBORGReader(paths)
    else:
        partreader = csiborgtools.read.QuijoteReader(paths)

    fprint(f"processing simulation `{nsim}`.", verbose)
    nsnap = max(paths.get_snapshots(nsim, simname))
    if simname == "csiborg":
        nsnap0 = 1
    elif simname == "quijote":
        nsnap0 = -1
    else:
        raise ValueError(f"Unknown simulation `{simname}`.")

    pid0 = partreader.read_snapshot(nsnap0, nsim, "pid")
    pos = partreader.read_snapshot(nsnap0, nsim, "pos")

    # First enforce them to already be sorted and then apply reverse
    # sorting from the final snapshot.
    pos = pos[numpy.argsort(pid0)]
    del pid0
    collect()

    pidf = partreader.read_snapshot(nsnap, nsim, "pid")
    pos = pos[numpy.argsort(numpy.argsort(pidf))]

    del pidf
    collect()

    # In Quijote some particles are position precisely at the edge of the
    # box. Move them to be just inside.
    if simname == "quijote":
        mask = pos >= 1
        if numpy.any(mask):
            spacing = numpy.spacing(pos[mask])
            assert numpy.max(spacing) <= 1e-5
            pos[mask] -= spacing

    fname = paths.processed_output(nsim, simname, halo_finder)
    fprint(f"dumping particles for `{nsim}` to `{fname}`", verbose)
    with h5py.File(fname, "r+") as f:
        group = f.create_group("snapshot_initial")
        group.attrs["header"] = "Initial snapshot data."
        dset = group.create_dataset("pos", data=pos)
        dset.attrs["header"] = "DM particle positions in box units."


def calculate_initial(nsim, simname, halo_finder, verbose):
    """Calculate the Lagrangian patch centre of mass and size."""
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    fname = paths.processed_output(nsim, simname, halo_finder)
    fprint("loading the particle information.", verbose)
    with h5py.File(fname, "r") as f:
        pos = f["snapshot_initial"]["pos"][:]
        mass = f["snapshot_final"]["mass"][:]
        hid = f["halo_catalogue"]["hid"][:]
        hid2map = csiborgtools.read.make_halomap_dict(
            f["snapshot_final"]["halo_map"][:])

    if simname == "csiborg":
        kwargs = {"box_size": 2048, "bckg_halfsize": 512}
    else:
        kwargs = {"box_size": 512, "bckg_halfsize": 256}
    overlapper = csiborgtools.match.ParticleOverlap(**kwargs)

    lagpatch_pos = numpy.full((len(hid), 3), numpy.nan, dtype=numpy.float32)
    lagpatch_size = numpy.full(len(hid), numpy.nan, dtype=numpy.float32)
    lagpatch_ncells = numpy.full(len(hid), numpy.nan, dtype=numpy.int32)

    for i in trange(len(hid), disable=not verbose):
        h = hid[i]
        # These are unasigned particles.
        if h == 0:
            continue
        parts_pos = csiborgtools.read.load_halo_particles(h, pos, hid2map)
        parts_mass = csiborgtools.read.load_halo_particles(h, mass, hid2map)

        # Skip if the halo has no particles or is too small.
        if parts_pos is None or parts_pos.size < 20:
            continue

        cm = csiborgtools.center_of_mass(parts_pos, parts_mass, boxsize=1.0)
        sep = csiborgtools.periodic_distance(parts_pos, cm, boxsize=1.0)
        delta = overlapper.make_delta(parts_pos, parts_mass, subbox=True)

        lagpatch_pos[i] = cm
        lagpatch_size[i] = numpy.percentile(sep, 99)
        lagpatch_ncells[i] = csiborgtools.delta2ncells(delta)

    with h5py.File(fname, "r+") as f:
        grp = f["halo_catalogue"]
        dset = grp.create_dataset("lagpatch_pos", data=lagpatch_pos)
        dset.attrs["header"] = "Lagrangian patch centre of mass in box units."

        dset = grp.create_dataset("lagpatch_size", data=lagpatch_size)
        dset.attrs["header"] = "Lagrangian patch size in box units."

        dset = grp.create_dataset("lagpatch_ncells", data=lagpatch_ncells)
        dset.attrs["header"] = f"Lagrangian patch number of cells on a {kwargs['boxsize']}^3 grid."  # noqa

        f.close()


def main(nsim, args):
    # Process the final snapshot
    process_snapshot(nsim, args.simname, args.halofinder, True)
    # Then add do it the initial snapshot data
    add_initial_snapshot(nsim, args.simname, args.halofinder, True)
    # Calculate the Lagrangian patch size properties
    calculate_initial(nsim, args.simname, args.halofinder, True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, default="csiborg",
                        choices=["csiborg", "quijote"],
                        help="Simulation name")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. If `-1` processes all.")
    parser.add_argument("--halofinder", type=str, help="Halo finder")
    args = parser.parse_args()

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    nsims = get_nsims(args, paths)

    def _main(nsim):
        main(nsim, args)

    work_delegation(_main, nsims, MPI.COMM_WORLD)
