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
from tqdm import trange, tqdm
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

    desc = {"hid": f"Halo finder ID ({halo_finder})of the particle.",
            "pos": "DM particle positions in box units.",
            "vel": "DM particle velocity in km / s.",
            "mass": "DM particle mass in Msun / h.",
            "pid": "DM particle ID",
            }

    fname = paths.processed_output(nsim, simname, halo_finder)

    fprint(f"loading HIDs of IC {nsim}.", verbose)
    hids = partreader.read_halo_id(nsnap, nsim, halo_finder, verbose)
    collect()

    fprint(f"sorting HIDs of IC {nsim}.")
    sort_indxs = numpy.argsort(hids)

    with h5py.File(fname, "w") as f:
        group = f.create_group("snapshot_final")
        group.attrs["header"] = "Snapshot data at z = 0."

        fprint("dumping halo IDs.", verbose)
        dset = group.create_dataset("halo_ids", data=hids[sort_indxs])
        dset.attrs["header"] = desc["hid"]
        del hids
        collect()

        fprint("reading, sorting and dumping the snapshot particles.", verbose)
        for kind in ["pos", "vel", "mass", "pid"]:
            x = partreader.read_snapshot(nsnap, nsim, kind)[sort_indxs]

            if simname == "csiborg" and kind == "vel":
                x = box.box2vel(x) if simname == "csiborg" else x

            if simname == "csiborg" and kind == "mass":
                x = box.box2solarmass(x) if simname == "csiborg" else x

            dset = f["snapshot_final"].create_dataset(kind, data=x)
            dset.attrs["header"] = desc[kind]
            del x
            collect()

    del sort_indxs
    collect()

    fprint(f"creating a halo map for IC {nsim}.")
    with h5py.File(fname, "r") as f:
        part_hids = f["snapshot_final"]["halo_ids"][:]
    # We loop over the unique halo IDs and remove the 0 halo ID
    unique_halo_ids = numpy.unique(part_hids)
    unique_halo_ids = unique_halo_ids[unique_halo_ids != 0]
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
        f.close()

    # Lastly create the halo catalogue
    with h5py.File(fname, "r+") as f:
        group = f.create_group("halo_catalogue")
        group.attrs["header"] = f"{halo_finder} halo catalogue."
        group.create_dataset("index", data=unique_halo_ids)
        f.close()


def add_initial_snapshot(nsim, simname, halo_finder, verbose):
    """
    Sort the initial snapshot particles according to their final snapshot and
    add them to the final snapshot's HDF5 file.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    fname = paths.processed_output(nsim, simname, halo_finder)

    if simname == "csiborg":
        partreader = csiborgtools.read.CSiBORGReader(paths)
    else:
        partreader = csiborgtools.read.QuijoteReader(paths)

    fprint(f"processing simulation `{nsim}`.", verbose)
    if simname == "csiborg":
        nsnap0 = 1
    elif simname == "quijote":
        nsnap0 = -1
    else:
        raise ValueError(f"Unknown simulation `{simname}`.")

    fprint("loading and sorting the initial PID.", verbose)
    sort_indxs = numpy.argsort(partreader.read_snapshot(nsnap0, nsim, "pid"))

    fprint("loading the final particles.", verbose)
    with h5py.File(fname, "r") as f:
        sort_indxs_final = f["snapshot_final/pid"][:]
        f.close()

    fprint("sorting the particles according to the final snapshot.", verbose)
    sort_indxs_final = numpy.argsort(numpy.argsort(sort_indxs_final))
    sort_indxs = sort_indxs[sort_indxs_final]

    del sort_indxs_final
    collect()

    fprint("loading and sorting the initial particle position.", verbose)
    pos = partreader.read_snapshot(nsnap0, nsim, "pos")[sort_indxs]

    del sort_indxs
    collect()

    # In Quijote some particles are position precisely at the edge of the
    # box. Move them to be just inside.
    if simname == "quijote":
        mask = pos >= 1
        if numpy.any(mask):
            spacing = numpy.spacing(pos[mask])
            assert numpy.max(spacing) <= 1e-5
            pos[mask] -= spacing

    fprint(f"dumping particles for `{nsim}` to `{fname}`.", verbose)
    with h5py.File(fname, "r+") as f:
        if "snapshot_initial" in f.keys():
            del f["snapshot_initial"]
        group = f.create_group("snapshot_initial")
        group.attrs["header"] = "Initial snapshot data."
        dset = group.create_dataset("pos", data=pos)
        dset.attrs["header"] = "DM particle positions in box units."

        f.close()


def calculate_initial(nsim, simname, halo_finder, verbose):
    """Calculate the Lagrangian patch centre of mass and size."""
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    fname = paths.processed_output(nsim, simname, halo_finder)
    fprint("loading the particle information.", verbose)
    f = h5py.File(fname, "r")
    pos = f["snapshot_initial/pos"]
    mass = f["snapshot_final/mass"]
    hid = f["halo_catalogue/index"][:]
    hid2map = csiborgtools.read.make_halomap_dict(
        f["snapshot_final/halo_map"][:])

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
        if parts_pos is None or parts_pos.size < 5:
            continue

        cm = csiborgtools.center_of_mass(parts_pos, parts_mass, boxsize=1.0)
        sep = csiborgtools.periodic_distance(parts_pos, cm, boxsize=1.0)
        delta = overlapper.make_delta(parts_pos, parts_mass, subbox=True)

        lagpatch_pos[i] = cm
        lagpatch_size[i] = numpy.percentile(sep, 99)
        lagpatch_ncells[i] = csiborgtools.delta2ncells(delta)

    f.close()
    collect()

    with h5py.File(fname, "r+") as f:
        grp = f["halo_catalogue"]
        dset = grp.create_dataset("lagpatch_pos", data=lagpatch_pos)
        dset.attrs["header"] = "Lagrangian patch centre of mass in box units."

        dset = grp.create_dataset("lagpatch_size", data=lagpatch_size)
        dset.attrs["header"] = "Lagrangian patch size in box units."

        dset = grp.create_dataset("lagpatch_ncells", data=lagpatch_ncells)
        dset.attrs["header"] = f"Lagrangian patch number of cells on a {kwargs['box_size']}^3 grid."  # noqa

        f.close()


def make_phew_halo_catalogue(nsim, verbose):
    """
    Process the PHEW halo catalogue for a CSiBORG simulation at all snapshots.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    snapshots = paths.get_snapshots(nsim, "csiborg")
    reader = csiborgtools.read.CSiBORGReader(paths)
    keys_write = ["index", "x", "y", "z", "mass_cl", "parent",
                  "ultimate_parent", "summed_mass"]

    # Create a HDF5 file to store all this.
    fname = paths.processed_phew(nsim)
    with h5py.File(fname, "w") as f:
        f.close()

    for nsnap in tqdm(snapshots, disable=not verbose, desc="Snapshot"):
        try:
            data = reader.read_phew_clumps(nsnap, nsim, verbose=False)
        except FileExistsError:
            continue

        with h5py.File(fname, "r+") as f:
            if str(nsnap) in f:
                print(f"Group {nsnap} already exists. Deleting.", flush=True)
                del f[str(nsnap)]
            grp = f.create_group(str(nsnap))
            for key in keys_write:
                grp.create_dataset(key, data=data[key])

            grp.attrs["header"] = f"CSiBORG PHEW clumps at snapshot {nsnap}."
            f.close()

    # Now write the redshifts
    scale_factors = numpy.full(len(snapshots), numpy.nan, dtype=numpy.float32)
    for i, nsnap in enumerate(snapshots):
        box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)
        scale_factors[i] = box._aexp

    redshifts = scale_factors[-1] / scale_factors - 1

    with h5py.File(fname, "r+") as f:
        grp = f.create_group("info")
        grp.create_dataset("redshift", data=redshifts)
        grp.create_dataset("snapshots", data=snapshots)
        grp.create_dataset("Om0", data=[box.Om0])
        grp.create_dataset("boxsize", data=[box.boxsize])
        f.close()


def make_merger_tree_file(nsim, verbose):
    """
    Process the `.dat` merger tree files and dump them into a HDF5 file.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    reader = csiborgtools.read.CSiBORGReader(paths)
    snaps = paths.get_snapshots(nsim, "csiborg")

    fname = paths.processed_merger_tree(nsim)
    with h5py.File(fname, "w") as f:
        f.close()

    for nsnap in tqdm(snaps, desc="Loading merger files",
                      disable=not verbose):
        try:
            data = reader.read_merger_tree(nsnap, nsim)
        except FileExistsError:
            continue

        with h5py.File(fname, "r+") as f:
            grp = f.create_group(str(nsnap))

            grp.create_dataset("clump",
                               data=data[:, 0].astype(numpy.int32))
            grp.create_dataset("progenitor",
                               data=data[:, 1].astype(numpy.int32))
            grp.create_dataset("progenitor_outputnr",
                               data=data[:, 2].astype(numpy.int32))
            grp.create_dataset("desc_mass",
                               data=data[:, 3].astype(numpy.float32))
            grp.create_dataset("desc_npart",
                               data=data[:, 4].astype(numpy.int32))
            grp.create_dataset("desc_pos",
                               data=data[:, 5:8].astype(numpy.float32))
            grp.create_dataset("desc_vel",
                               data=data[:, 8:11].astype(numpy.float32))
            f.close()


def append_merger_tree_mass_to_phew_catalogue(nsim, verbose):
    """
    Append mass of haloes from mergertree files to the PHEW catalogue. The
    difference between this and the PHEW value is that the latter is written
    before unbinding is performed.

    Note that currently only does this for the highest snapshot.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    snapshots = paths.get_snapshots(nsim, "csiborg")
    merger_reader = csiborgtools.read.MergerReader(nsim, paths)

    for nsnap in tqdm(snapshots, disable=not verbose, desc="Snapshot"):
        # TODO do this for all later
        if nsnap < 930:
            continue
        try:
            phewcat = csiborgtools.read.CSiBORGPHEWCatalogue(nsnap, nsim,
                                                             paths)
        except ValueError:
            phewcat.close()
            continue

        mergertree_mass = merger_reader.match_mass_to_phewcat(phewcat)
        phewcat.close()

        fname = paths.processed_phew(nsim)
        with h5py.File(fname, "r+") as f:
            grp = f[str(nsnap)]
            grp.create_dataset("mergertree_mass_new", data=mergertree_mass)
            f.close()


def main(nsim, args):
    if args.make_final:
        process_snapshot(nsim, args.simname, args.halofinder, True)

    if args.make_initial:
        add_initial_snapshot(nsim, args.simname, args.halofinder, True)
        calculate_initial(nsim, args.simname, args.halofinder, True)

    if args.make_phew:
        make_phew_halo_catalogue(nsim, True)

    if args.make_merger:
        make_merger_tree_file(nsim, True)

    if args.append_merger_mass:
        append_merger_tree_mass_to_phew_catalogue(nsim, True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, default="csiborg",
                        choices=["csiborg", "quijote"],
                        help="Simulation name")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. If `-1` processes all.")
    parser.add_argument("--halofinder", type=str, help="Halo finder")
    parser.add_argument("--make_final", action="store_true", default=False,
                        help="Process the final snapshot.")
    parser.add_argument("--make_initial", action="store_true", default=False,
                        help="Process the initial snapshot.")
    parser.add_argument("--make_phew", action="store_true", default=False,
                        help="Process the PHEW halo catalogue.")
    parser.add_argument("--make_merger", action="store_true", default=False,
                        help="Process the merger tree files.")
    parser.add_argument("--append_merger_mass", action="store_true",
                        default=False,
                        help="Append the merger tree mass to the PHEW cat.")

    args = parser.parse_args()
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(args, paths)

    def _main(nsim):
        main(nsim, args)

    work_delegation(_main, nsims, MPI.COMM_WORLD)
