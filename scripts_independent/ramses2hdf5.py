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
Script to convert a RAMSES snapshot to a compressed HDF5 file. Be careful
because reading the HDF5 file requires `hdf5plugin` package to be installed.
If `halomaker_path` is provided, particles will be sorted by their halo ID to
allow for fast halo lookups.
"""
from argparse import ArgumentParser
from datetime import datetime
from gc import collect
from os.path import exists, join
from warnings import catch_warnings, filterwarnings

import hdf5plugin
import numpy
import pynbody
from h5py import File
from numba import jit
from tqdm import tqdm, trange


def fprint(msg, verbose=True):
    """Print and flush a message with a timestamp."""
    if verbose:
        print(f"{datetime.now()}:   {msg}", flush=True)


def load_snapshot(snapshot_path, kind):
    """
    Load a RAMSES snapshot and return the requested `kind` data.

    Parameters
    ----------
    snapshot_path : str
        Path to RAMSES snapshot folder.
    kind : str
        Kind of data to load. Options are: `pid`, `pos`, `vel` or `mass`.

    Returns
    -------
    x : np.ndarray
    """
    with catch_warnings():
        filterwarnings("ignore", category=UserWarning)
        sim = pynbody.load(snapshot_path)

    if kind == "pid":
        x = numpy.array(sim["iord"], dtype=numpy.uint32)
    elif kind == "pos" or kind == "mass":
        x = numpy.array(sim[kind], dtype=numpy.float32)
    elif kind == "vel":
        x = numpy.array(sim[kind], dtype=numpy.float16)
    else:
        raise ValueError(f"Unknown kind `{kind}`. "
                         "Options are: `pid`, `pos`, `vel` or `mass`.")

    return x


###############################################################################
#                           Halomaker support                                 #
###############################################################################


def read_halomaker_id(halomaker_path, pids):
    fprint("mapping particle IDs to their indices.")
    pids_idx = {pid: i for i, pid in enumerate(pids)}

    # Unassigned particle IDs are assigned a halo ID of 0.
    fprint("mapping HIDs to their array indices.")
    hids = numpy.zeros(pids.size, dtype=numpy.int32)

    # Read line-by-line to avoid loading the whole file into memory.
    with open(halomaker_path, 'r') as file:
        for line in tqdm(file, desc="Reading membership"):
            hid, pid = map(int, line.split())
            hids[pids_idx[pid]] = hid

    del pids_idx
    collect()

    return hids


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


def make_offset_map(part_hids):
    """
    Make group offsets for a list of particles' halo IDs. This is a
    2-dimensional array, where the first column is the halo ID, the second
    column is the start index of the halo in the particle list, and the third
    index is the end index of the halo in the particle list. The start index is
    inclusive, while the end index is exclusive.
    """
    unique_halo_ids = numpy.unique(part_hids)
    unique_halo_ids = unique_halo_ids[unique_halo_ids != 0]
    halo_map = numpy.full((unique_halo_ids.size, 3), numpy.nan,
                          dtype=numpy.uint64)
    start_loop, niters = 0, unique_halo_ids.size
    for i in trange(niters):
        hid = unique_halo_ids[i]
        k0, kf = minmax_halo(hid, part_hids, start_loop=start_loop)
        halo_map[i, :] = hid, k0, kf
        start_loop = kf

    return halo_map

###############################################################################
#                       Main conversion functions                             #
###############################################################################


def convert_to_hdf5(snapshot_path, output_path, halomaker_path=None):
    """
    Convert a RAMSES CSiBORG1 snapshot to a compressed HDF5 file.

    Parameters
    ----------
    snapshot_path : str
        Path to RAMSES snapshot folder.
    output_path : str, optional
        Path to output HDF5 file.
    halomaker_path : str, optional
        Path to HaloMaker particle membership file. Optional, if provided
        particles will be sorted by their halo membership.

    Returns
    -------
    None
    """
    blosc_kwargs = {"cname": "blosclz",
                    "clevel": 9,
                    "shuffle": hdf5plugin.Blosc.SHUFFLE,
                    }

    pids = load_snapshot(snapshot_path, "pid")

    if halomaker_path is not None:
        halo_ids = read_halomaker_id(halomaker_path, pids)
        fprint("sorting HIDs.")
        sort_indxs = numpy.argsort(halo_ids)
        halo_ids = halo_ids[sort_indxs]

    with File(output_path, 'w') as f:
        print(f"{datetime.now()}: creating dataset `ParticleIDs`...",
              flush=True)
        if halomaker_path is not None:
            pids = pids[sort_indxs]
        f.create_dataset("ParticleIDs", data=pids,
                         **hdf5plugin.Blosc(**blosc_kwargs))
        del pids
        collect()

        print(f"{datetime.now()}: creating dataset `Coordinates`...",
              flush=True)
        box2mpch = 677.7
        coord = load_snapshot(snapshot_path, "pos") * box2mpch
        if halomaker_path is not None:
            coord = coord[sort_indxs]
        f.create_dataset("Coordinates", data=coord,
                         **hdf5plugin.Blosc(**blosc_kwargs))
        del coord
        collect()

        print(f"{datetime.now()}: creating dataset `Velocities`...",
              flush=True)
        box2kms = 67682.75228061239
        vel = load_snapshot(snapshot_path, "vel") * box2kms
        if halomaker_path is not None:
            vel = vel[sort_indxs]
        f.create_dataset("Velocities", data=vel,
                         **hdf5plugin.Blosc(**blosc_kwargs))
        del vel
        collect()

        print(f"{datetime.now()}: creating dataset `Masses`...",
              flush=True)
        box2msunh = 2.6543271649678946e+19
        mass = load_snapshot(snapshot_path, "mass") * box2msunh
        if halomaker_path is not None:
            mass = mass[sort_indxs]
        f.create_dataset("Masses", data=mass,
                         **hdf5plugin.Blosc(**blosc_kwargs))

        header = f.create_dataset("Header", (0,))
        header.attrs["BoxSize"] = 677.7  # Mpc/h
        header.attrs["Omega0"] = 0.307
        header.attrs["OmegaBaryon"] = 0.0
        header.attrs["OmegaLambda"] = 0.693
        header.attrs["HubleParam"] = 0.6777

        print(f"{datetime.now()}: done with `{snapshot_path}`.", flush=True)

    if halomaker_path is not None:
        halo_map = make_offset_map(halo_ids)
        # Dump the halo mapping.
        with File(output_path, "r+") as f:
            dset = f["GroupOffset"].create_dataset("halo_map", data=halo_map)
            dset.attrs["header"] = """
            Halo to particle mapping. Columns are HID, start index, end index.
            """
            f.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert RAMSES snapshot to a compressed HDF5 file.")  # noqa
    parser.add_argument("--snapshot_path", type=str, required=True,
                        help="Path to RAMSES snapshot folder.")
    parser.add_argument("--output_path", type=str, required=False,
                        default=None,
                        help="Path to output HDF5 file. By default stored in the same folder as the snapshot.")  # noqa
    parser.add_argument("--halomaker_path", type=str, required=False,
                        default=None,
                        help="Path to HaloMaker particle membership file.")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = join(args.snapshot_path, "compressed_snapshot.hdf5")

    for path in [args.snapshot_path, args.halomaker_path]:
        if path is not None and not exists(path):
            raise RuntimeError(f"Stopping! `{path}` does not exist.")

    convert_to_hdf5(args.snapshot_path, args.output_path, args.halomaker_path)
