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
"""
from argparse import ArgumentParser
from datetime import datetime
from os.path import exists, join

import hdf5plugin
import numpy
import pynbody
from h5py import File


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


def convert_to_hdf5(snapshot_path, output_path=None):
    """
    Convert a RAMSES CSiBORG1 snapshot to a compressed HDF5 file.

    Parameters
    ----------
    snapshot_path : str
        Path to RAMSES snapshot folder.
    output_path : str, optional
        Path to output HDF5 file. By default stored in the same folder as the
        snapshot.

    Returns
    -------
    None
    """
    if output_path is None:
        output_path = join(snapshot_path, "compressed_snapshot.hdf5")

    blosc_kwargs = {"cname": "blosclz",
                    "clevel": 9,
                    "shuffle": hdf5plugin.Blosc.SHUFFLE,
                    }

    if exists(output_path):
        raise RuntimeError(f"Stopping! `{output_path}` already exists.")

    with File(output_path, 'w') as f:
        print(f"{datetime.now()}: creating dataset `ParticleIDs`...",
              flush=True)
        f.create_dataset("ParticleIDs",
                         data=load_snapshot(snapshot_path, "pid"),
                         **hdf5plugin.Blosc(**blosc_kwargs))

        print(f"{datetime.now()}: creating dataset `Coordinates`...",
              flush=True)
        box2mpch = 677.7
        f.create_dataset("Coordinates",
                         data=load_snapshot(snapshot_path, "pos") * box2mpch,
                         **hdf5plugin.Blosc(**blosc_kwargs))

        print(f"{datetime.now()}: creating dataset `Velocities`...",
              flush=True)
        box2kms = 67682.75228061239
        f.create_dataset("Velocities",
                         data=load_snapshot(snapshot_path, "vel") * box2kms,
                         **hdf5plugin.Blosc(**blosc_kwargs))

        print(f"{datetime.now()}: creating dataset `Masses`...",
              flush=True)
        box2msunh = 2.6543271649678946e+19
        f.create_dataset("Masses",
                         data=load_snapshot(snapshot_path, "mass") * box2msunh,
                         **hdf5plugin.Blosc(**blosc_kwargs))

        header = f.create_dataset("Header", (0,))
        header["BoxSize"] = 677.7  # Mpc/h
        header["Omega0"] = 0.307
        header["OmegaBaryon"] = 0.0
        header["OmegaLambda"] = 0.693
        header["HubleParam"] = 0.6777

        print(f"{datetime.now()}: done with `{snapshot_path}`.", flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert RAMSES snapshot to a compressed HDF5 file.")  # noqa
    parser.add_argument("--snapshot_path", type=str, required=True,
                        help="Path to RAMSES snapshot folder.")
    parser.add_argument("--output_path", type=str, required=False,
                        default=None,
                        help="Path to output HDF5 file. By default stored in the same folder as the snapshot.")  # noqa
    args = parser.parse_args()

    convert_to_hdf5(args.snapshot_path, args.output_path)
