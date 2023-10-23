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
"""Convert the HDF5 CSiBORG particle file to an ASCII file."""
from argparse import ArgumentParser

import h5py
import numpy
from mpi4py import MPI
from taskmaster import work_delegation
from tqdm import trange

import csiborgtools
from utils import get_nsims


def positions_to_ascii(positions, output_filename, boxsize=None,
                       chunk_size=50_000, verbose=True):
    """
    Convert array of positions to an ASCII file. If `boxsize` is given,
    multiples the positions by it.
    """
    total_size = len(positions)

    if verbose:
        print(f"Number of rows to write: {total_size}")

    with open(output_filename, 'w') as out_file:
        # Write the header
        out_file.write("#px py pz\n")

        # Loop through data in chunks
        for i in trange(0, total_size, chunk_size,
                        desc=f"Writing to ... `{output_filename}`",
                        disable=not verbose):

            end = i + chunk_size
            if end > total_size:
                end = total_size

            data_chunk = positions[i:end]
            # Convert to positions Mpc / h
            data_chunk = data_chunk[:, :3]

            if boxsize is not None:
                data_chunk *= boxsize

            chunk_str = "\n".join([f"{x:.4f} {y:.4f} {z:.4f}"
                                   for x, y, z in data_chunk])
            out_file.write(chunk_str + "\n")


def extract_positions(nsim, simname, paths, kind):
    """
    Extract either the particle or halo positions.
    """
    if kind == "particles":
        fname = paths.processed_output(nsim, simname, "FOF")
        return h5py.File(fname, 'r')["snapshot_final/pos"][:]

    if kind == "particles_rsp":
        raise NotImplementedError("RSP of particles is not implemented yet.")

    fpath = paths.observer_peculiar_velocity("PCS", 512, nsim)
    vpec_observer = numpy.load(fpath)["observer_vp"][0, :]
    cat = csiborgtools.read.CSiBORGHaloCatalogue(
        nsim, paths, "halo_catalogue", "FOF", bounds={"dist": (0, 155.5)},
        observer_velocity=vpec_observer)

    if kind == "halos":
        return cat["cartesian_pos"]

    if kind == "halos_rsp":
        return cat["cartesian_redshift_pos"]

    raise ValueError(f"Unknown kind `{kind}`. Allowed values are: "
                     "`particles`, `particles_rsp`, `halos`, `halos_rsp`.")


def main(args, paths):
    boxsize = 677.7 if "particles" in args.kind else None
    pos = extract_positions(args.nsim, args.simname, paths, args.kind)
    output_filename = paths.ascii_positions(args.nsim, args.kind)
    positions_to_ascii(pos, output_filename, boxsize=boxsize)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--kind", type=str, required=True,
                        choices=["particles", "particles_rsp", "halos", "halos_rsp"],  # noqa
                        help="Kind of data to extract.")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. If `-1` processes all.")
    parser.add_argument("--simname", type=str, default="csiborg",
                        choices=["csiborg"],
                        help="Simulation name")
    args = parser.parse_args()

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(args, paths)

    def _main(nsim):
        main(nsim, paths, args.kind)

    work_delegation(_main, nsims, MPI.COMM_WORLD)
