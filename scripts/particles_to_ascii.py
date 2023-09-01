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

import csiborgtools
import h5py

from mpi4py import MPI

from utils import get_nsims
from tqdm import trange

from taskmaster import work_delegation


def h5_to_ascii(nsim, paths, chunk_size=50_000, verbose=True):
    """
    Convert the HDF5 CSiBORG particle file to an ASCII file. Outputs only
    particle positions in Mpc / h. Ignores the unequal particle masses.
    """
    fname = paths.particles(nsim, args.simname)
    boxsize = 677.7

    fname_out = fname.replace(".h5", ".txt")

    with h5py.File(fname, 'r') as f:
        dataset = f["particles"]
        total_size = dataset.shape[0]

        if verbose:
            print(f"Number of rows to write: {total_size}")

        with open(fname_out, 'w') as out_file:
            # Write the header
            out_file.write("#px,py,pz\n")

            # Loop through data in chunks
            for i in trange(0, total_size, chunk_size,
                            desc=f"Writing to ... `{fname_out}`",
                            disable=not verbose):
                end = i + chunk_size
                if end > total_size:
                    end = total_size

                data_chunk = dataset[i:end]
                # Convert to positions Mpc / h
                data_chunk = data_chunk[:, :3] * boxsize

                chunk_str = "\n".join([f"{x:.4f},{y:.4f},{z:.4f}"
                                       for x, y, z in data_chunk])
                out_file.write(chunk_str + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. If `-1` processes all.")
    parser.add_argument("--simname", type=str, default="csiborg",
                        choices=["csiborg"],
                        help="Simulation name")
    args = parser.parse_args()

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(args, paths)

    def main(nsim):
        h5_to_ascii(nsim, paths, verbose=MPI.COMM_WORLD.Get_size() == 1)

    work_delegation(main, nsims, MPI.COMM_WORLD)
