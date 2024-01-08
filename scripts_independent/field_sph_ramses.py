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
Script to construct the density and velocity fields for a simulation snapshot.
The SPH filter is implemented in the cosmotool package.
"""
from argparse import ArgumentParser
from os.path import join

import numpy as np
from h5py import File

from field_sph_gadget import now, run_sph_filter
from process_snapshot import CSiBORG1Reader


def prepare_csiborg1(nsim, output_path):
    """
    Prepare a RAMSES snapshot for the SPH filter.

    Parameters
    ----------
    nsim : int
        Simulation index.
    output_path : str
        Path to the output HDF5 file.

    Returns
    -------
    None
    """
    reader = CSiBORG1Reader(nsim, "final")

    with File(output_path, 'w') as target:
        print(f"{now()}: loading positions.")
        pos = reader.read_snapshot("pos")
        print(f"{now()}: loading velocities.")
        vel = reader.read_snapshot("vel")
        print(f"{now()}: loading masses.")
        mass = reader.read_snapshot("mass")

        print(f"Writing {len(pos)} particles to {output_path}.")
        dset = target.create_dataset("particles", (len(pos), 7),
                                     dtype=np.float32)

        dset[:, :3] = pos
        print(f"{now()}: written positions.")
        dset[:, 3:6] = vel
        print(f"{now()}: written velocities.")
        dset[:, 6] = mass
        print(f"{now()}: written masses.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate SPH density and velocity field.")  # noqa
    parser.add_argument("--nsim", type=int, required=True,
                        help="Simulation index")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["prepare", "run"], help="Mode of operation.")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to the output HDF5 file.")
    parser.add_argument("--resolution", type=int, required=True,
                        help="Resolution of the density and velocity field.")
    parser.add_argument("--scratch_space", type=str, required=True,
                        help="Path to a folder where temporary files can be stored.")  # noqa
    parser.add_argument("--SPH_executable", type=str, required=True,
                        help="Path to the `simple3DFilter` executable.")
    parser.add_argument("--snapshot_kind", type=str, required=True,
                        choices=["ramses"],
                        help="Kind of the simulation snapshot.")
    args = parser.parse_args()

    if args.snapshot_kind != "ramses":
        raise NotImplementedError("Only RAMSES snapshots are supported.")

    particles_path = join(args.scratch_space,
                          f"ramses_{str(args.nsim).zfill(5)}.hdf5")
    output_path = join(args.output_folder,
                       f"sph_ramses_{str(args.nsim).zfill(5)}.hdf5")

    print("---------- SPH Density & Velocity Field Job Information ----------")
    print(f"Mode:              {args.mode}")
    print(f"Simulation index:  {args.nsim}")
    print(f"Paticles path:     {particles_path}")
    print(f"Output path:       {output_path}")
    print(f"Resolution:        {args.resolution}")
    print(f"SPH executable:    {args.SPH_executable}")
    print(f"Snapshot kind:     {args.snapshot_kind}")
    print("------------------------------------------------------------------")
    print(flush=True)

    if args.mode == "prepare":
        prepare_csiborg1(args.nsim, particles_path)
    elif args.mode == "run":
        output_path = join(args.output_folder,
                           f"sph_ramses_{str(args.nsim).zfill(5)}.hdf5")
        boxsize = 677.7
        run_sph_filter(particles_path, output_path, boxsize, args.resolution,
                       args.SPH_executable)
    else:
        raise NotImplementedError(f"Mode `{args.mode}` is not supported.")
