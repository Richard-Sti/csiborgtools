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
Script to sort the initial snapshot particles according to their final
snapshot ordering, which is sorted by the halo IDs.

Ensures the following units:
    - Positions in box units.
    - Masses in :math:`M_\odot / h`.
"""
from argparse import ArgumentParser
from datetime import datetime
from gc import collect

import h5py
import numpy
from mpi4py import MPI
from taskmaster import work_delegation

import csiborgtools
from utils import get_nsims


def _main(nsim, simname, verbose):
    """
    Sort the initial snapshot particles according to their final snapshot
    ordering and dump them into a HDF5 file.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    if simname == "csiborg":
        partreader = csiborgtools.read.CSiBORGReader(paths)
    else:
        partreader = csiborgtools.read.QuijoteReader(paths)

    print(f"{datetime.now()}:   processing simulation `{nsim}`.", flush=True)
    # We first load the particle IDs in the final snapshot.
    pidf = csiborgtools.read.read_h5(paths.particles(nsim, simname))
    pidf = pidf["particle_ids"]
    # Then we load the particles in the initil snapshot and make sure that
    # their particle IDs are sorted as in the final snapshot. Again, because of
    # precision this must be read as structured.
    if simname == "csiborg":
        pars_extract = ["x", "y", "z", "M", "ID"]
        # CSiBORG's initial snapshot ID
        nsnap = 1
    else:
        pars_extract = None
        # Use this to point the reader to the ICs snapshot
        nsnap = -1
    part0, pid0 = partreader.read_particle(
        nsnap, nsim, pars_extract, return_structured=False, verbose=verbose)

    # In CSiBORG we need to convert particle masses from box units.
    if simname == "csiborg":
        box = csiborgtools.read.CSiBORGBox(
            max(paths.get_snapshots(nsim, simname)), nsim, paths)
        part0[:, 3] = box.box2solarmass(part0[:, 3])

    # Quijote's initial snapshot information also contains velocities but we
    # don't need those.
    if simname == "quijote":
        part0 = part0[:, [0, 1, 2, 6]]
        # In Quijote some particles are position precisely at the edge of the
        # box. Move them to be just inside.
        pos = part0[:, :3]
        mask = pos >= 1
        if numpy.any(mask):
            spacing = numpy.spacing(pos[mask])
            assert numpy.max(spacing) <= 1e-5
            pos[mask] -= spacing

    # First enforce them to already be sorted and then apply reverse
    # sorting from the final snapshot.
    part0 = part0[numpy.argsort(pid0)]
    del pid0
    collect()
    part0 = part0[numpy.argsort(numpy.argsort(pidf))]
    fout = paths.initmatch(nsim, simname, "particles")
    if verbose:
        print(f"{datetime.now()}: dumping particles for `{nsim}` to `{fout}`",
              flush=True)
    with h5py.File(fout, "w") as f:
        f.create_dataset("particles", data=part0)


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, default="csiborg",
                        choices=["csiborg", "quijote"],
                        help="Simulation name")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. If `-1` processes all.")
    args = parser.parse_args()

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(args, paths)

    def main(nsim):
        _main(nsim, args.simname, MPI.COMM_WORLD.Get_size() == 1)

    work_delegation(main, nsims, MPI.COMM_WORLD)
