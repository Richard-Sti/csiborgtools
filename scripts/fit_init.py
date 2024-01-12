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
Script to calculate the particle centre of mass and Lagrangian patch size in
the initial snapshot. The initial snapshot particles are read from the sorted
files.
"""
from argparse import ArgumentParser
from datetime import datetime

import csiborgtools
import numpy
from mpi4py import MPI
from taskmaster import work_delegation
from tqdm import tqdm

from utils import get_nsims


def _main(nsim, simname, verbose):
    """
    Calculate and save the Lagrangian halo centre of mass and Lagrangian patch
    size in the initial snapshot.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    simname : str
        Simulation name.
    verbose : bool
        Verbosity flag.

    Returns
    -------
    None
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    cols = [("index", numpy.int32),
            ("x", numpy.float32),
            ("y", numpy.float32),
            ("z", numpy.float32),
            ("lagpatch_size", numpy.float32),
            ("lagpatch_ncells", numpy.int32),]

    if simname == "csiborg1":
        snap = csiborgtools.read.CSiBORG1Snapshot(nsim, 1, paths,
                                                  keep_snapshot_open=True)
        cat = csiborgtools.read.CSiBORG1Catalogue(nsim, paths, snapshot=snap)
        fout = f"/mnt/extraspace/rstiskalek/csiborg1/chain_{nsim}/initial_lagpatch.npy"  # noqa
    elif "csiborg2" in simname:
        kind = simname.split("_")[-1]
        snap = csiborgtools.read.CSiBORG2Snapshot(nsim, 0, kind, paths,
                                                  keep_snapshot_open=True)
        cat = csiborgtools.read.CSiBORG2Catalogue(nsim, 99, kind, paths,
                                                  snapshot=snap)
        fout = f"/mnt/extraspace/rstiskalek/csiborg2_{kind}/catalogues/initial_lagpatch_{nsim}.npy"  # noqa
    elif simname == "quijote":
        snap = csiborgtools.read.QuijoteSnapshot(nsim, "ICs", paths,
                                                 keep_snapshot_open=True)
        cat = csiborgtools.read.QuijoteHaloCatalogue(nsim, paths,
                                                     snapshot=snap)
        fout = f"/mnt/extraspace/rstiskalek/quijote/fiducial_processed/chain_{nsim}/initial_lagpatch.npy"  # noqa
    else:
        raise ValueError(f"Unknown simulation name `{simname}`.")

    boxsize = csiborgtools.simname2boxsize(simname)

    # Initialise the overlapper.
    if simname == "csiborg" or "csiborg2" in simname:
        kwargs = {"box_size": 2048, "bckg_halfsize": 512}
    else:
        kwargs = {"box_size": 512, "bckg_halfsize": 256}
    overlapper = csiborgtools.match.ParticleOverlap(**kwargs)

    out = csiborgtools.read.cols_to_structured(len(cat), cols)
    for i, hid in enumerate(tqdm(cat["index"]) if verbose else cat["index"]):
        out["index"][i] = hid

        pos = cat.snapshot.halo_coordinates(hid)
        mass = cat.snapshot.halo_masses(hid)

        # Calculate the centre of mass and the Lagrangian patch size.
        cm = csiborgtools.center_of_mass(pos, mass, boxsize=boxsize)
        distances = csiborgtools.periodic_distance(pos, cm, boxsize=boxsize)
        out["x"][i], out["y"][i], out["z"][i] = cm
        out["lagpatch_size"][i] = numpy.percentile(distances, 99)

        pos /= boxsize  # need to normalize the positions to be [0, 1).
        # Calculate the number of cells with > 0 density.
        delta = overlapper.make_delta(pos, mass, subbox=True)
        out["lagpatch_ncells"][i] = csiborgtools.delta2ncells(delta)

    # Now save it
    if verbose:
        print(f"{datetime.now()}: dumping fits to .. `{fout}`.", flush=True)
    with open(fout, "wb") as f:
        numpy.save(f, out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str,
                        choices=["csiborg1", "csiborg2_main", "csiborg2_random", "csiborg2_varysmalll", "quijote"],  # noqa
                        help="Simulation name")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. If `-1` processes all.")
    args = parser.parse_args()

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(args, paths)

    def main(nsim):
        _main(nsim, args.simname, MPI.COMM_WORLD.Get_size() == 1)

    work_delegation(main, nsims, MPI.COMM_WORLD)
