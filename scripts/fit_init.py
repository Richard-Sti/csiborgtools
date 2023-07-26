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
Script to calculate the particle centre of mass, Lagrangian patch size in the
initial snapshot. The initial snapshot particles are read from the sorted
files.
"""
from argparse import ArgumentParser
from datetime import datetime

import numpy
from mpi4py import MPI
from taskmaster import work_delegation
from tqdm import tqdm

from utils import get_nsims

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools


def _main(nsim, simname, verbose):
    """
    Calculate the Lagrangian halo centre of mass and Lagrangian patch size in
    the initial snapshot.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    simname : str
        Simulation name.
    verbose : bool
        Verbosity flag.
    """
    if simname == "quijote":
        raise NotImplementedError("Quijote not implemented yet.")

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    cols = [("index", numpy.int32),
            ("x", numpy.float32),
            ("y", numpy.float32),
            ("z", numpy.float32),
            ("lagpatch_size", numpy.float32),
            ("lagpatch_ncells", numpy.int32),]

    parts = csiborgtools.read.read_h5(paths.initmatch(nsim, "particles"))
    parts = parts['particles']
    halo_map = csiborgtools.read.read_h5(paths.particles(nsim, simname))
    halo_map = halo_map["halomap"]

    cat = csiborgtools.read.CSiBORGHaloCatalogue(
        nsim, paths, rawdata=True, load_fitted=False, load_initial=False)
    hid2map = {hid: i for i, hid in enumerate(halo_map[:, 0])}

    out = csiborgtools.read.cols_to_structured(len(cat), cols)
    for i, hid in enumerate(tqdm(cat["index"]) if verbose else cat["index"]):
        out["index"][i] = hid
        part = csiborgtools.read.load_halo_particles(hid, parts, halo_map,
                                                     hid2map)

        # Skip if the halo has no particles or is too small.
        if part is None or part.size < 100:
            continue

        pos, mass = part[:, :3], part[:, 3]
        # Calculate the centre of mass and the Lagrangian patch size.
        cm = csiborgtools.fits.center_of_mass(pos, mass, boxsize=1.0)
        distances = csiborgtools.fits.periodic_distance(pos, cm, boxsize=1.0)
        out["x"][i], out["y"][i], out["z"][i] = cm
        out["lagpatch_size"][i] = numpy.percentile(distances, 99)

        # Calculate the number of cells with > 0 density.
        overlapper = csiborgtools.match.ParticleOverlap()
        delta = overlapper.make_delta(pos, mass, subbox=True)
        out["lagpatch_ncells"][i] = csiborgtools.fits.delta2ncells(delta)

    # Now save it
    fout = paths.initmatch(nsim, "fit")
    if verbose:
        print(f"{datetime.now()}: dumping fits to .. `{fout}`.", flush=True)
    with open(fout, "wb") as f:
        numpy.save(f, out)


if __name__ == "__main__":
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
