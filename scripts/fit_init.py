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
from tqdm import tqdm

from utils import get_nsims

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools


# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
verbose = nproc == 1

# Argument parser
parser = ArgumentParser()
parser.add_argument("--simname", type=str, default="csiborg",
                    choices=["csiborg", "quijote"],
                    help="Simulation name")
parser.add_argument("--nsims", type=int, nargs="+", default=None,
                    help="IC realisations. If `-1` processes all simulations.")
args = parser.parse_args()
paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
partreader = csiborgtools.read.ParticleReader(paths)

nsims = get_nsims(args, paths)

cols_collect = [("index", numpy.int32),
                ("x", numpy.float32),
                ("y", numpy.float32),
                ("z", numpy.float32),
                ("lagpatch_size", numpy.float32),
                ("lagpatch_ncells", numpy.int32),]


# MPI loop over simulations
jobs = csiborgtools.fits.split_jobs(len(nsims), nproc)[rank]
for nsim in [nsims[i] for i in jobs]:
    nsnap = max(paths.get_snapshots(nsim))
    overlapper = csiborgtools.match.ParticleOverlap()
    print(f"{datetime.now()}: rank {rank} calculating simulation `{nsim}`.",
          flush=True)

    parts = csiborgtools.read.read_h5(paths.initmatch(nsim, "particles"))
    parts = parts['particles']
    halo_map = csiborgtools.read.read_h5(paths.particles(nsim))
    halo_map = halo_map["halomap"]
    cat = csiborgtools.read.CSiBORGHaloCatalogue(
        nsim, paths, rawdata=True, load_fitted=False, load_initial=False)
    hid2map = {hid: i for i, hid in enumerate(halo_map[:, 0])}

    out = csiborgtools.read.cols_to_structured(len(cat), cols_collect)
    for i, hid in enumerate(tqdm(cat["index"]) if verbose else cat["index"]):
        out["index"][i] = hid
        part = csiborgtools.read.load_halo_particles(hid, parts, halo_map,
                                                     hid2map)

        # Skip if the halo is too small.
        if part is None or part.size < 100:
            continue

        # Calculate the centre of mass and the Lagrangian patch size.
        dist, cm = csiborgtools.fits.dist_centmass(part)
        # We enforce a maximum patchsize of 0.075 in box coordinates.
        patchsize = min(numpy.percentile(dist, 99), 0.075)
        out["x"][i], out["y"][i], out["z"][i] = cm
        out["lagpatch_size"][i] = patchsize

        # Calculate the number of cells with > 0 density.
        delta = overlapper.make_delta(part[:, :3], part[:, 3], subbox=True)
        out["lagpatch_ncells"][i] = csiborgtools.fits.delta2ncells(delta)

    # Now save it
    fout = paths.initmatch(nsim, "fit")
    print(f"{datetime.now()}: dumping fits to .. `{fout}`.",
          flush=True)
    with open(fout, "wb") as f:
        numpy.save(f, out)
