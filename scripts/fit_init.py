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
parser.add_argument("--ics", type=int, nargs="+", default=None,
                    help="IC realisations. If `-1` processes all simulations.")
args = parser.parse_args()
paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
partreader = csiborgtools.read.ParticleReader(paths)

if args.ics is None or args.ics[0] == -1:
    ics = paths.get_ics(tonew=True)
else:
    ics = args.ics

cols_collect = [("index", numpy.int32),
                ("x", numpy.float32),
                ("y", numpy.float32),
                ("z", numpy.float32),
                ("lagpatch", numpy.float32),]


# MPI loop over simulations
jobs = csiborgtools.fits.split_jobs(len(ics), nproc)[rank]
for nsim in [ics[i] for i in jobs]:
    nsnap = max(paths.get_snapshots(nsim))
    print(f"{datetime.now()}: rank {rank} calculating simulation `{nsim}`.",
          flush=True)

    parts = csiborgtools.read.read_h5(paths.initmatch_path(nsim, "particles"))
    parts = parts['particles']
    clump_map = csiborgtools.read.read_h5(paths.particles_path(nsim))
    clump_map = clump_map["clumpmap"]
    clumps_cat = csiborgtools.read.ClumpsCatalogue(nsim, paths, rawdata=True,
                                                   load_fitted=False)
    clid2map = {clid: i for i, clid in enumerate(clump_map[:, 0])}
    ismain = clumps_cat.ismain

    out = csiborgtools.read.cols_to_structured(len(clumps_cat), cols_collect)
    indxs = clumps_cat["index"]
    for i, hid in enumerate(tqdm(indxs) if verbose else indxs):
        out["index"][i] = hid
        if not ismain[i]:
            continue

        part = csiborgtools.read.load_parent_particles(hid, parts, clump_map,
                                                       clid2map, clumps_cat)
        # Skip if the halo is too small.
        if part is None or part.size < 100:
            continue

        dist, cm = csiborgtools.fits.dist_centmass(part)
        # We enforce a maximum patchsize of 0.075 in box coordinates.
        patchsize = min(numpy.percentile(dist, 99), 0.075)
        out["x"][i], out["y"][i], out["z"][i] = cm
        out["lagpatch"][i] = patchsize

    out = out[ismain]
    # Now save it
    fout = paths.initmatch_path(nsim, "fit")
    print(f"{datetime.now()}: dumping fits to .. `{fout}`.",
          flush=True)
    with open(fout, "wb") as f:
        numpy.save(f, out)
