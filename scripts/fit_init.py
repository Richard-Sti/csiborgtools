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
from os import remove
from os.path import join

import joblib
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
ftemp = lambda kind, nsim, rank: join(paths.temp_dumpdir, f"{kind}_{nsim}_{rank}.p")  # noqa

if args.ics is None or args.ics[0] == -1:
    ics = paths.get_ics(tonew=True)
else:
    ics = args.ics

cols_collect = [("index", numpy.int32),
                ("x", numpy.float32),
                ("y", numpy.float32),
                ("z", numpy.float32),
                ("lagpatch", numpy.float32),]


# We loop over simulations. Each simulation is then procesed with MPI.
for nsim in ics:
    nsnap = max(paths.get_snapshots(nsim))
    if rank == 0:
        print(f"{datetime.now()}: calculating simulation `{nsim}`.",
              flush=True)

    parts = csiborgtools.read.read_h5(paths.initmatch_path(nsim, "particles"))
    parts = parts['particles']
    clump_map = csiborgtools.read.read_h5(paths.particles_path(nsim))
    clump_map = clump_map["clump_map"]
    clumps_cat = csiborgtools.read.ClumpsCatalogue(nsim, paths, rawdata=True,
                                                   load_fitted=False)
    clid2map = {clid: i for i, clid in enumerate(clump_map[:, 0])}

    ntasks = len(clumps_cat)
    ismain = clumps_cat.ismain

    jobs = csiborgtools.fits.split_jobs(ntasks, nproc)[rank]
    _out = csiborgtools.read.cols_to_structured(len(jobs), cols_collect)
    for i, j in enumerate(tqdm(jobs)) if nproc == 1 else enumerate(jobs):
        hid = clumps_cat["index"][j]
        _out["index"][i] = hid
        part = csiborgtools.fits.load_parent_particles(hid, parts, clump_map,
                                                       clid2map, clumps_cat)
        # Skip if the halo is too small.
        if part is None or part.size < 100:
            continue

        raddist, cmpos = csiborgtools.match.dist_centmass(part)
        patchsize = csiborgtools.match.dist_percentile(raddist, [99],
                                                       distmax=0.075)
        _out["x"][i], _out["y"][i], _out["z"][i] = cmpos
        _out["lagpatch"][i] = patchsize
    # Dump the results of this rank to a temporary file.
    joblib.dump(_out, ftemp("fits", nsim, rank))

    # Now we wait for all ranks, then collect the results and save it.
    comm.Barrier()
    if rank == 0:
        print(f"{datetime.now()}: collecting results for {nsim}.", flush=True)
        out = csiborgtools.read.cols_to_structured(ntasks, cols_collect)
        hid2arrpos = {indx: i for i, indx in enumerate(clumps_cat["index"])}
        for i in range(nproc):
            inp = joblib.load(ftemp("fits", nsim, i))
            for j in range(inp.size):
                k = hid2arrpos[inp["index"][j]]
                for key in inp.dtype.names:
                    out[key][k] = inp[key][j]

            remove(ftemp("fits", nsim, i))

        # Now save it
        fout_fit = paths.initmatch_path(nsim, "fit")
        print(f"{datetime.now()}: dumping fits to .. `{fout_fit}`.",
              flush=True)
        with open(fout_fit, "wb") as f:
            numpy.save(f, out)
