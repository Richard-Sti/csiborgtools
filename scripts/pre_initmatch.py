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
Script to calculate the particle centre of mass and Lagrangian patch size in the initial
snapshot. Optinally dumps the particle files, however this requires a lot of memory.
"""
from argparse import ArgumentParser
from datetime import datetime
from distutils.util import strtobool
from gc import collect
from os import remove
from os.path import join

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
parser.add_argument("--dump", type=lambda x: bool(strtobool(x)))
args = parser.parse_args()
paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
partreader = csiborgtools.read.ParticleReader(paths)
ftemp = join(paths.temp_dumpdir, "initmatch_{}_{}_{}.npy")

# We loop over all particles and then use MPI when matching halos to the
# initial snapshot and dumping them.
for i, nsim in enumerate(paths.get_ics(tonew=True)):
    if rank == 0:
        print("{}: reading simulation {}.".format(datetime.now(), nsim), flush=True)
    nsnap = max(paths.get_snapshots(nsim))

    # We first load particles in the initial and final snapshots and sort them
    # by their particle IDs so that we can match them by array position.
    # `clump_ids` are the clump IDs of particles.
    part0 = partreader.read_particle(
        1, nsim, ["x", "y", "z", "M", "ID"], verbose=verbose
    )
    part0 = part0[numpy.argsort(part0["ID"])]

    pid = partreader.read_particle(nsnap, nsim, ["ID"], verbose=verbose)["ID"]
    clump_ids = partreader.read_clumpid(nsnap, nsim, verbose=verbose)
    clump_ids = clump_ids[numpy.argsort(pid)]
    # Release the particle IDs, we will not need them anymore now that both
    # particle arrays are matched in ordering.
    del pid
    collect()

    # Particles whose clump ID is 0 are unassigned to a clump, so we can get
    # rid of them to speed up subsequent operations. Again we release the mask.
    mask = clump_ids > 0
    clump_ids = clump_ids[mask]
    part0 = part0[mask]
    del mask
    collect()

    # Calculate the centre of mass of each parent halo, the Lagrangian patch
    # size and optionally the initial snapshot particles belonging to this
    # parent halo. Dumping the particles will take majority of time.
    if rank == 0:
        print(
            "{}: calculating {}th simulation {}.".format(datetime.now(), i, nsim),
            flush=True,
        )
    # We load up the clump catalogue which contains information about the
    # ultimate  parent halos of each clump. We will loop only over the clump
    # IDs of ultimate parent halos and add their substructure particles and at
    # the end save these.
    cat = csiborgtools.read.ClumpsCatalogue(
        nsim, paths, load_fitted=False, rawdata=True
    )
    parent_ids = cat["index"][cat.ismain][:500]
    jobs = csiborgtools.fits.split_jobs(parent_ids.size, nproc)[rank]
    for i in tqdm(jobs) if verbose else jobs:
        clid = parent_ids[i]
        mmain_indxs = cat["index"][cat["parent"] == clid]

        mmain_mask = numpy.isin(clump_ids, mmain_indxs, assume_unique=True)
        mmain_particles = part0[mmain_mask]

        raddist, cmpos = csiborgtools.match.dist_centmass(mmain_particles)
        patchsize = csiborgtools.match.dist_percentile(raddist, [99], distmax=0.075)
        with open(ftemp.format(nsim, clid, "fit"), "wb") as f:
            numpy.savez(f, cmpos=cmpos, patchsize=patchsize)

        if args.dump:
            with open(ftemp.format(nsim, clid, "particles"), "wb") as f:
                numpy.save(f, mmain_particles)

    # We force clean up the memory before continuing.
    del part0, clump_ids
    collect()

    # We now wait for all processes and then use the 0th process to collect the results.
    # We first collect just the Lagrangian patch size information.
    comm.Barrier()
    if rank == 0:
        print("{}: collecting fits...".format(datetime.now()), flush=True)
        dtype = {
            "names": ["index", "x", "y", "z", "lagpatch"],
            "formats": [numpy.int32] + [numpy.float32] * 4,
        }
        out = numpy.full(parent_ids.size, numpy.nan, dtype=dtype)
        for i, clid in enumerate(parent_ids):
            fpath = ftemp.format(nsim, clid, "fit")
            with open(fpath, "rb") as f:
                inp = numpy.load(f)
                out["index"][i] = clid
                out["x"][i] = inp["cmpos"][0]
                out["y"][i] = inp["cmpos"][1]
                out["z"][i] = inp["cmpos"][2]
                out["lagpatch"][i] = inp["patchsize"]
            remove(fpath)

        fout = paths.initmatch_path(nsim, "fit")
        print("{}: dumping fits to .. `{}`.".format(datetime.now(), fout), flush=True)
        with open(fout, "wb") as f:
            numpy.save(f, out)

        # We now optionally collect the individual clumps and store them in an archive,
        # which has the benefit of being a single file that can be easily read in.
        if args.dump:
            print("{}: collecting particles...".format(datetime.now()), flush=True)
            out = {}
            for clid in parent_ids:
                fpath = ftemp.format(nsim, clid, "particles")
                with open(fpath, "rb") as f:
                    out.update({str(clid): numpy.load(f)})

            fout = paths.initmatch_path(nsim, "particles")
            print(
                "{}: dumping particles to .. `{}`.".format(datetime.now(), fout),
                flush=True,
            )
            with open(fout, "wb") as f:
                numpy.savez(f, **out)

            # Again we force clean up the memory before continuing.
            del out
            collect()
