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
initial snapshot and the particle mapping.
"""
from argparse import ArgumentParser
from os.path import join
from datetime import datetime
from gc import collect
import joblib
from os import remove

import h5py
import numpy
from mpi4py import MPI
from tqdm import trange

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

if args.ics is None or args.ics == -1:
    ics = paths.get_ics(tonew=True)
else:
    ics = args.ics

# We loop over simulations. Each simulation is then procesed with MPI, rank 0
# loads the data and broadcasts it to other ranks.
for nsim in ics:
    nsnap = max(paths.get_snapshots(nsim))
    if rank == 0:
        print(f"{datetime.now()}: reading simulation {nsim}.", flush=True)

        # We first load particles in the initial and final snapshots and sort
        # them by their particle IDs so that we can match them by array
        # position. `clump_ids` are the clump IDs of particles.
        part0 = partreader.read_particle(1, nsim, ["x", "y", "z", "M", "ID"],
                                         verbose=True,
                                         return_structured=False)
        part0 = part0[numpy.argsort(part0[:, -1])]
        part0 = part0[:, :-1]  # Now we no longer need the particle IDs

        pid = partreader.read_particle(nsnap, nsim, ["ID"], verbose=True,
                                       return_structured=False).reshape(-1, )
        clump_ids = partreader.read_clumpid(nsnap, nsim, verbose=True)
        clump_ids = clump_ids[numpy.argsort(pid)]
        # Release the particle IDs, we will not need them anymore now that both
        # particle arrays are matched in ordering.
        del pid
        collect()

        # Particles whose clump ID is 0 are unassigned to a clump, so we can
        # get rid of them to speed up subsequent operations. We will not need
        # these. Again we release the mask.
        mask = clump_ids > 0
        clump_ids = clump_ids[mask]
        part0 = part0[mask, :]
        del mask
        collect()

        print(f"{datetime.now()}: dumping particles for {nsim}.", flush=True)
        with h5py.File(paths.initmatch_path(nsim, "particles"), "w") as f:
            f.create_dataset("particles", data=part0)

        print(f"{datetime.now()}: broadcasting simulation {nsim}.", flush=True)
    # Stop all ranks and figure out array shapes from the 0th rank
    comm.Barrier()
    if rank == 0:
        shape = numpy.array([*part0.shape], dtype=numpy.int32)
    else:
        shape = numpy.empty(2, dtype=numpy.int32)
    comm.Bcast(shape, root=0)

    # Now broadcast the particle arrays to all ranks
    if rank > 0:
        part0 = numpy.empty(shape, dtype=numpy.float32)
        clump_ids = numpy.empty(shape[0], dtype=numpy.int32)

    comm.Bcast(part0, root=0)
    comm.Bcast(clump_ids, root=0)
    print(f"{datetime.now()}: simulation {nsim} broadcasted.", flush=True)
    print("rank", rank, clump_ids[:5], part0[:5, :], flush=True)

    # Calculate the centre of mass of each parent halo, the Lagrangian patch
    # size and optionally the initial snapshot particles belonging to this
    # parent halo. Dumping the particles will take majority of time.
    if rank == 0:
        print(f"{datetime.now()}: calculating simulation {nsim}.", flush=True)
    # We load up the clump catalogue which contains information about the
    # ultimate  parent halos of each clump. We will loop only over the clump
    # IDs of ultimate parent halos and add their substructure particles and at
    # the end save these.
    cat = csiborgtools.read.ClumpsCatalogue(nsim, paths, load_fitted=False,
                                            rawdata=True)
    parent_ids = cat["index"][cat.ismain]
    parent_ids = parent_ids[:100]
    hid2arrpos = {indx: j for j, indx in enumerate(parent_ids)}
    # And we pre-allocate the output array for this simulation.
    dtype = {"names": ["index", "x", "y", "z", "lagpatch"],
             "formats": [numpy.int32] + [numpy.float32] * 4}
    # We MPI loop over the individual halos
    jobs = csiborgtools.fits.split_jobs(parent_ids.size, nproc)[rank]
    _out_fits = numpy.full(len(jobs), numpy.nan, dtype=dtype)
    _out_map = {}
    for i in trange(len(jobs)) if verbose else range(len(jobs)):
        clid = parent_ids[jobs[i]]
        mmain_indxs = cat["index"][cat["parent"] == clid]

        mmain_mask = numpy.isin(clump_ids, mmain_indxs, assume_unique=True)
        mmain_particles = part0[mmain_mask, :]
        # If the number of particles is too small, we skip this halo.
        if mmain_particles.size < 100:
            continue

        raddist, cmpos = csiborgtools.match.dist_centmass(mmain_particles)
        patchsize = csiborgtools.match.dist_percentile(raddist, [99],
                                                       distmax=0.075)
        # Write the temporary results
        _out_fits["index"][i] = clid
        _out_fits["x"][i], _out_fits["y"][i], _out_fits["z"][i] = cmpos
        _out_fits["lagpatch"][i] = patchsize
        _out_map.update({str(clid): numpy.where(mmain_mask)[0]})

    # Dump the results of this rank to a temporary file.
    joblib.dump(_out_fits, ftemp("fits", nsim, rank))
    joblib.dump(_out_map, ftemp("map", nsim, rank))

    del part0, clump_ids,
    collect()

    # Now we wait for all ranks, then collect the results and save it.
    comm.Barrier()
    if rank == 0:
        print(f"{datetime.now()}: collecting results for {nsim}.", flush=True)
        out_fits = numpy.full(parent_ids.size, numpy.nan, dtype=dtype)
        out_map = {}
        for i in range(nproc):
            # Merge the map dictionaries
            out_map = out_map | joblib.load(ftemp("map", nsim, i))
            # Now merge the structured arrays
            _out_fits = joblib.load(ftemp("fits", nsim, i))
            for j in range(_out_fits.size):
                k = hid2arrpos[_out_fits["index"][j]]
                for par in dtype["names"]:
                    out_fits[par][k] = _out_fits[par][j]

            remove(ftemp("fits", nsim, i))
            remove(ftemp("map", nsim, i))

        # Now save it
        fout_fit = paths.initmatch_path(nsim, "fit")
        print(f"{datetime.now()}: dumping fits to .. `{fout_fit}`.",
              flush=True)
        with open(fout_fit, "wb") as f:
            numpy.save(f, out_fits)

        fout_map = paths.initmatch_path(nsim, "halomap")
        print(f"{datetime.now()}: dumping mapping to .. `{fout_map}`.",
              flush=True)
        with h5py.File(fout_map, "w") as f:
            for hid, indxs in out_map.items():
                f.create_dataset(hid, data=indxs)

        # We force clean up the memory before continuing.
        del out_map, out_fits
    collect()
