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
from datetime import datetime
from gc import collect

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

if args.ics is None or args.ics == -1:
    ics = paths.get_ics(tonew=True)
else:
    ics = args.ics

# We MPI loop over simulations. Particle matching within a simulation then
# performed on a single core.
jobs = csiborgtools.fits.split_jobs(len(ics), nproc)[rank]
for i in jobs:
    nsim = ics[i]
    nsnap = max(paths.get_snapshots(nsim))
    print(f"{datetime.now()}: Rank {rank} reading simulation {nsim}.",
          flush=True)

    # We first load particles in the initial and final snapshots and sort them
    # by their particle IDs so that we can match them by array position.
    # `clump_ids` are the clump IDs of particles.
    part0 = partreader.read_particle(1, nsim, ["x", "y", "z", "M", "ID"],
                                     verbose=verbose, return_structured=False)
    part0 = part0[numpy.argsort(part0[:, -1])]
    part0 = part0[:, :-1]  # Now we no longer need the particle IDs

    pid = partreader.read_particle(nsnap, nsim, ["ID"], verbose=verbose,
                                   return_structured=False).reshape(-1, )
    clump_ids = partreader.read_clumpid(nsnap, nsim, verbose=verbose)
    clump_ids = clump_ids[numpy.argsort(pid)]
    # Release the particle IDs, we will not need them anymore now that both
    # particle arrays are matched in ordering.
    del pid
    collect()

    # Particles whose clump ID is 0 are unassigned to a clump, so we can get
    # rid of them to speed up subsequent operations. We will not need these.
    # Again we release the mask.
    mask = clump_ids > 0
    clump_ids = clump_ids[mask]
    part0 = part0[mask, :]
    del mask
    collect()

    print(f"{datetime.now()}: rank {rank} dumping particles for {nsim}.",
          flush=True)
    # We already now save the initial snapshot particles.
    with h5py.File(paths.initmatch_path(nsim, "particles"), "w") as f:
        f.create_dataset("particles", data=part0)

    # Calculate the centre of mass of each parent halo, the Lagrangian patch
    # size and optionally the initial snapshot particles belonging to this
    # parent halo. Dumping the particles will take majority of time.
    print(f"{datetime.now()}: rank {rank} calculating simulation {nsim}.",
          flush=True)
    # We load up the clump catalogue which contains information about the
    # ultimate  parent halos of each clump. We will loop only over the clump
    # IDs of ultimate parent halos and add their substructure particles and at
    # the end save these.
    cat = csiborgtools.read.ClumpsCatalogue(nsim, paths, load_fitted=False,
                                            rawdata=True)
    parent_ids = cat["index"][cat.ismain]
    # And we pre-allocate the output array for this simulation.
    dtype = {"names": ["index", "x", "y", "z", "lagpatch"],
             "formats": [numpy.int32] + [numpy.float32] * 4}
    out_fits = numpy.full(parent_ids.size, numpy.nan, dtype=dtype)
    out_map = {}
    niters = parent_ids.size
    for i in trange(niters) if verbose else range(niters):
        clid = parent_ids[i]
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
        out_fits["index"][i] = clid
        out_fits["x"][i], out_fits["y"][i], out_fits["z"][i] = cmpos
        out_fits["lagpatch"][i] = patchsize

        out_map.update({str(clid): numpy.where(mmain_mask)[0]})

    # We now save the results for this simulation.
    fout_fit = paths.initmatch_path(nsim, "fit")
    print(f"{datetime.now()}: rank {rank} dumping fits to .. `{fout_fit}`.",
          flush=True)
    with open(fout_fit, "wb") as f:
        numpy.save(f, out_fits)

    fout_map = paths.initmatch_path(nsim, "halomap")
    print(f"{datetime.now()}: rank {rank} dumping mapping to .. `{fout_map}`.",
          flush=True)
    with h5py.File(fout_map, "w") as f:
        for hid, indxs in out_map.items():
            f.create_dataset(hid, data=indxs)

    # We force clean up the memory before continuing.
    del part0, clump_ids, out_map, out_fits
    collect()
