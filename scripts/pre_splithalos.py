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
"""Script to split particles to indivudual files according to their clump."""
from datetime import datetime
from gc import collect
from glob import glob
from os import remove
from os.path import join

import numpy
from mpi4py import MPI
from TaskmasterMPI import master_process, worker_process
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

paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
verbose = nproc == 1
partcols = ['x', 'y', 'z', "vx", "vy", "vz", 'M']


def do_split(nsim):
    nsnap = max(paths.get_snapshots(nsim))
    reader = csiborgtools.read.ParticleReader(paths)
    ftemp_base = join(
        paths.temp_dumpdir,
        "split_{}_{}".format(str(nsim).zfill(5), str(nsnap).zfill(5))
        )
    ftemp = ftemp_base + "_{}.npz"

    # Load the particles and their clump IDs
    particles = reader.read_particle(nsnap, nsim, partcols, verbose=verbose)
    particle_clumps = reader.read_clumpid(nsnap, nsim, verbose=verbose)
    # Drop all particles whose clump index is 0 (not assigned to any clump)
    assigned_mask = particle_clumps != 0
    particle_clumps = particle_clumps[assigned_mask]
    particles = particles[assigned_mask]
    del assigned_mask
    collect()

    # Load the clump indices
    clumpinds = reader.read_clumps(nsnap, nsim, cols="index")["index"]
    # Some of the clumps have no particles, so we do not loop over them
    clumpinds = clumpinds[numpy.isin(clumpinds, particle_clumps)]

    # Loop over the clump indices and save the particles to a temporary file
    # every 10000 clumps. We will later read this back and combine into a
    # single file.
    out = {}
    for i, clind in enumerate(tqdm(clumpinds) if verbose else clumpinds):
        key = str(clind)
        out.update({str(clind): particles[particle_clumps == clind]})

        # REMOVE bump this back up
        if i % 10000 == 0 or i == clumpinds.size - 1:
            numpy.savez(ftemp.format(i), **out)
            out = {}

    # Clear up memory because we will be loading everything back
    del particles, particle_clumps, clumpinds
    collect()

    # Now load back in every temporary file, combine them into a single
    # dictionary  and save as a single .npz file.
    out = {}
    for file in glob(ftemp_base + '*'):
        inp = numpy.load(file)
        for key in inp.files:
            out.update({key: inp[key]})
        remove(file)

    numpy.savez(paths.split_path(nsnap, nsim), **out)


###############################################################################
#                             MPI task delegation                             #
###############################################################################


if nproc > 1:
    if rank == 0:
        tasks = list(paths.get_ics(tonew=False))
        master_process(tasks, comm, verbose=True)
    else:
        worker_process(do_split, comm, verbose=False)
else:
    tasks = paths.get_ics(tonew=False)
    tasks = [tasks[0]]  # REMOVE
    for task in tasks:
        print("{}: completing task `{}`.".format(datetime.now(), task))
        do_split(task)

comm.Barrier()