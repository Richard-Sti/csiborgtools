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
Script to load in the simulation particles and dump them to a HDF5 file for the
SPH density field calculation.
"""

from datetime import datetime
from distutils.util import strtobool
from gc import collect

import h5py
import numpy
from mpi4py import MPI
from tqdm import tqdm

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools

from argparse import ArgumentParser

# We set up the MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

# And next parse all the arguments and set up CSiBORG objects
parser = ArgumentParser()
parser.add_argument("--ics", type=int, nargs="+", default=None,
                    help="IC realisatiosn. If `-1` processes all simulations.")
parser.add_argument("--pos_only", type=lambda x: bool(strtobool(x)),
                    help="Do we only dump positions?")
args = parser.parse_args()

verbose = nproc == 1
paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
partreader = csiborgtools.read.ParticleReader(paths)

if args.pos_only:
    pars_extract = ['x', 'y', 'z', 'M']
else:
    pars_extract = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'M']

if args.ics is None or args.ics == -1:
    ics = paths.get_ics(tonew=False)
else:
    ics = args.ics

# MPI loop over individual simulations. We read in the particles from RAMSES
# files and dump them to a HDF5 file.
jobs = csiborgtools.fits.split_jobs(len(ics), nproc)[rank]
for i in jobs:
    nsim = ics[i]
    nsnap = max(paths.get_snapshots(nsim))
    print(f"{datetime.now()}: Rank {rank} loading particles {nsim}.",
          flush=True)

    parts = partreader.read_particle(nsnap, nsim, pars_extract,
                                     return_structured=False, verbose=verbose)
    kind = "pos" if args.pos_only else None

    print(f"{datetime.now()}: Rank {rank} dumping particles from {nsim}.",
          flush=True)

    with h5py.File(paths.particle_h5py_path(nsim, kind), "w") as f:
        f.create_dataset("particles", data=parts)
    del parts
    collect()
    # If we are dumping only particle positions, then we are done.
    if args.pos_only:
        continue

    print(f"{datetime.now()}: Rank {rank} mapping particles from {nsim}.",
          flush=True)
    # If not, then load the clump IDs and prepare the memory mapping. We find
    # which array positions correspond to which clump IDs and save it. With
    # this we can then lazily load into memory the particles for each clump.
    part_cids = partreader.read_clumpid(nsnap, nsim, verbose=verbose)
    cat = csiborgtools.read.ClumpsCatalogue(nsim, paths, load_fitted=False,
                                            rawdata=True)
    clumpinds = cat["index"]
    # Some of the clumps have no particles, so we do not loop over them
    clumpinds = clumpinds[numpy.isin(clumpinds, part_cids)]

    for i, cid in enumerate(tqdm(clumpinds) if verbose else clumpinds):
        key = str(cid)

        indxs = numpy.where(part_cids == cid)[0]
        with h5py.File(paths.particle_h5py_path(nsim, "clumpmap"), "w") as f:
            f.create_dataset(str(cid), data=indxs)

    del part_cids, cat, clumpinds
    collect()
