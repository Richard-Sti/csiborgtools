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
from gc import collect
from distutils.util import strtobool

import h5py
from mpi4py import MPI

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
parser.add_argument("--with_vel", type=lambda x: bool(strtobool(x)),
                    help="Whether to include velocities in the particle file.")
args = parser.parse_args()
paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
partreader = csiborgtools.read.ParticleReader(paths)
if args.with_vel:
    pars_extract = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'M']
else:
    pars_extract = ['x', 'y', 'z', 'M']
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
    print(f"{datetime.now()}: Rank {rank} completing simulation {nsim}.",
          flush=True)

    out = partreader.read_particle(
        nsnap, nsim, pars_extract, return_structured=False, verbose=nproc == 1)

    with h5py.File(paths.particle_h5py_path(nsim), "w") as f:
        dset = f.create_dataset("particles", data=out)

    del out
    collect()
