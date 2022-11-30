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
MPI script to calculate the matter cross power spectrum between CSiBORG
IC realisations.
"""
from argparse import ArgumentParser
import numpy
import joblib
from datetime import datetime
from itertools import combinations
from os.path import join
from os import remove
from gc import collect
from mpi4py import MPI
import Pk_library as PKL
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools
import utils


parser = ArgumentParser()
parser.add_argument("--grid", type=int)
args = parser.parse_args()

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

paths = csiborgtools.read.CSiBORGPaths()
ics = paths.ic_ids
n_sims = len(ics)

# File paths
ftemp = join(utils.dumpdir, "temp_crosspk", "out_{}_{}")
fout = join(utils.dumpdir, "crosspk", "out_{}_{}.p")


jobs = csiborgtools.fits.split_jobs(n_sims, nproc)[rank]
for n in jobs:
    print("Rank {}@{}: saving {}th delta.".format(rank, datetime.now(), n))
    # Set the paths
    n_sim = ics[n]
    paths.set_info(n_sim, paths.get_maximum_snapshot(n_sim))
    # Set reader and the box
    reader = csiborgtools.read.ParticleReader(paths)
    box = csiborgtools.units.BoxUnits(paths)
    # Read particles
    particles = reader.read_particle(["x", "y", "z", "M"], verbose=False)
    # Calculate the overdensity field
    field = csiborgtools.field.DensityField(particles, box)
    delta = field.overdensity_field(args.grid, verbose=False)
    # Dump the results
    numpy.save(ftemp.format(n_sim, "delta") + ".npy", delta)
    joblib.dump(box._aexp, ftemp.format(n_sim, "Om0") + ".p")


# Try to clean up memory and wait for all processes to finish
del particles, delta
collect()
comm.Barrier()

# Get off-diagonal elements and append the diagoal
combs = [c for c in combinations(range(n_sims), 2)]
for i in range(n_sims):
    combs.append((i, i))
prev_delta = [-1, None, None]  # i, delta, aexp

jobs = csiborgtools.fits.split_jobs(len(combs), nproc)[rank]
for n in jobs:
    i, j = combs[n]
    print("Rank {}@{}: combination {}.".format(rank, datetime.now(), (i, j)))

    # If i same as last time then don't have to load it
    if prev_delta[0] == i:
        delta_i = prev_delta[1]
        aexp_i = prev_delta[2]
    else:
        delta_i = numpy.load(ftemp.format(ics[i], "delta") + ".npy")
        aexp_i = joblib.load(ftemp.format(ics[i], "Om0") + ".p")
        # Store in prev_delta
        prev_delta[0] = i
        prev_delta[1] = delta_i
        prev_delta[2] = aexp_i

    # Get jth delta
    delta_j = numpy.load(ftemp.format(ics[j], "delta") + ".npy")
    aexp_j = joblib.load(ftemp.format(ics[j], "Om0") + ".p")

    # Verify the difference between the scale factors! Say more than 1%
    daexp = abs((aexp_i - aexp_j) / aexp_i)
    if daexp > 0.01:
        raise ValueError("Boxes {} and {} final snapshot scale factors "
                         "disagree by `{}`!".format(ics[i], ics[j], daexp))

    # Calculate the cross power spectrum
    Pk = PKL.XPk([delta_i, delta_j], 1., axis=1, MAS=["CIC", "CIC"], threads=1)
    joblib.dump(Pk, fout.format(ics[i], ics[j]))


# Clean up the temp files
comm.Barrier()
if rank == 0:
    print("Cleaning up the temporary files...")
    for ic in ics:
        remove(ftemp.format(ic, "delta") + ".npy")
        remove(ftemp.format(ic, "Om0") + ".p")

    print("All finished!")
