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
IC realisations. Units are Mpc/h.
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
parser.add_argument("--halfwidth", type=float, default=0.5)
args = parser.parse_args()

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()
MAS = "CIC"  # mass asignment scheme

paths = csiborgtools.read.CSiBORGPaths()
ics = paths.ic_ids
n_sims = len(ics)

# File paths
ftemp = join(utils.dumpdir, "temp_crosspk",
             "out_{}_{}" + "_{}".format(args.halfwidth))
fout = join(utils.dumpdir, "crosspk",
            "out_{}_{}" + "_{}.p".format(args.halfwidth))


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
    # Halfwidth -- particle selection
    if args.halfwidth < 0.5:
        hw = args.halfwidth
        mask = ((0.5 - hw < particles['x']) & (particles['x'] < 0.5 + hw)
                & (0.5 - hw < particles['y']) & (particles['y'] < 0.5 + hw)
                & (0.5 - hw < particles['z']) & (particles['z'] < 0.5 + hw))
        # Subselect the particles
        particles = particles[mask]
        # Rescale to range [0, 1]
        for p in ('x', 'y', 'z'):
            particles[p] = (particles[p] - 0.5 + hw) / (2 * hw)

        length = box.box2mpc(2 * hw) * box.h  # Mpc/h
    else:
        mask = None
        length = box.box2mpc(1) * box.h  # Mpc/h
    # Calculate the overdensity field
    field = csiborgtools.field.DensityField(particles, length, box, MAS)
    delta = field.overdensity_field(args.grid, verbose=False)
    aexp = box._aexp

    # Try to clean up memory
    del field, particles, box, reader, mask
    collect()

    # Dump the results
    with open(ftemp.format(n_sim, "delta") + ".npy", "wb") as f:
        numpy.save(f, delta)
    joblib.dump([aexp, length], ftemp.format(n_sim, "lengths") + ".p")

    # Try to clean up memory
    del delta
    collect()


comm.Barrier()

# Get off-diagonal elements and append the diagoal
combs = [c for c in combinations(range(n_sims), 2)]
for i in range(n_sims):
    combs.append((i, i))
prev_delta = [-1, None, None, None]  # i, delta, aexp, length

jobs = csiborgtools.fits.split_jobs(len(combs), nproc)[rank]
for n in jobs:
    i, j = combs[n]
    print("Rank {}@{}: combination {}.".format(rank, datetime.now(), (i, j)))

    # If i same as last time then don't have to load it
    if prev_delta[0] == i:
        delta_i = prev_delta[1]
        aexp_i = prev_delta[2]
        length_i = prev_delta[3]
    else:
        with open(ftemp.format(ics[i], "delta") + ".npy", "rb") as f:
            delta_i = numpy.load(f)
        aexp_i, length_i = joblib.load(ftemp.format(ics[i], "lengths") + ".p")
        # Store in prev_delta
        prev_delta[0] = i
        prev_delta[1] = delta_i
        prev_delta[2] = aexp_i
        prev_delta[3] = length_i

    # Get jth delta
    with open(ftemp.format(ics[j], "delta") + ".npy", "rb") as f:
        delta_j = numpy.load(f)
    aexp_j, length_j = joblib.load(ftemp.format(ics[j], "lengths") + ".p")

    # Verify the difference between the scale factors! Say more than 1%
    daexp = abs((aexp_i - aexp_j) / aexp_i)
    if daexp > 0.01:
        raise ValueError(
            "Boxes {} and {} final snapshot scale factors disagree by "
            "`{}` percent!".format(ics[i], ics[j], daexp * 100))
    # Check how well the boxsizes agree
    dlength = abs((length_i - length_j) / length_i)
    if dlength > 0.001:
        raise ValueError("Boxes {} and {} box sizes disagree by `{}` percent!"
                         .format(ics[i], ics[j], dlength * 100))

    # Calculate the cross power spectrum
    Pk = PKL.XPk([delta_i, delta_j], length_i, axis=1, MAS=[MAS, MAS],
                 threads=1)
    joblib.dump(Pk, fout.format(ics[i], ics[j]))

    del delta_i, delta_j, Pk
    collect()


# Clean up the temp files
comm.Barrier()
if rank == 0:
    print("Cleaning up the temporary files...")
    for ic in ics:
        remove(ftemp.format(ic, "delta") + ".npy")
        remove(ftemp.format(ic, "lengths") + ".p")

    print("All finished!")
