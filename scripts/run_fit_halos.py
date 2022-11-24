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
A script to fit halos (concentration, ...). The particle array of each CSiBORG
realisation must have been split in advance by `run_split_halos`.
"""

import numpy
from datetime import datetime
from os.path import join
from mpi4py import MPI
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools
import utils

F64 = numpy.float64
I64 = numpy.int64


# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


dumpdir = utils.dumpdir
loaddir = join(utils.dumpdir, "temp")
cols_collect = [("npart", I64), ("totpartmass", F64), ("Rs", F64),
                ("vx", F64), ("vy", F64), ("vz", F64),
                ("rho0", F64), ("conc", F64), ("rmin", F64),
                ("rmax", F64), ("r200", F64), ("r500", F64),
                ("m200", F64), ("m500", F64)]
paths = csiborgtools.read.CSiBORGPaths()

for i, n_sim in enumerate(paths.ic_ids):
    if rank == 0:
        print("{}: calculating {}th simulation.".format(datetime.now(), i))
    # Correctly set the paths!
    n_snap = paths.get_maximum_snapshot(n_sim)
    paths.set_info(n_sim, n_snap)

    box = csiborgtools.units.BoxUnits(paths)

    jobs = csiborgtools.fits.split_jobs(utils.n_splits, nproc)[rank]
    for n_split in jobs:
        parts, part_clumps, clumps = csiborgtools.fits.load_split_particles(
            n_split, paths, n_snap, remove_split=False)

        N = clumps.size
        cols = [("index", I64), ("npart", I64), ("totpartmass", F64),
                ("Rs", F64), ("rho0", F64), ("conc", F64),
                ("vx", F64), ("vy", F64), ("vz", F64),
                ("rmin", F64), ("rmax", F64),
                ("r200", F64), ("r500", F64), ("m200", F64), ("m500", F64)]
        out = csiborgtools.utils.cols_to_structured(N, cols)
        out["index"] = clumps["index"]

        for n in range(N):
            # Pick clump and its particles
            xs = csiborgtools.fits.pick_single_clump(n, parts, part_clumps,
                                                     clumps)
            clump = csiborgtools.fits.Clump.from_arrays(*xs, rhoc=box.box_rhoc)
            out["npart"][n] = clump.Npart
            out["rmin"][n] = clump.rmin
            out["rmax"][n] = clump.rmax
            out["totpartmass"][n] = clump.total_particle_mass
            out["vx"][n] = numpy.average(clump.vel[:, 0], weights=clump.m)
            out["vy"][n] = numpy.average(clump.vel[:, 1], weights=clump.m)
            out["vz"][n] = numpy.average(clump.vel[:, 2], weights=clump.m)

            # Spherical overdensity radii and masses
            rs, ms = clump.spherical_overdensity_mass([200, 500])
            out["r200"][n] = rs[0]
            out["r500"][n] = rs[1]
            out["m200"][n] = ms[0]
            out["m500"][n] = ms[1]

            # NFW profile fit
            if clump.Npart > 10 and numpy.isfinite(out["r200"][n]):
                nfwpost = csiborgtools.fits.NFWPosterior(clump)
                logRs, __ = nfwpost.maxpost_logRs()
                Rs = 10**logRs
                if not numpy.isnan(logRs):
                    out["Rs"][n] = Rs
                    out["rho0"][n] = nfwpost.rho0_from_Rs(Rs)
                    out["conc"][n] = out["r200"][n] / Rs

        csiborgtools.read.dump_split(out, n_split, paths)

    # Wait until all jobs finished before moving to another simulation
    comm.Barrier()

    # Use the rank 0 to combine outputs for this CSiBORG realisation
    if rank == 0:
        print("Collecting results!")
        partreader = csiborgtools.read.ParticleReader(paths)
        out_collected = csiborgtools.read.combine_splits(
            utils.n_splits, paths, cols_collect, remove_splits=True,
            verbose=False)
        fname = join(paths.dumpdir, "ramses_out_{}_{}.npy"
                     .format(str(paths.n_sim).zfill(5),
                             str(paths.n_snap).zfill(5)))
        print("Saving results to `{}`.".format(fname))
        numpy.save(fname, out_collected)

    comm.Barrier()

if rank == 0:
    print("All finished! See ya!")
