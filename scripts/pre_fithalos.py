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
realisation must have been split in advance by `runsplit_halos`.
"""
from os.path import join
from datetime import datetime
import numpy
from mpi4py import MPI
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools
import utils


# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
dumpdir = "/mnt/extraspace/rstiskalek/csiborg/"
loaddir = join(dumpdir, "temp")
cols_collect = [("npart", numpy.int64), ("totpartmass", numpy.float64),
                ("Rs", numpy.float64), ("vx", numpy.float64),
                ("vy", numpy.float64), ("vz", numpy.float64),
                ("Lx", numpy.float64), ("Ly", numpy.float64),
                ("Lz", numpy.float64), ("rho0", numpy.float64),
                ("conc", numpy.float64), ("rmin", numpy.float64),
                ("rmax", numpy.float64), ("r200", numpy.float64),
                ("r500", numpy.float64), ("m200", numpy.float64),
                ("m500", numpy.float64), ("lambda200c", numpy.float64)]


for i, nsim in enumerate(paths.ic_ids(tonew=False)):
    if rank == 0:
        print("{}: calculating {}th simulation.".format(datetime.now(), i))
    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.units.BoxUnits(nsnap, nsim, paths)

    jobs = csiborgtools.fits.split_jobs(utils.Nsplits, nproc)[rank]
    for nsplit in jobs:
        parts, part_clumps, clumps = csiborgtools.fits.load_split_particles(
            nsplit, nsnap, nsim, paths, remove_split=False)

        N = clumps.size
        cols = [("index", numpy.int64), ("npart", numpy.int64),
                ("totpartmass", numpy.float64), ("Rs", numpy.float64),
                ("rho0", numpy.float64), ("conc", numpy.float64),
                ("lambda200c", numpy.float64), ("vx", numpy.float64),
                ("vy", numpy.float64), ("vz", numpy.float64),
                ("Lx", numpy.float64), ("Ly", numpy.float64),
                ("Lz", numpy.float64), ("rmin", numpy.float64),
                ("rmax", numpy.float64), ("r200", numpy.float64),
                ("r500", numpy.float64), ("m200", numpy.float64),
                ("m500", numpy.float64)]
        out = csiborgtools.utils.cols_to_structured(N, cols)
        out["index"] = clumps["index"]

        for n in range(N):
            # Pick clump and its particles
            xs = csiborgtools.fits.pick_single_clump(n, parts, part_clumps,
                                                     clumps)
            clump = csiborgtools.fits.Clump.from_arrays(
                *xs, rhoc=box.box_rhoc, G=box.box_G)
            out["npart"][n] = clump.Npart
            out["rmin"][n] = clump.rmin
            out["rmax"][n] = clump.rmax
            out["totpartmass"][n] = clump.total_particle_mass
            out["vx"][n] = numpy.average(clump.vel[:, 0], weights=clump.m)
            out["vy"][n] = numpy.average(clump.vel[:, 1], weights=clump.m)
            out["vz"][n] = numpy.average(clump.vel[:, 2], weights=clump.m)
            out["Lx"][n], out["Ly"][n], out["Lz"][n] = clump.angular_momentum

            # Spherical overdensity radii and masses
            rs, ms = clump.spherical_overdensity_mass([200, 500])
            out["r200"][n] = rs[0]
            out["r500"][n] = rs[1]
            out["m200"][n] = ms[0]
            out["m500"][n] = ms[1]
            out["lambda200c"][n] = clump.lambda200c

            # NFW profile fit
            if clump.Npart > 10 and numpy.isfinite(out["r200"][n]):
                nfwpost = csiborgtools.fits.NFWPosterior(clump)
                logRs, __ = nfwpost.maxpost_logRs()
                Rs = 10**logRs
                if not numpy.isnan(logRs):
                    out["Rs"][n] = Rs
                    out["rho0"][n] = nfwpost.rho0_from_Rs(Rs)
                    out["conc"][n] = out["r200"][n] / Rs

        csiborgtools.read.dump_split(out, nsplit, nsnap, nsim, paths)

    # Wait until all jobs finished before moving to another simulation
    comm.Barrier()

    # Use the rank 0 to combine outputs for this CSiBORG realisation
    if rank == 0:
        print("Collecting results!")
        partreader = csiborgtools.read.ParticleReader(paths)
        out_collected = csiborgtools.read.combine_splits(
            utils.Nsplits, nsnap, nsim, partreader, cols_collect,
            remove_splits=True, verbose=False)
        fname = paths.hcat_path(nsim)
        print("Saving results to `{}`.".format(fname))
        numpy.save(fname, out_collected)

    comm.Barrier()

if rank == 0:
    print("All finished! See ya!")
