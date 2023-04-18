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
from datetime import datetime
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

paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
partreader = csiborgtools.read.ParticleReader(paths)
nfwpost = csiborgtools.fits.NFWPosterior()
ftemp = join(paths.temp_dumpdir, "fit_clump_{}_{}_{}.npy")
cols_collect = [
    ("npart", numpy.int64),
    ("totpartmass", numpy.float64),
    ("vx", numpy.float64),
    ("vy", numpy.float64),
    ("vz", numpy.float64),
    ("conc", numpy.float64),
    ("rho0", numpy.float64),
    ("r200", numpy.float64),
    ("r500", numpy.float64),
    ("m200", numpy.float64),
    ("m500", numpy.float64),
    ("lambda200", numpy.float64),
]


def fit_clump(particles, clump, box):
    obj = csiborgtools.fits.Clump(particles, clump, box)

    out = {}
    out["npart"] = len(obj)
    out["totpartmass"] = numpy.sum(obj["M"])
    out["vx"] = numpy.average(obj.vel[:, 0], weights=obj["M"])
    out["vy"] = numpy.average(obj.vel[:, 1], weights=obj["M"])
    out["vz"] = numpy.average(obj.vel[:, 2], weights=obj["M"])
    out["r200"], out["m200"] = obj.spherical_overdensity_mass(200)
    out["r500"], out["m500"] = obj.spherical_overdensity_mass(500)
    if out["npart"] > 10 and numpy.isfinite(out["r200"]):
        Rs, rho0 = nfwpost.fit(obj)
        out["conc"] = Rs / out["r200"]
        out["rho0"] = rho0
    if numpy.isfinite(out["r200"]):
        out["lambda200"] = obj.lambda_bullock(out["r200"])
    return out


for i, nsim in enumerate(paths.get_ics(tonew=False)):
    if rank == 0:
        print(
            "{}: calculating {}th simulation `{}`.".format(datetime.now(), i, nsim),
            flush=True,
        )
    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.BoxUnits(nsnap, nsim, paths)

    # Archive of clumps, keywords are their clump IDs
    particle_archive = numpy.load(paths.split_path(nsnap, nsim))
    nclumps = len(particle_archive.files)
    clumpsarr = partreader.read_clumps(nsnap, nsim, cols=["index", "x", "y", "z"])
    clumpid2arrpos = {ind: ii for ii, ind in enumerate(clumpsarr["index"])}

    # We split the clumps among the processes. Each CPU calculates a fraction
    # of them and dumps the results in a structured array.
    jobs = csiborgtools.fits.split_jobs(nclumps, nproc)
    out = csiborgtools.read.cols_to_structured(len(jobs[rank]), cols_collect)
    for i in tqdm(jobs[rank]) if nproc == 1 else jobs[rank]:
        clumpid = clumpsarr["index"][i]
        try:
            _out = fit_clump(
                particle_archive[str(clumpid)], clumpsarr[clumpid2arrpos[clumpid]], box
            )
        except KeyError:
            pass

        for key in _out.keys():
            out[key][i] = _out[key]

    fout = ftemp.format(str(nsim).zfill(5), str(nsnap).zfill(5), rank)
    print("{}: rank {} saving to `{}`.".format(datetime.now(), rank, fout), flush=True)
    numpy.save(fout, out)
    # We saved this CPU's results in a temporary file. Wait now for the other
    # CPUs and then collect results from the 0th rank and save them.
    comm.Barrier()

    if rank == 0:
        print(
            "{}: collecting results for simulation `{}`.".format(datetime.now(), nsim),
            flush=True,
        )
        out = csiborgtools.read.cols_to_structured(nclumps, cols_collect)
        k = 0
        for i in range(nproc):
            inp = numpy.load(ftemp.format(str(nsim).zfill(5), str(nsnap).zfill(5), i))
            for j in range(jobs[i]):
                for key in inp.dtype.names:
                    out[key][k] = inp[key][j]
                k += 1

        numpy.save(paths.structfit_path(nsnap, nsim, "clumps"), out)

    # We now wait before moving on to another simulation.
    comm.Barrier()
