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
from argparse import ArgumentParser
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

parser = ArgumentParser()
parser.add_argument("--kind", type=str, choices=["halos", "clumps"])
args = parser.parse_args()


# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
partreader = csiborgtools.read.ParticleReader(paths)
nfwpost = csiborgtools.fits.NFWPosterior()
ftemp = join(paths.temp_dumpdir, "fit_clump_{}_{}_{}.npy")
cols_collect = [
    ("npart", numpy.int32),
    ("totpartmass", numpy.float32),
    ("vx", numpy.float32),
    ("vy", numpy.float32),
    ("vz", numpy.float32),
    ("conc", numpy.float32),
    ("rho0", numpy.float32),
    ("r200c", numpy.float32),
    ("r500c", numpy.float32),
    ("m200c", numpy.float32),
    ("m500c", numpy.float32),
    ("lambda200c", numpy.float32),
    ("r200m", numpy.float32),
    ("m200m", numpy.float32),
]


def fit_clump(particles, clump, box):
    obj = csiborgtools.fits.Clump(particles, clump, box)

    out = {}
    out["npart"] = len(obj)
    out["totpartmass"] = numpy.sum(obj["M"])
    out["vx"] = numpy.average(obj.vel[:, 0], weights=obj["M"])
    out["vy"] = numpy.average(obj.vel[:, 1], weights=obj["M"])
    out["vz"] = numpy.average(obj.vel[:, 2], weights=obj["M"])
    out["r200c"], out["m200c"] = obj.spherical_overdensity_mass(200, kind="crit")
    out["r500c"], out["m500c"] = obj.spherical_overdensity_mass(500, kind="crit")
    if out["npart"] > 10 and numpy.isfinite(out["r200c"]):
        Rs, rho0 = nfwpost.fit(obj)
        out["conc"] = Rs / out["r200c"]
        out["rho0"] = rho0
    if numpy.isfinite(out["r200c"]):
        out["lambda200c"] = obj.lambda_bullock(out["r200c"])

    out["r200m"], out["m200m"] = obj.spherical_overdensity_mass(200, kind="matter")
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
    clumpsarr = partreader.read_clumps(nsnap, nsim, cols=["index", "x", "y", "z"])
    nclumps = clumpsarr.size

    # We split the clumps among the processes. Each CPU calculates a fraction
    # of them and dumps the results in a structured array.
    jobs = csiborgtools.fits.split_jobs(nclumps, nproc)
    out = csiborgtools.read.cols_to_structured(len(jobs[rank]), cols_collect)
    for i in tqdm(jobs[rank]) if nproc == 1 else jobs[rank]:
        clumpid = clumpsarr["index"][i]
        # We check whether this clump has some associated particles and then
        # fit it and store the results in `out`.
        try:
            part = particle_archive[str(clumpid)]
        except KeyError:
            part = None

        if part is not None:
            _out = fit_clump(part, clumpsarr[i], box)

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
        # We write to the output array. Load data from each CPU and append to
        # the output array.
        out = csiborgtools.read.cols_to_structured(nclumps, cols_collect)
        k = 0
        for i in range(nproc):
            inp = numpy.load(ftemp.format(str(nsim).zfill(5), str(nsnap).zfill(5), i))
            for j in jobs[i]:
                for key in inp.dtype.names:
                    out[key][k] = inp[key][j]
                k += 1

        fout = paths.structfit_path(nsnap, nsim, "clumps")
        print("Saving to `{}`.".format(fout), flush=True)
        numpy.save(fout, out)

    # We now wait before moving on to another simulation.
    comm.Barrier()
