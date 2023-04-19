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
    ("index", numpy.int32),
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


def fit_clump(particles, clump_info, box):
    """
    Fit an object. Can be eithe a clump or a parent halo.
    """
    obj = csiborgtools.fits.Clump(particles, clump_info, box)

    out = {}
    out["index"] = clump_info["index"]
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


def load_clump_particles(clumpid, particle_archive, clumps_cat):
    """
    Load a clump's particles from the particle archive. If it is not there, i.e
    clump has no associated particles, return `None`.
    """
    try:
        part = particle_archive[str(clumpid)]
    except KeyError:
        part = None
    return part


def load_parent_particles(clumpid, particle_archive, clumps_cat):
    """
    Load a parent halo's particles.
    """
    indxs = clumps_cat["index"][clumps_cat["parent"] == clumpid]

    # We first load the particles of each clump belonging to this parent and then
    # concatenate them for further analysis.
    clumps = []
    for ind in indxs:
        parts = load_clump_particles(ind, particle_archive, clumps_cat)
        if parts is not None:
            clumps.append([parts, None])

    if len(clumps) == 0:
        return None
    return csiborgtools.match.concatenate_clumps(clumps, include_velocities=True)


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
    clumps_cat = csiborgtools.read.ClumpsCatalogue(
        nsim, paths, maxdist=None, minmass=None
    )
    # We check whether we fit halos or clumps, will be indexing over different
    # iterators.
    if args.kind == "halos":
        ismain = clumps_cat.ismain
    else:
        ismain = numpy.ones(len(clumps_cat), dtype=bool)
    ntasks = len(clumps_cat)
    # We split the clumps among the processes. Each CPU calculates a fraction
    # of them and dumps the results in a structured array. Even if we are
    # calculating parent halo this index runs over all clumps.
    jobs = csiborgtools.fits.split_jobs(ntasks, nproc)
    out = csiborgtools.read.cols_to_structured(len(jobs[rank]), cols_collect)
    for i in tqdm(jobs[rank]) if nproc == 1 else jobs[rank]:
        # If we are fitting halos and this clump is not a main, then continue.
        if args.kind == "halos" and not ismain[i]:
            continue

        clumpid = clumps_cat["index"][i]
        if args.kind == "halos":
            part = load_parent_particles(clumpid, particle_archive, clumps_cat)
        else:
            part = load_clump_particles(clumpid, particle_archive, clumps_cat)

        if part is not None:
            _out = fit_clump(part, clumps_cat[i], box)

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
        out = csiborgtools.read.cols_to_structured(ntasks, cols_collect)
        k = 0
        for i in range(nproc):
            inp = numpy.load(ftemp.format(str(nsim).zfill(5), str(nsnap).zfill(5), i))
            for j in jobs[i]:
                for key in inp.dtype.names:
                    out[key][k] = inp[key][j]
                k += 1

        # If we were analysing main halos, then remove array indices that do
        # not correspond to parent halos.
        if args.kind == "halos":
            out = out[ismain]

        fout = paths.structfit_path(nsnap, nsim, "clumps")
        print("Saving to `{}`.".format(fout), flush=True)
        numpy.save(fout, out)

    # We now wait before moving on to another simulation.
    comm.Barrier()
