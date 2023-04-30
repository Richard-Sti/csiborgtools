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
    out["npart"] = len(obj)
    out["totpartmass"] = numpy.sum(obj["M"])
    for i, v in enumerate(["vx", "vy", "vz"]):
        out[v] = numpy.average(obj.vel[:, i], weights=obj["M"])
    # Overdensity masses
    out["r200c"], out["m200c"] = obj.spherical_overdensity_mass(200,
                                                                kind="crit")
    out["r500c"], out["m500c"] = obj.spherical_overdensity_mass(500,
                                                                kind="crit")
    out["r200m"], out["m200m"] = obj.spherical_overdensity_mass(200,
                                                                kind="matter")
    # NFW fit
    if out["npart"] > 10 and numpy.isfinite(out["r200c"]):
        Rs, rho0 = nfwpost.fit(obj)
        out["conc"] = Rs / out["r200c"]
        out["rho0"] = rho0
    # Spin within R200c
    if numpy.isfinite(out["r200c"]):
        out["lambda200c"] = obj.lambda_bullock(out["r200c"])
    return out


def load_clump_particles(clumpid, particles, clump_map):
    """
    Load a clump's particles. If it is not there, i.e clump has no associated
    particles, return `None`.
    """
    try:
        return particles[clump_map[clumpid], :]
    except KeyError:
        return None


def load_parent_particles(clumpid, particles, clump_map, clumps_cat):
    """
    Load a parent halo's particles.
    """
    indxs = clumps_cat["index"][clumps_cat["parent"] == clumpid]
    # We first load the particles of each clump belonging to this parent
    # and then concatenate them for further analysis.
    clumps = []
    for ind in indxs:
        parts = load_clump_particles(ind, particles, clump_map)
        if parts is not None:
            clumps.append(parts)

    if len(clumps) == 0:
        return None
    return numpy.concatenate(clumps)


# We now start looping over all simulations
for i, nsim in enumerate(paths.get_ics(tonew=False)):
    if rank == 0:
        print(f"{datetime.now()}: calculating {i}th simulation `{nsim}`.",
              flush=True)
    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.BoxUnits(nsnap, nsim, paths)

    # Particle archive
    particles = h5py.File(paths.particle_h5py_path(nsim), 'r')["particles"]
    clump_map = h5py.File(paths.particle_h5py_path(nsim, "clumpmap"), 'r')
    clumps_cat = csiborgtools.read.ClumpsCatalogue(nsim, paths, maxdist=None,
                                                   minmass=None, rawdata=True,
                                                   load_fitted=False)
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
    jobs = csiborgtools.fits.split_jobs(ntasks, nproc)[rank]
    out = csiborgtools.read.cols_to_structured(len(jobs), cols_collect)
    for i, j in enumerate(tqdm(jobs)) if nproc == 1 else enumerate(jobs):
        clumpid = clumps_cat["index"][j]
        out["index"][i] = clumpid

        # If we are fitting halos and this clump is not a main, then continue.
        if args.kind == "halos" and not ismain[j]:
            continue

        if args.kind == "halos":
            part = load_parent_particles(clumpid, particles, clump_map,
                                         clumps_cat)
        else:
            part = load_clump_particles(clumpid, particles, clump_map)

        # We fit the particles if there are any. If not we assign the index,
        # otherwise it would be NaN converted to integers (-2147483648) and
        # yield an error further down.
        if part is not None:
            _out = fit_clump(part, clumps_cat[j], box)
            for key in _out.keys():
                out[key][i] = _out[key]

    fout = ftemp.format(str(nsim).zfill(5), str(nsnap).zfill(5), rank)
    if nproc == 0:
        print(f"{datetime.now()}: rank {rank} saving to `{fout}`.", flush=True)
    numpy.save(fout, out)
    # We saved this CPU's results in a temporary file. Wait now for the other
    # CPUs and then collect results from the 0th rank and save them.
    comm.Barrier()

    if rank == 0:
        print(f"{datetime.now()}: collecting results for simulation `{nsim}`.",
              flush=True)
        # We write to the output array. Load data from each CPU and append to
        # the output array.
        out = csiborgtools.read.cols_to_structured(ntasks, cols_collect)
        clumpid2outpos = {indx: i
                          for i, indx in enumerate(clumps_cat["index"])}
        for i in range(nproc):
            inp = numpy.load(ftemp.format(str(nsim).zfill(5),
                                          str(nsnap).zfill(5), i))
            for j, clumpid in enumerate(inp["index"]):
                k = clumpid2outpos[clumpid]
                for key in inp.dtype.names:
                    out[key][k] = inp[key][j]

        # If we were analysing main halos, then remove array indices that do
        # not correspond to parent halos.
        if args.kind == "halos":
            out = out[ismain]

        fout = paths.structfit_path(nsnap, nsim, args.kind)
        print(f"Saving to `{fout}`.", flush=True)
        numpy.save(fout, out)

    # We now wait before moving on to another simulation.
    comm.Barrier()
