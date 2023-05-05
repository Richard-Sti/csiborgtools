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
verbose = nproc == 1

parser = ArgumentParser()
parser.add_argument("--kind", type=str, choices=["halos", "clumps"])
parser.add_argument("--ics", type=int, nargs="+", default=None,
                    help="IC realisations. If `-1` processes all simulations.")
args = parser.parse_args()
paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
partreader = csiborgtools.read.ParticleReader(paths)
nfwpost = csiborgtools.fits.NFWPosterior()

if args.ics is None or args.ics[0] == -1:
    ics = paths.get_ics(tonew=False)
else:
    ics = args.ics

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


# We MPI loop over all simulations.
jobs = csiborgtools.fits.split_jobs(len(ics), nproc)[rank]
for nsim in [ics[i] for i in jobs]:
    print(f"{datetime.now()}: rank {rank} calculating simulation `{nsim}`.",
          flush=True)
    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.BoxUnits(nsnap, nsim, paths)

    # Particle archive
    f = csiborgtools.read.read_h5(paths.particles_path(nsim))
    particles = f["particles"]
    clump_map = f["clumpmap"]
    clid2map = {clid: i for i, clid in enumerate(clump_map[:, 0])}
    clumps_cat = csiborgtools.read.ClumpsCatalogue(nsim, paths, rawdata=True,
                                                   load_fitted=False)
    # We check whether we fit halos or clumps, will be indexing over different
    # iterators.
    if args.kind == "halos":
        ismain = clumps_cat.ismain
    else:
        ismain = numpy.ones(len(clumps_cat), dtype=bool)

    # Even if we are calculating parent halo this index runs over all clumps.
    out = csiborgtools.read.cols_to_structured(len(clumps_cat), cols_collect)
    indxs = clumps_cat["index"]
    for i, clid in enumerate(tqdm(indxs)) if verbose else enumerate(indxs):
        clid = clumps_cat["index"][i]
        out["index"][i] = clid
        # If we are fitting halos and this clump is not a main, then continue.
        if args.kind == "halos" and not ismain[i]:
            continue

        if args.kind == "halos":
            part = csiborgtools.read.load_parent_particles(
                clid, particles, clump_map, clid2map, clumps_cat)
        else:
            part = csiborgtools.read.load_clump_particles(clid, particles,
                                                          clump_map, clid2map)

        # We fit the particles if there are any. If not we assign the index,
        # otherwise it would be NaN converted to integers (-2147483648) and
        # yield an error further down.
        if part is None:
            continue

        _out = fit_clump(part, clumps_cat[i], box)
        for key in _out.keys():
            out[key][i] = _out[key]

    # Finally, we save the results. If we were analysing main halos, then
    # remove array indices that do not correspond to parent halos.
    if args.kind == "halos":
        out = out[ismain]

    fout = paths.structfit_path(nsnap, nsim, args.kind)
    print(f"Saving to `{fout}`.", flush=True)
    numpy.save(fout, out)
