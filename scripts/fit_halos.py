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
A script to fit FoF halos (concentration, ...). The particle array of each
CSiBORG realisation must have been processed in advance by `pre_dumppart.py`.
"""
from argparse import ArgumentParser
from datetime import datetime

import numpy
from mpi4py import MPI
from tqdm import trange

from utils import get_nsims

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
parser.add_argument("--nsims", type=int, nargs="+", default=None,
                    help="IC realisations. If `-1` processes all simulations.")
args = parser.parse_args()
paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
partreader = csiborgtools.read.ParticleReader(paths)
nfwpost = csiborgtools.fits.NFWPosterior()
nsims = get_nsims(args, paths)

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
    ("r500m", numpy.float32),
    ("m500m", numpy.float32),
    ]


def fit_halo(particles, clump_info, box):
    obj = csiborgtools.fits.Clump(particles, clump_info, box)

    out = {}
    out["npart"] = len(obj)
    out["totpartmass"] = numpy.sum(obj["M"])
    for i, v in enumerate(["vx", "vy", "vz"]):
        out[v] = numpy.average(obj.vel[:, i], weights=obj["M"])
    # Overdensity masses
    for n in [200, 500]:
        out[f"r{n}c"], out[f"m{n}c"] = obj.spherical_overdensity_mass(
            n, kind="crit", npart_min=10)
        out[f"r{n}m"], out[f"m{n}m"] = obj.spherical_overdensity_mass(
            n, kind="matter", npart_min=10)
    # NFW fit
    if out["npart"] > 10 and numpy.isfinite(out["r200c"]):
        Rs, rho0 = nfwpost.fit(obj)
        out["conc"] = out["r200c"] / Rs
        out["rho0"] = rho0
    # Spin within R200c
    if numpy.isfinite(out["r200c"]):
        out["lambda200c"] = obj.lambda_bullock(out["r200c"])
    return out


# We MPI loop over all simulations.
jobs = csiborgtools.fits.split_jobs(len(nsims), nproc)[rank]
for nsim in [nsims[i] for i in jobs]:
    print(f"{datetime.now()}: rank {rank} calculating simulation `{nsim}`.",
          flush=True)
    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)

    # Particle archive
    f = csiborgtools.read.read_h5(paths.particles(nsim))
    particles = f["particles"]
    halo_map = f["halomap"]
    hid2map = {clid: i for i, clid in enumerate(halo_map[:, 0])}
    cat = csiborgtools.read.CSiBORGHaloCatalogue(
        nsim, paths, with_lagpatch=False, load_initial=False, rawdata=True,
        load_fitted=False)
    # Even if we are calculating parent halo this index runs over all clumps.
    out = csiborgtools.read.cols_to_structured(len(cat), cols_collect)
    indxs = cat["index"]
    for i in trange(len(cat)) if verbose else range(len(cat)):
        hid = cat["index"][i]
        out["index"][i] = hid

        part = csiborgtools.read.load_halo_particles(hid, particles, halo_map,
                                                     hid2map)
        # We fit the particles if there are any. If not we assign the index,
        # otherwise it would be NaN converted to integers (-2147483648) and
        # yield an error further down.
        if part is None:
            continue

        _out = fit_halo(part, cat[i], box)
        for key in _out.keys():
            out[key][i] = _out[key]

    fout = paths.structfit(nsnap, nsim)
    print(f"Saving to `{fout}`.", flush=True)
    numpy.save(fout, out)
