# Copyright (C) 2023 Richard Stiskalek
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
A script to calculate the particle's separation from the CM and save it.
Currently MPI is not supported.
"""
from argparse import ArgumentParser
from datetime import datetime
from gc import collect

import numpy
from mpi4py import MPI
from tqdm import trange

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools

parser = ArgumentParser()
parser.add_argument("--ics", type=int, nargs="+", default=None,
                    help="IC realisatiosn. If `-1` processes all simulations.")
args = parser.parse_args()

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

if nproc > 1:
    raise NotImplementedError("MPI is not implemented implemented yet.")

paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
cols_collect = [("r", numpy.float32), ("M", numpy.float32)]
if args.ics is None or args.ics == -1:
    nsims = paths.get_ics()
else:
    nsims = args.ics


# We loop over simulations. Here later optionally add MPI.
for i, nsim in enumerate(nsims):
    if rank == 0:
        now = datetime.now()
        print(f"{now}: calculating {i}th simulation `{nsim}`.", flush=True)
    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)

    f = csiborgtools.read.read_h5(paths.particles_path(nsim))
    particles = f["particles"]
    clump_map = f["clumpmap"]
    clid2map = {clid: i for i, clid in enumerate(clump_map[:, 0])}
    clumps_cat = csiborgtools.read.ClumpsCatalogue(nsim, paths, rawdata=True,
                                                   load_fitted=False)
    ismain = clumps_cat.ismain
    ntasks = len(clumps_cat)

    # We loop over halos and add ther particle positions to this dictionary,
    # which we will later save as an archive.
    out = {}
    for j in trange(ntasks) if nproc == 1 else range(ntasks):
        # If we are fitting halos and this clump is not a main, then continue.
        if not ismain[j]:
            continue

        clumpid = clumps_cat["index"][j]
        parts = csiborgtools.read.load_parent_particles(
            clumpid, particles, clump_map, clid2map, clumps_cat)
        # If we have no particles, then do not save anything.
        if parts is None:
            continue
        obj = csiborgtools.fits.Clump(parts, clumps_cat[j], box)
        r200m, m200m = obj.spherical_overdensity_mass(200, npart_min=10,
                                                      kind="matter")
        r = obj.r()
        mask = r <= r200m

        _out = csiborgtools.read.cols_to_structured(numpy.sum(mask),
                                                    cols_collect)

        _out["r"] = r[mask]
        _out["M"] = obj["M"][mask]
        out[str(clumpid)] = _out

    # Finished, so we save everything.
    fout = paths.radpos_path(nsnap, nsim)
    now = datetime.now()
    print(f"{now}: saving radial profiles for simulation {nsim} to `{fout}`",
          flush=True)
    numpy.savez(fout, **out)

    # Clean up the memory just to be sure.
    del out
    collect()
