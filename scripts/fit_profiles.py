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
"""A script to calculate the particle's separation from the CM and save it."""
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

paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
partreader = csiborgtools.read.ParticleReader(paths)
cols_collect = [("r", numpy.float32), ("M", numpy.float32)]
if args.ics is None or args.ics == -1:
    nsims = paths.get_ics(tonew=False)
else:
    nsims = args.ics


def load_clump_particles(clumpid, particle_archive):
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
    # We first load the particles of each clump belonging to this
    # parent and then concatenate them for further analysis.
    clumps = []
    for ind in indxs:
        parts = load_clump_particles(ind, particle_archive)
        if parts is not None:
            clumps.append(parts)

    if len(clumps) == 0:
        return None
    return csiborgtools.match.concatenate_parts(clumps)


# We loop over simulations. Here later optionlaly add MPI.
for i, nsim in enumerate(nsims):
    if rank == 0:
        now = datetime.now()
        print(f"{now}: calculating {i}th simulation `{nsim}`.", flush=True)
    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.BoxUnits(nsnap, nsim, paths)

    # Archive of clumps, keywords are their clump IDs
    particle_archive = numpy.load(paths.split_path(nsnap, nsim))
    clumps_cat = csiborgtools.read.ClumpsCatalogue(nsim, paths, maxdist=None,
                                                   minmass=None, rawdata=True,
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

        parts = load_parent_particles(clumpid, particle_archive, clumps_cat)
        obj = csiborgtools.fits.Clump(parts, clumps_cat[j], box)
        r200m, m200m = obj.spherical_overdensity_mass(200, npart_min=10,
                                                      kind="matter")
        r = obj.r()
        mask = r <= r200m

        _out = csiborgtools.read.cols_to_structured(numpy.sum(mask),
                                                    cols_collect)

        _out["r"] = r[mask]
        _out["M"] = obj["M"][mask]

        out[str(clumps_cat["index"][i])] = _out

    # Finished, so we save everything.
    fout = paths.radpos_path(nsnap, nsim)
    now = datetime.now()
    print(f"{now}: saving radial profiles for simulation {nsim} to `{fout}`",
          flush=True)
    numpy.savez(fout, **out)

    # Clean up the memory just to be sure.
    del out
    collect()
