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
A script to fit FoF halos (concentration, ...). The CSiBORG particle array of
each realisation must have been processed in advance by `pre_dumppart.py`.
Quijote is not supported yet
"""
from argparse import ArgumentParser

import numpy
from mpi4py import MPI
from taskmaster import work_delegation
from tqdm import trange

from utils import get_nsims

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools


def fit_halo(particles, box):
    """
    Fit a single halo from the particle array.

    Parameters
    ----------
    particles : 2-dimensional array of shape `(n_particles, 3)`
        Particle array. The columns must be `x`, `y`, `z`, `vx`, `vy`, `vz`,
        `M`.
    box : object derived from :py:class`csiborgtools.read.BaseBox`
        Box object.

    Returns
    -------
    out : dict
    """
    halo = csiborgtools.fits.Halo(particles, box)

    out = {}
    out["npart"] = len(halo)
    out["totpartmass"] = numpy.sum(halo["M"])
    for i, v in enumerate(["vx", "vy", "vz"]):
        out[v] = numpy.average(halo.vel[:, i], weights=halo["M"])

    m200c, r200c, cm = halo.spherical_overdensity_mass(200, kind="crit",
                                                       maxiter=50)
    out["m200c"] = m200c
    out["r200c"] = r200c
    # out["lambda200c"] = halo.lambda_bullock(cm, r200c)
    out["conc"] = halo.nfw_concentration(cm, r200c)
    return out


def _main(nsim, simname, verbose):
    """
    Fit the FoF halos.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    simname : str
        Simulation name.
    verbose : bool
        Verbosity flag.
    """
    if simname == "quijote":
        raise NotImplementedError("Quijote not implemented yet.")

    cols = [("index", numpy.int32),
            ("npart", numpy.int32),
            ("totpartmass", numpy.float32),
            ("vx", numpy.float32),
            ("vy", numpy.float32),
            ("vz", numpy.float32),
            ("conc", numpy.float32),
            ("r200c", numpy.float32),
            ("m200c", numpy.float32),
            ("lambda200c", numpy.float32),]

    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)

    # Particle archive
    f = csiborgtools.read.read_h5(paths.particles(nsim))
    particles = f["particles"]
    halo_map = f["halomap"]
    hid2map = {hid: i for i, hid in enumerate(halo_map[:, 0])}
    cat = csiborgtools.read.CSiBORGHaloCatalogue(
        nsim, paths, with_lagpatch=False, load_initial=False, rawdata=True,
        load_fitted=False)

    out = csiborgtools.read.cols_to_structured(len(cat), cols)
    for i in trange(len(cat)) if verbose else range(len(cat)):
        hid = cat["index"][i]
        out["index"][i] = hid
        part = csiborgtools.read.load_halo_particles(hid, particles, halo_map,
                                                     hid2map)
        # Skip if no particles.
        if part is None:
            continue

        _out = fit_halo(part, box)
        for key in _out.keys():
            out[key][i] = _out[key]

    fout = paths.structfit(nsnap, nsim)
    if verbose:
        print(f"Saving to `{fout}`.", flush=True)
    numpy.save(fout, out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, default="csiborg",
                        choices=["csiborg", "quijote", "quijote_full"],
                        help="Simulation name")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="IC realisations. If `-1` processes all.")
    args = parser.parse_args()

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = get_nsims(args, paths)

    def main(nsim):
        _main(nsim, args.simname, MPI.COMM_WORLD.Get_size() == 1)

    work_delegation(main, nsims, MPI.COMM_WORLD)
