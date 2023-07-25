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
A script to calculate the auto-2PCF of CSiBORG catalogues.
"""
from argparse import ArgumentParser
from datetime import datetime
from distutils.util import strtobool

import joblib
import numpy
import yaml
from mpi4py import MPI

from taskmaster import work_delegation
from utils import open_catalogues

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools


def do_auto(args, config, cats, nsim, paths):
    tpcf = csiborgtools.clustering.Mock2PCF()
    rvs_gen = csiborgtools.clustering.RVSinsphere(args.Rmax)
    bins = numpy.logspace(
        numpy.log10(config["rpmin"]), numpy.log10(config["rpmax"]),
        config["nrpbins"] + 1,)
    cat = cats[nsim]

    pos = cat.position(in_initial=False, cartesian=True)
    nrandom = int(config["randmult"] * pos.shape[0])
    rp, wp = tpcf(pos, rvs_gen, nrandom, bins)

    fout = paths.knnauto(args.simname, args.run, nsim)
    joblib.dump({"rp": rp, "wp": wp}, fout)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=str, help="Run name.")
    parser.add_argument("--simname", type=str, choices=["csiborg", "quijote"],
                        help="Simulation name")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="Indices of simulations to cross. If `-1` processes all simulations.")  # noqa
    parser.add_argument("--Rmax", type=float, default=155/0.705,
                        help="High-resolution region radius")  # noqa
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        default=False)
    args = parser.parse_args()

    with open("./cluster_tpcf_auto.yml", "r") as file:
        config = yaml.safe_load(file)

    comm = MPI.COMM_WORLD
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    cats = open_catalogues(args, config, paths, comm)

    if args.verbose and comm.Get_rank() == 0:
        print(f"{datetime.now()}: starting to calculate the 2PCF statistic.")

    def do_work(nsim):
        return do_auto(args, config, cats, nsim, paths)

    nsims = list(cats.keys())
    work_delegation(do_work, nsims, comm)
