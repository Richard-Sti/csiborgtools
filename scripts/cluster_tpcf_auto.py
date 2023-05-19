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

TODO:
    - [ ] Add support for new catalogue readers. Currently will not work.
"""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from warnings import warn

import joblib
import numpy
import yaml
from mpi4py import MPI

from taskmaster import master_process, worker_process

from .cluster_knn_auto import read_single

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools


###############################################################################
#                            MPI and arguments                                #
###############################################################################
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

parser = ArgumentParser()
parser.add_argument("--runs", type=str, nargs="+")
parser.add_argument("--ics", type=int, nargs="+", default=None,
                    help="IC realisations. If `-1` processes all simulations.")
parser.add_argument("--simname", type=str, choices=["csiborg", "quijote"])
args = parser.parse_args()
with open("../scripts/tpcf_auto.yml", "r") as file:
    config = yaml.safe_load(file)

Rmax = 155 / 0.705  # Mpc (h = 0.705) high resolution region radius
paths = csiborgtools.read.Paths()
tpcf = csiborgtools.clustering.Mock2PCF()

if args.ics is None or args.ics[0] == -1:
    if args.simname == "csiborg":
        ics = paths.get_ics("csiborg")
    else:
        ics = paths.get_quijote_ics()
else:
    ics = args.ics

###############################################################################
#                                 Analysis                                    #
###############################################################################


def do_auto(run, nsim):
    _config = config.get(run, None)
    if _config is None:
        warn("No configuration for run {}.".format(run), stacklevel=1)
        return

    rvs_gen = csiborgtools.clustering.RVSinsphere(Rmax)
    bins = numpy.logspace(
        numpy.log10(config["rpmin"]),
        numpy.log10(config["rpmax"]),
        config["nrpbins"] + 1,
    )
    cat = read_single(nsim, _config)
    pos = cat.position(in_initial=False, cartesian=True)
    nrandom = int(config["randmult"] * pos.shape[0])
    rp, wp = tpcf(pos, rvs_gen, nrandom, bins)

    fout = paths.tpcfauto(args.simname, run, nsim)
    joblib.dump({"rp": rp, "wp": wp}, fout)


def do_runs(nsim):
    for run in args.runs:
        do_auto(run, nsim)


###############################################################################
#                             MPI task delegation                             #
###############################################################################


if nproc > 1:
    if rank == 0:
        tasks = deepcopy(ics)
        master_process(tasks, comm, verbose=True)
    else:
        worker_process(do_runs, comm, verbose=False)
else:
    tasks = deepcopy(ics)
    for task in tasks:
        print("{}: completing task `{}`.".format(datetime.now(), task))
        do_runs(task)
comm.Barrier()


if rank == 0:
    print("{}: all finished.".format(datetime.now()))
quit()  # Force quit the script
