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
A script to calculate the KNN-CDF for a set of CSiBORG halo catalogues.

TODO:
    - [ ] Update catalogue readers.
    - [ ] Update paths.
    - [ ] Update to cross-correlate different mass populations from different
    simulations.
"""
from argparse import ArgumentParser
from datetime import datetime
from itertools import combinations
from warnings import warn

import joblib
import numpy
import yaml
from mpi4py import MPI
from sklearn.neighbors import NearestNeighbors
from taskmaster import master_process, worker_process

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
parser.add_argument("--simname", type=str, choices=["csiborg", "quijote"])
args = parser.parse_args()
with open("../scripts/knn_cross.yml", "r") as file:
    config = yaml.safe_load(file)

Rmax = 155 / 0.705  # Mpc (h = 0.705) high resolution region radius
paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
ics = paths.get_ics("csiborg")
knncdf = csiborgtools.clustering.kNN_1DCDF()

###############################################################################
#                               Analysis                                      #
###############################################################################


def read_single(selection, cat):
    mmask = numpy.ones(len(cat), dtype=bool)
    pos = cat.positions(False)
    # Primary selection
    psel = selection["primary"]
    pmin, pmax = psel.get("min", None), psel.get("max", None)
    if pmin is not None:
        mmask &= cat[psel["name"]] >= pmin
    if pmax is not None:
        mmask &= cat[psel["name"]] < pmax
    return pos[mmask, ...]


def do_cross(run, ics):
    _config = config.get(run, None)
    if _config is None:
        warn("No configuration for run {}.".format(run), stacklevel=1)
        return
    rvs_gen = csiborgtools.clustering.RVSinsphere(Rmax)
    knn1, knn2 = NearestNeighbors(), NearestNeighbors()

    cat1 = csiborgtools.read.ClumpsCatalogue(ics[0], paths, max_dist=Rmax)
    pos1 = read_single(_config, cat1)
    knn1.fit(pos1)

    cat2 = csiborgtools.read.ClumpsCatalogue(ics[1], paths, max_dist=Rmax)
    pos2 = read_single(_config, cat2)
    knn2.fit(pos2)

    rs, cdf0, cdf1, joint_cdf = knncdf.joint(
        knn1,
        knn2,
        rvs_gen=rvs_gen,
        nneighbours=int(config["nneighbours"]),
        rmin=config["rmin"],
        rmax=config["rmax"],
        nsamples=int(config["nsamples"]),
        neval=int(config["neval"]),
        batch_size=int(config["batch_size"]),
        random_state=config["seed"],
    )

    corr = knncdf.joint_to_corr(cdf0, cdf1, joint_cdf)
    fout = paths.knncross_path(args.simname, run, ics)
    joblib.dump({"rs": rs, "corr": corr}, fout)


def do_runs(nsims):
    for run in args.runs:
        do_cross(run, nsims)


###############################################################################
#                         Crosscorrelation calculation                        #
###############################################################################


if nproc > 1:
    if rank == 0:
        tasks = list(combinations(ics, 2))
        master_process(tasks, comm, verbose=True)
    else:
        worker_process(do_runs, comm, verbose=False)
else:
    tasks = list(combinations(ics, 2))
    for task in tasks:
        print("{}: completing task `{}`.".format(datetime.now(), task))
        do_runs(task)
comm.Barrier()


if rank == 0:
    print("{}: all finished.".format(datetime.now()))
quit()  # Force quit the script
