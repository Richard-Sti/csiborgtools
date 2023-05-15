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
"""A script to calculate the KNN-CDF for a set of CSiBORG halo catalogues."""
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
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
with open("../scripts/knn_auto.yml", "r") as file:
    config = yaml.safe_load(file)

Rmax = 155 / 0.705  # Mpc (h = 0.705) high resolution region radius
totvol = 4 * numpy.pi * Rmax**3 / 3
paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
knncdf = csiborgtools.clustering.kNN_1DCDF()
if args.simname == "csiborg":
    ics = paths.get_ics()
else:
    ics = paths.get_quijote_ics()


###############################################################################
#                                 Analysis                                    #
###############################################################################


def read_single(nsim, selection, nobs=None):
    # We first read the full catalogue without applying any bounds.
    if args.nsim == "csiborg":
        cat = csiborgtools.read.HaloCatalogue(nsim, paths)
    else:
        cat = csiborgtools.read.QuijoteHaloCatalogue(nsim, paths, nsnap=4,
                                                     origin=nobs)

    cat.apply_bounds({"dist": (0, Rmax)})
    # We then first read off the primary selection bounds.
    ps = selection["primary"]
    cat.apply_bounds({ps["name"]: (ps.get("min", None), ps.get("max", None))})

    # Now the secondary selection bounds. If needed transfrom the secondary
    # property before applying the bounds.
    if "secondary" in selection:
        ss = selection["secondary"]
        name = ps["name"]

        if ss.get("toperm", False):
            cat[name] = numpy.random.permutation(cat[name])

        if ss.get("marked", False):
            cat[name] = csiborgtools.clustering.normalised_marks(
                cat[ps["name"]], cat[ss["name"]], nbins=config["nbins_marks"])
        cat.apply_bounds({name: (ps.get("min", None), ps.get("max", None))})
    return cat


def do_auto(run, nsim, nobs=None):
    """Calculate the kNN-CDF single catalgoue autocorrelation."""
    _config = config.get(run, None)
    if _config is None:
        warn(f"No configuration for run {run}.", UserWarning, stacklevel=1)
        return

    rvs_gen = csiborgtools.clustering.RVSinsphere(Rmax)
    cat = read_single(nsim, _config, nobs=nobs)
    knn = cat.knn(in_initial=False)
    rs, cdf = knncdf(
        knn, rvs_gen=rvs_gen, nneighbours=config["nneighbours"],
        rmin=config["rmin"], rmax=config["rmax"],
        nsamples=int(config["nsamples"]), neval=int(config["neval"]),
        batch_size=int(config["batch_size"]), random_state=config["seed"])

    fout = paths.knnauto_path(args.simname, run, nsim, nobs)
    joblib.dump({"rs": rs, "cdf": cdf, "ndensity": len(cat) / totvol}, fout)


def do_cross_rand(run, nsim, nobs=None):
    """Calculate the kNN-CDF cross catalogue random correlation."""
    _config = config.get(run, None)
    if _config is None:
        warn(f"No configuration for run {run}.", UserWarning, stacklevel=1)
        return

    rvs_gen = csiborgtools.clustering.RVSinsphere(Rmax)
    cat = read_single(nsim, _config)
    knn1 = cat.knn(in_initial=False)

    knn2 = NearestNeighbors()
    pos2 = rvs_gen(len(cat).shape[0])
    knn2.fit(pos2)

    rs, cdf0, cdf1, joint_cdf = knncdf.joint(
        knn1, knn2, rvs_gen=rvs_gen, nneighbours=int(config["nneighbours"]),
        rmin=config["rmin"], rmax=config["rmax"],
        nsamples=int(config["nsamples"]), neval=int(config["neval"]),
        batch_size=int(config["batch_size"]), random_state=config["seed"])
    corr = knncdf.joint_to_corr(cdf0, cdf1, joint_cdf)
    fout = paths.knnauto_path(args.simname, run, nsim, nobs)
    joblib.dump({"rs": rs, "corr": corr}, fout)


def do_runs(nsim):
    for run in args.runs:
        if "random" in run:
            do_cross_rand(run, nsim)
        else:
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
