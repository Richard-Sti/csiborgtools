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


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


###############################################################################
#                                 Analysis                                    #
###############################################################################


def read_single(nsim, selection, nobs=None):
    # We first read the full catalogue without applying any bounds.
    if args.simname == "csiborg":
        cat = csiborgtools.read.HaloCatalogue(nsim, paths)
    else:
        cat = csiborgtools.read.QuijoteHaloCatalogue(nsim, paths, nsnap=4,
                                                     origin=nobs)

    cat.apply_bounds({"dist": (0, Rmax)})
    # We then first read off the primary selection bounds.
    sel = selection["primary"]
    pname = None
    xs = sel["name"] if isinstance(sel["name"], list) else [sel["name"]]
    for _name in xs:
        if _name in cat.keys:
            pname = _name
    if pname is None:
        raise KeyError(f"Invalid names `{sel['name']}`.")

    cat.apply_bounds({pname: (sel.get("min", None), sel.get("max", None))})

    # Now the secondary selection bounds. If needed transfrom the secondary
    # property before applying the bounds.
    if "secondary" in selection:
        sel = selection["secondary"]
        sname = None
        xs = sel["name"] if isinstance(sel["name"], list) else [sel["name"]]
        for _name in xs:
            if _name in cat.keys:
                sname = _name
        if sname is None:
            raise KeyError(f"Invalid names `{sel['name']}`.")

        if sel.get("toperm", False):
            cat[sname] = numpy.random.permutation(cat[sname])

        if sel.get("marked", False):
            cat[sname] = csiborgtools.clustering.normalised_marks(
                cat[pname], cat[sname], nbins=config["nbins_marks"])
        cat.apply_bounds({sname: (sel.get("min", None), sel.get("max", None))})

    if selection.get("poisson", False):
        rvs_gen = csiborgtools.clustering.RVSinsphere(Rmax)
        pos = rvs_gen(len(cat), random_state=None)
        for i, p in enumerate(['x', 'y', 'z']):
            cat.data[p] = pos[:, i]

    return cat


def do_auto(run, nsim, nobs=None):
    """Calculate the kNN-CDF single catalgoue autocorrelation."""
    _config = config.get(run, None)
    if _config is None:
        warn(f"No configuration for run {run}.", UserWarning, stacklevel=1)
        return

    rvs_gen = csiborgtools.clustering.RVSinsphere(Rmax)
    # rvs_gen = csiborgtools.clustering.RVSinbox(Rmax)
    cat = read_single(nsim, _config, nobs=nobs)
    knn = cat.knn(in_initial=False)
    rs, cdf = knncdf(
        knn, rvs_gen=rvs_gen, nneighbours=config["nneighbours"],
        rmin=config["rmin"], rmax=config["rmax"],
        nsamples=int(config["nsamples"]), neval=int(config["neval"]),
        batch_size=int(config["batch_size"]), random_state=config["seed"])

    fout = paths.knnauto(args.simname, run, nsim, nobs)
    print(f"Saving output to `{fout}`.")
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
    fout = paths.knnauto(args.simname, run, nsim, nobs)
    print(f"Saving output to `{fout}`.", flush=True)
    joblib.dump({"rs": rs, "corr": corr}, fout)


def do_runs(nsim):
    for run in args.runs:
        iters = [0, 5] if args.simname == "quijote" else [None]
        for nobs in iters:
            if "random" in run:
                do_cross_rand(run, nsim, nobs)
            else:
                do_auto(run, nsim, nobs)


###############################################################################
#                             MPI task delegation                             #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--runs", type=str, nargs="+")
    parser.add_argument("--simname", type=str, choices=["csiborg", "quijote"])
    parser.add_argument("--ics", type=int, nargs="+", default=None,
                        help="IC realisations. If `-1` processes all simulations.")  # noqa
    args = parser.parse_args()
    with open("../scripts/cluster_knn_auto.yml", "r") as file:
        config = yaml.safe_load(file)

    Rmax = 155 / 0.705  # Mpc (h = 0.705) high resolution region radius
    totvol = 4 * numpy.pi * Rmax**3 / 3
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    knncdf = csiborgtools.clustering.kNN_1DCDF()

    if args.ics is None or args.ics[0] == -1:
        if args.simname == "csiborg":
            ics = paths.get_ics("csiborg")
        else:
            ics = paths.get_quijote_ics()
    else:
        ics = args.ics

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
