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
A script to calculate the KNN-CDF for a set of halo catalogues.
"""
from argparse import ArgumentParser
from datetime import datetime
from distutils.util import strtobool

import joblib
import numpy
import yaml
from mpi4py import MPI
from sklearn.neighbors import NearestNeighbors
from taskmaster import work_delegation

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys

    sys.path.append("../")
    import csiborgtools

from utils import open_catalogues


def do_auto(args, config, cats, nsim, paths):
    """
    Calculate the kNN-CDF single catalogue auto-correlation.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    config : dict
        Configuration dictionary.
    cats : dict
        Dictionary of halo catalogues. Keys are simulation indices, values are
        the catalogues.
    nsim : int
        Simulation index.
    paths : csiborgtools.paths.Paths
        Paths object.

    Returns
    -------
    None
    """
    cat = cats[nsim]
    rvs_gen = csiborgtools.clustering.RVSinsphere(args.Rmax, cat.boxsize)
    knncdf = csiborgtools.clustering.kNN_1DCDF()
    knn = cat.knn(in_initial=False, subtract_observer=False, periodic=True)
    rs, cdf = knncdf(
        knn, rvs_gen=rvs_gen, nneighbours=config["nneighbours"],
        rmin=config["rmin"], rmax=config["rmax"],
        nsamples=int(config["nsamples"]), neval=int(config["neval"]),
        batch_size=int(config["batch_size"]), random_state=config["seed"])
    totvol = (4 / 3) * numpy.pi * args.Rmax ** 3
    fout = paths.knnauto(args.simname, args.run, nsim)
    if args.verbose:
        print(f"Saving output to `{fout}`.")
    joblib.dump({"rs": rs, "cdf": cdf, "ndensity": len(cat) / totvol}, fout)


def do_cross_rand(args, config, cats, nsim, paths):
    """
    Calculate the kNN-CDF cross catalogue random correlation.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    config : dict
        Configuration dictionary.
    cats : dict
        Dictionary of halo catalogues. Keys are simulation indices, values are
        the catalogues.
    nsim : int
        Simulation index.
    paths : csiborgtools.paths.Paths
        Paths object.

    Returns
    -------
    None
    """
    cat = cats[nsim]
    rvs_gen = csiborgtools.clustering.RVSinsphere(args.Rmax, cat.boxsize)
    knn1 = cat.knn(in_initial=False, subtract_observer=False, periodic=True)

    knn2 = NearestNeighbors()
    pos2 = rvs_gen(len(cat).shape[0])
    knn2.fit(pos2)

    knncdf = csiborgtools.clustering.kNN_1DCDF()
    rs, cdf0, cdf1, joint_cdf = knncdf.joint(
        knn1, knn2, rvs_gen=rvs_gen, nneighbours=int(config["nneighbours"]),
        rmin=config["rmin"], rmax=config["rmax"],
        nsamples=int(config["nsamples"]), neval=int(config["neval"]),
        batch_size=int(config["batch_size"]), random_state=config["seed"])
    corr = knncdf.joint_to_corr(cdf0, cdf1, joint_cdf)

    fout = paths.knnauto(args.simname, args.run, nsim)
    if args.verbose:
        print(f"Saving output to `{fout}`.", flush=True)
    joblib.dump({"rs": rs, "corr": corr}, fout)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=str, help="Run name.")
    parser.add_argument("--simname", type=str, choices=["csiborg", "quijote"],
                        help="Simulation name")
    parser.add_argument("--nsims", type=int, nargs="+", default=None,
                        help="Indices of simulations to cross. If `-1` processes all simulations.")  # noqa
    parser.add_argument("--Rmax", type=float, default=155,
                        help="High-resolution region radius")  # noqa
    parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)),
                        default=False)
    args = parser.parse_args()

    with open("./cluster_knn_auto.yml", "r") as file:
        config = yaml.safe_load(file)
    comm = MPI.COMM_WORLD
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    cats = open_catalogues(args, config, paths, comm)

    if args.verbose and comm.Get_rank() == 0:
        print(f"{datetime.now()}: starting to calculate the kNN statistic.")

    def do_work(nsim):
        if "random" in args.run:
            do_cross_rand(args, config, cats, nsim, paths)
        else:
            do_auto(args, config, cats, nsim, paths)

    nsims = list(cats.keys())
    work_delegation(do_work, nsims, comm, master_verbose=args.verbose)

    comm.Barrier()
    if comm.Get_rank() == 0:
        print(f"{datetime.now()}: all finished. Quitting.")
