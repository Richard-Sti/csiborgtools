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
from os.path import join
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from itertools import combinations
from mpi4py import MPI
from TaskmasterMPI import master_process, worker_process
from sklearn.neighbors import NearestNeighbors
import joblib
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
parser.add_argument("--rmin", type=float)
parser.add_argument("--rmax", type=float)
parser.add_argument("--nneighbours", type=int)
parser.add_argument("--nsamples", type=int)
parser.add_argument("--neval", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

Rmax = 155 / 0.705  # Mpc/h high resolution region radius
mass_threshold = [1e12, 1e13, 1e14]  # Msun
ics = [7444, 7468, 7492, 7516, 7540, 7564, 7588, 7612, 7636, 7660, 7684,
       7708, 7732, 7756, 7780, 7804, 7828, 7852, 7876, 7900, 7924, 7948,
       7972, 7996, 8020, 8044, 8068, 8092, 8116, 8140, 8164, 8188, 8212,
       8236, 8260, 8284, 8308, 8332, 8356, 8380, 8404, 8428, 8452, 8476,
       8500, 8524, 8548, 8572, 8596, 8620, 8644, 8668, 8692, 8716, 8740,
       8764, 8788, 8812, 8836, 8860, 8884, 8908, 8932, 8956, 8980, 9004,
       9028, 9052, 9076, 9100, 9124, 9148, 9172, 9196, 9220, 9244, 9268,
       9292, 9316, 9340, 9364, 9388, 9412, 9436, 9460, 9484, 9508, 9532,
       9556, 9580, 9604, 9628, 9652, 9676, 9700, 9724, 9748, 9772, 9796,
       9820, 9844]
ics = [7444, 7468, 7492, 7516, 7540]
dumpdir = "/mnt/extraspace/rstiskalek/csiborg/knn"
fout_auto = join(dumpdir, "auto", "knncdf_{}.p")
fout_cross = join(dumpdir, "cross", "knncdf_{}_{}.p")


###############################################################################
#                               Analysis                                      #
###############################################################################
knncdf = csiborgtools.match.kNN_CDF()


def do_auto(ic):
    out = {}
    cat = csiborgtools.read.HaloCatalogue(ic, max_dist=Rmax)

    for i, mmin in enumerate(mass_threshold):
        knn = NearestNeighbors()
        knn.fit(cat.positions[cat["totpartmass"] > mmin, ...])

        rs, cdf = knncdf(knn, nneighbours=args.nneighbours, Rmax=Rmax,
                         rmin=args.rmin, rmax=args.rmax, nsamples=args.nsamples,
                         neval=args.neval, batch_size=args.batch_size,
                         random_state=args.seed, verbose=False)
        out.update({"cdf_{}".format(i): cdf})

    out.update({"rs": rs, "mass_threshold": mass_threshold})
    joblib.dump(out, fout_auto.format(ic))


def do_cross(ics):
    out = {}
    cat1 = csiborgtools.read.HaloCatalogue(ics[0], max_dist=Rmax)
    cat2 = csiborgtools.read.HaloCatalogue(ics[1], max_dist=Rmax)

    for i, mmin in enumerate(mass_threshold):
        knn1 = NearestNeighbors()
        knn1.fit(cat1.positions[cat1["totpartmass"] > mmin, ...])

        knn2 = NearestNeighbors()
        knn2.fit(cat2.positions[cat2["totpartmass"] > mmin, ...])

        rs, cdf0, cdf1, joint_cdf = knncdf.joint(
            knn1, knn2, nneighbours=args.nneighbours, Rmax=Rmax,
            rmin=args.rmin, rmax=args.rmax, nsamples=args.nsamples,
            neval=args.neval, batch_size=args.batch_size,
            random_state=args.seed)

        corr = knncdf.joint_to_corr(cdf0, cdf1, joint_cdf)

        out.update({"corr_{}".format(i): corr})

    out.update({"rs": rs, "mass_threshold": mass_threshold})
    joblib.dump(out, fout_cross.format(*ics))



###############################################################################
#                          Autocorrelation calculation                        #
###############################################################################


if nproc > 1:
    if rank == 0:
        tasks = deepcopy(ics)
        master_process(tasks, comm, verbose=True)
    else:
        worker_process(do_auto, comm, verbose=False)
else:
    tasks = deepcopy(ics)
    for task in tasks:
        print("{}: completing task `{}`.".format(datetime.now(), task))
        do_auto(task)
comm.Barrier()


###############################################################################
#                         Crosscorrelation calculation                        #
###############################################################################


if nproc > 1:
    if rank == 0:
        tasks = list(combinations(ics, 2))
        master_process(tasks, comm, verbose=True)
    else:
        worker_process(do_cross, comm, verbose=False)
else:
    tasks = deepcopy(ics)
    for task in tasks:
        print("{}: completing task `{}`.".format(datetime.now(), task))
        do_cross(task)
comm.Barrier()


if rank == 0:
    print("{}: all finished.".format(datetime.now()))
quit()  # Force quit the script
