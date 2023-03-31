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
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

Rmax = 155 / 0.705  # Mpc/h high resolution region radius
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
dumpdir = "/mnt/extraspace/rstiskalek/csiborg/knn"
fout = join(dumpdir, "knncdf_{}.p")


###############################################################################
#                               Analysis                                      #
###############################################################################
knncdf = csiborgtools.match.kNN_CDF()


def do_task(ic):
    cat = csiborgtools.read.HaloCatalogue(ic, min_mass=1e13,
                                          max_dist=Rmax)
    knn = NearestNeighbors()
    knn.fit(cat.positions)

    rs, cdfs = knncdf(knn, nneighbours=3, Rmax=Rmax, rmin=args.rmin,
                      rmax=args.rmax, nsamples=args.nsamples, neval=args.neval,
                      random_state=args.seed, verbose=False)
    joblib.dump({"rs": rs, "cdfs": cdfs}, fout.format(ic))


if nproc > 1:
    if rank == 0:
        tasks = deepcopy(ics)
        master_process(tasks, comm, verbose=True)
    else:
        worker_process(do_task, comm, verbose=False)
else:
    tasks = deepcopy(ics)
    for task in tasks:
        print("{}: completing task `{}`.".format(datetime.now(), task))
        do_task(task)


comm.Barrier()
if rank == 0:
    print("{}: all finished.".format(datetime.now()))
quit()  # Force quit the script