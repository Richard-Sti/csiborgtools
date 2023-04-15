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
Script to generate the mmain files, i.e. sums up the substructe of children.
"""
from datetime import datetime
from mpi4py import MPI
import numpy
from TaskmasterMPI import master_process, worker_process
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools

# Get MPI things
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()

paths = csiborgtools.read.CSiBORGPaths(**csiborgtools.paths_glamdring)
mmain_reader = csiborgtools.read.MmainReader(paths)

def do_mmain(nsim):
    nsnap = max(paths.get_snapshots(nsim))
    # NOTE: currently works for highest snapshot anyway
    mmain, ultimate_parent = mmain_reader.make_mmain(nsim, verbose=True)
    numpy.savez(paths.mmain_path(nsnap, nsim),
                mmain=mmain, ultimate_parent=ultimate_parent)

###############################################################################
#                             MPI task delegation                             #
###############################################################################

if nproc > 1:
    if rank == 0:
        tasks = paths.get_ics(tonew=False)
        master_process(tasks, comm, verbose=True)
    else:
        worker_process(do_mmain, comm, verbose=False)
else:
    tasks = paths.get_ics(tonew=False)
    for task in tasks:
        print("{}: completing task `{}`.".format(datetime.now(), task))
        do_mmain(task)

comm.Barrier()
