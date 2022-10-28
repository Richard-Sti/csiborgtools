# Copyright (C) 2022 Richard Stiskalek, Deaglan Bartlett
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
Tools for fitting the halos and distributing the jobs.
"""


import numpy
from os.path import join
from tqdm import trange
from ..io import nparts_to_start_ind


def clump_with_particles(particle_clumps, clumps, verbose=True):
    """
    Count how many particles does each clump have.

    Parameters
    ----------
    particle_clumps : 1-dimensional array
        Array of particles' clump IDs.
    clumps : structured array
        The clumps array.

    Returns
    -------
    with_particles : 1-dimensional array
        Array of whether a clump has any particles.
    """
    if verbose:
        print("Determining unique particles' clump IDs...")
    unique_particle_clumps = numpy.unique(particle_clumps)
    if verbose:
        print("Checking which clumps have particles...")
    return numpy.isin(clumps["index"], unique_particle_clumps)


def distribute_halos(Nsplits, clumps):
    """
    Evenly distribute clump indices to smaller splits. Clumps should only be
    clumps that contain particles.

    Parameters
    ----------
    Nsplits : int
        Number of splits.
    clumps : structured array
        The clumps array.

    Returns
    -------
    splits : 2-dimensional array
        Array of starting and ending indices of each CPU of shape `(Njobs, 2)`.
    """
    # Make sure these are unique IDs
    indxs = clumps["index"]
    if indxs.size > numpy.unique((indxs)).size:
        raise ValueError("`clump_indxs` constains duplicate indices.")
    Ntotal = indxs.size
    Njobs_per_cpu = numpy.ones(Nsplits, dtype=int) * Ntotal // Nsplits
    # Split the remainder Ntotal % Njobs among the CPU
    Njobs_per_cpu[:Ntotal % Nsplits] += 1
    start = nparts_to_start_ind(Njobs_per_cpu)
    return numpy.vstack([start, start + Njobs_per_cpu]).T


def dump_particles(particles, particle_clumps, clumps, Nsplits, dumpfolder,
                   Nsim, Nsnap, verbose=True):
    """
    Save the data needed for each split so that a process does not have to load
    everything. These clumps should already be only the ones with particles.

    Parameters
    ----------
    particles : structured array
        The particle array.
    particle_clumps : 1-dimensional array
        Array of particles' clump IDs.
    clumps : structured array
        The clumps array.
    Nsplits : int
        Number of times to split the clumps.
    dumpfolder : str
        Path to the folder where to dump the splits.
    Nsim : int
        CSiBORG simulation index.
    Nsnap : int
        Snapshot index.
    verbose : bool, optional
        Verbosity flag. By default `True`.

    Returns
    -------
    None
    """
    if particles.size != particle_clumps.size:
        raise ValueError("`particles` must correspond to `particle_clumps`.")
    # The starting clump index of each split
    splits = distribute_halos(Nsplits, clumps)
    fname = join(dumpfolder, "out_{}_snap_{}_{}.npz")

    iters = trange(Nsplits) if verbose else range(Nsplits)
    tot = 0
    for n in iters:
        # Lower and upper array index of the clumps array
        i, j = splits[n, :]
        # Lower and upper clump index. Need - 1 not to take the last val.
        ipart = clumps["index"][i]
        jpart = clumps["index"][j - 1]
        mask = (particle_clumps >= ipart) & (particle_clumps <= jpart)
        # Dump it!
        tot += mask.sum()
        fout = fname.format(Nsim, Nsnap, n)
        numpy.savez(fout, particles[mask], particle_clumps[mask], clumps[i:j])

    if tot != particle_clumps.size:
        raise RuntimeError("Num. of dumped particles `{}` does not particle "
                           "file size `{}`.".format(tot, particle_clumps.size))
