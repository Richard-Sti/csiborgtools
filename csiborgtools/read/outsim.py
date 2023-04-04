# Copyright (C) 2022 Richard Stiskalek, Harry Desmond
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
I/O functions for analysing the CSiBORG realisations.
"""
import numpy
from os.path import join
from os import remove
from tqdm import trange


def dump_split(arr, nsplit, nsnap, nsim, paths):
    """
    Dump an array from a split.

    Parameters
    ----------
    arr : n-dimensional or structured array
        Array to be saved.
    nsplit : int
         Split index.
    nsnap : int
        Snapshot index.
    nsim : int
        IC realisation index.
    paths : py:class`csiborgtools.read.CSiBORGPaths`
        CSiBORG paths-handling object with set `n_sim` and `n_snap`.

    Returns
    -------
    None
    """
    fname = join(paths.temp_dumpdir, "ramses_out_{}_{}_{}.npy"
                 .format(str(nsim).zfill(5), str(nsnap).zfill(5), nsplit))
    numpy.save(fname, arr)


def combine_splits(nsplits, nsnap, nsim, part_reader, cols_add,
                   remove_splits=False, verbose=True):
    """
    Combine results of many splits saved from `dump_split`. Identifies to which
    clump the clumps in the split correspond to by matching their index.
    Returns an array that contains the original clump data along with the newly
    calculated quantities.

    Paramaters
    ----------
    nsplits : int
        Total number of clump splits.
    nsnap : int
        Snapshot index.
    nsim : int
        IC realisation index.
    part_reader : py:class`csiborgtools.read.ParticleReadear`
        CSiBORG particle reader.
    cols_add : list of `(str, dtype)`
        Colums to add. Must be formatted as, for example,
        `[("npart", numpy.float64), ("totpartmass", numpy.float64)]`.
    remove_splits : bool, optional
        Whether to remove the splits files. By default `False`.
    verbose : bool, optional
        Verbosity flag. By default `True`.

    Returns
    -------
    out : structured array
        Clump array with appended results from the splits.
    """
    clumps = part_reader.read_clumps(nsnap, nsim, cols=None)
    # Get the old + new dtypes and create an empty array
    descr = clumps.dtype.descr + cols_add
    out = numpy.full(clumps.size, numpy.nan, dtype=descr)
    for par in clumps.dtype.names:  # Now put the old values into the array
        out[par] = clumps[par]

    # Filename of splits data
    froot = "ramses_out_{}_{}".format(str(nsim).zfill(5), str(nsnap).zfill(5))
    fname = join(part_reader.paths.temp_dumpdir, froot + "_{}.npy")

    # Iterate over splits and add to the output array
    cols_add_names = [col[0] for col in cols_add]
    iters = trange(nsplits) if verbose else range(nsplits)
    for n in iters:
        fnamesplit = fname.format(n)
        arr = numpy.load(fnamesplit)

        # Check that all halo indices from the split are in the clump file
        if not numpy.alltrue(numpy.isin(arr["index"], out["index"])):
            raise KeyError("....")
        # Mask of where to put the values from the split
        mask = numpy.isin(out["index"], arr["index"])
        for par in cols_add_names:
            out[par][mask] = arr[par]

        # Now remove this split
        if remove_splits:
            remove(fnamesplit)

    return out
