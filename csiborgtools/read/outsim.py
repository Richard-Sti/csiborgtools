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
from os.path import (join, dirname, basename, isfile)
from os import remove
from tqdm import trange
from astropy.io import ascii
from astropy.table import Table

I64 = numpy.int64
F64 = numpy.float64


def dump_split(arr, n_split, paths):
    """
    Dump an array from a split.

    Parameters
    ----------
    arr : n-dimensional or structured array
        Array to be saved.
    n_split: int
        The split index.
    paths : py:class`csiborgtools.read.CSiBORGPaths`
        CSiBORG paths-handling object with set `n_sim` and `n_snap`.
    n_sim : int
        The CSiBORG realisation index.
    n_snap : int
        The index of a redshift snapshot.
    outdir : string
        Directory where to save the temporary files.

    Returns
    -------
    None
    """
    n_sim = str(paths.n_sim).zfill(5)
    n_snap = str(paths.n_snap).zfill(5)
    fname = join(paths.temp_dumpdir, "ramses_out_{}_{}_{}.npy"
                 .format(n_sim, n_snap, n_split))
    numpy.save(fname, arr)


def combine_splits(n_splits, part_reader, cols_add, remove_splits=False,
                   verbose=True):
    """
    Combine results of many splits saved from `dump_split`. Identifies to which
    clump the clumps in the split correspond to by matching their index.
    Returns an array that contains the original clump data along with the newly
    calculated quantities.

    Paramaters
    ----------
    n_splits : int
        The total number of clump splits.
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
    # Load clumps to see how many there are and will add to this array
    n_sim = part_reader.paths.n_sim
    n_snap = part_reader.paths.n_snap
    clumps = part_reader.read_clumps(cols=None)

    # Get the old + new dtypes and create an empty array
    descr = clumps.dtype.descr + cols_add
    out = numpy.full(clumps.size, numpy.nan, dtype=descr)
    # Now put the old values into the array
    for par in clumps.dtype.names:
        out[par] = clumps[par]

    # Filename of splits data
    froot = "ramses_out_{}_{}".format(
        str(n_sim).zfill(5), str(n_snap).zfill(5))
    fname = join(part_reader.paths.temp_dumpdir, froot + "_{}.npy")

    # Iterate over splits and add to the output array
    cols_add_names = [col[0] for col in cols_add]
    iters = trange(n_splits) if verbose else range(n_splits)
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


def make_ascii_powmes(particles, fout, verbose=True):
    """
    Write an ASCII file with appropriate formatting for POWMES.
    This is an extremely memory inefficient implementation.

    Parameters
    ----------
    particles : structured array
        Array of particles.
    fout : str
        File path to store the ASCII file.
    verbose : bool, optional
        Verbosity flag. By default `True`.

    Returns
    -------
    None
    """
    out = Table()
    for p in ('x', 'y', 'z', 'M'):
        out[p] = particles[p]
    # If fout exists, remove
    if isfile(fout):
        remove(fout)

    # Write the temporaty file
    ftemp = join(dirname(fout), "_" + basename(fout))
    if verbose:
        print("Writing temporary file `{}`...".format(ftemp))
    ascii.write(out, ftemp, overwrite=True, delimiter=",", fast_writer=True)

    # Write to the first line the number of particles
    if verbose:
        print("Writing the full file `{}`...".format(fout))
    with open(ftemp, 'r') as fread, open(fout, 'w') as fwrite:
        fwrite.write(str(particles.size) + '\n')
        for i, line in enumerate(fread):
            if i == 0:
                continue
            fwrite.write(line)

    remove(ftemp)
