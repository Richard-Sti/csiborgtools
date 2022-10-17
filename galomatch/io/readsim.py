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

"""Functions to read in the particle and clump files."""

import numpy
from scipy.io import FortranFile
from os import listdir
from os.path import (join, isfile)
from tqdm import tqdm

from ..utils import cols_to_structured


F16 = numpy.float16
F32 = numpy.float32
F64 = numpy.float64
I32 = numpy.int32
I64 = numpy.int64
little_h = 0.705
BOXSIZE = 677.7 / little_h  # Mpc. Otherwise positions in [0, 1].
BOXMASS = 3.749e19  # Msun


def get_sim_path(n, fname="ramses_out_{}", srcdir="/mnt/extraspace/hdesmond"):
    """
    Get a path to a CSiBORG simulation.

    Parameters
    ----------
    n : int
        The index of the initial conditions (IC) realisation.
    fname : str, optional
        The file name. By default `ramses_out_{}`, where `n` is the IC index.
    srcdir : str, optional
        The file path to the folder where realisations of the ICs are stored.

    Returns
    -------
    path : str
        The complete path to the `n`th CSiBORG simulation.
    """
    return join(srcdir, fname.format(n))


def open_particle(n, simpath, verbose=True):
    """
    Open particle files to a given CSiBORG simulation.

    Parameters
    ----------
    n : int
        The index of a redshift snapshot.
    simpath : str
        The complete path to the CSiBORG simulation.
    verbose : bool, optional
        Verbosity flag.

    Returns
    -------
    nparts : 1-dimensional array
        Number of parts assosiated with each CPU.
    partfiles : list of `scipy.io.FortranFile`
        Opened part files.
    """
    # Zeros filled snapshot number and the snapshot path
    nout = str(n).zfill(5)
    snappath = join(simpath, "output_{}".format(nout))
    infopath = join(snappath, "info_{}.txt".format(nout))

    with open(infopath, "r") as f:
        ncpu = int(f.readline().split()[-1])
    if verbose:
        print("Reading in output `{}` with ncpu = `{}`.".format(nout, ncpu))

    # Check whether the unbinding file exists.
    snapdirlist = listdir(snappath)
    unbinding_file = "unbinding_{}.out00001".format(nout)
    if unbinding_file not in snapdirlist:
        raise FileNotFoundError(
            "Couldn't find `{}` in `{}`. Use mergertreeplot.py -h or --help to "
            "print help message.".format(unbinding_file, snappath))

    # First read the headers. Reallocate arrays and fill them.
    nparts = numpy.zeros(ncpu, dtype=int)
    partfiles = [None] * ncpu
    for cpu in range(ncpu):
        cpu_str = str(cpu + 1).zfill(5)
        fpath = join(snappath, "part_{}.out{}".format(nout, cpu_str))

        f = FortranFile(fpath)
        # Read in this order
        ncpuloc = f.read_ints()
        if ncpuloc != ncpu:
            raise ValueError("`ncpu = {}` of `{}` disagrees with `ncpu = {}` "
                             "of `{}`.".format(ncpu, infopath, ncpuloc, fpath))
        ndim = f.read_ints()
        nparts[cpu] = f.read_ints()
        localseed = f.read_ints()
        nstar_tot = f.read_ints()
        mstar_tot = f.read_reals('d')
        mstar_lost = f.read_reals('d')
        nsink = f.read_ints()

        partfiles[cpu] = f

    return nparts, partfiles


def read_sp(dtype, partfile):
    """
    Utility function to read a single particle file, depending on the dtype.

    Parameters
    ----------
    dtype : str
        The dtype of the part file to be read now.
    partfile : `scipy.io.FortranFile`
        Part file to read from.

    Returns
    -------
    out : 1-dimensional array
        The data read from the part file.
    n : int
        The index of the initial conditions (IC) realisation.
    simpath : str
        The complete path to the CSiBORG simulation.
    """
    if dtype in [F16, F32, F64]:
        return partfile.read_reals('d')
    elif dtype in [I32]:
        return partfile.read_ints()
    else:
        raise TypeError("Unexpected dtype `{}`.".format(dtype))


def nparts_to_start_ind(nparts):
    """
    Convert `nparts` array to starting indices in a pre-allocated array for looping over the CPU number.

    Parameters
    ----------
    nparts : 1-dimensional array
        Number of parts assosiated with each CPU.

    Returns
    -------
    start_ind : 1-dimensional array
        The starting indices calculated as a cumulative sum starting at 0.
    """
    return numpy.hstack([[0], numpy.cumsum(nparts[:-1])])


def read_particle(pars_extract, n, simpath, verbose=True):
    """
    Read particle files of a simulation at a given snapshot and return
    values of `pars_extract`.

    Parameters
    ----------
    pars_extract : list of str
        Parameters to be extacted.
    n : int
        The index of the redshift snapshot.
    simpath : str
        The complete path to the CSiBORG simulation.
    verbose : bool, optional
        Verbosity flag while for reading the CPU outputs.

    Returns
    -------
    out : structured array
        The data read from the particle file.
    """
    # Open the particle files
    nparts, partfiles = open_particle(n, simpath)
    if verbose:
        print("Opened {} particle files.".format(nparts.size))
    ncpu = nparts.size
    # Order in which the particles are written in the FortranFile
    forder = [("x", F16), ("y", F16), ("z", F16),
              ("vx", F16), ("vy", F16), ("vz", F16),
              ("M", F32), ("ID", I32), ("level", I32)]
    fnames = [fp[0] for fp in forder]
    fdtypes = [fp[1] for fp in forder]
    # Check there are no strange parameters
    for p in pars_extract:
        if p not in fnames:
            raise ValueError("Undefined parameter `{}`. Must be one of `{}`."
                             .format(p, fnames))

    npart_tot = numpy.sum(nparts)
    # A dummy array is necessary for reading the fortran files.
    dum = numpy.full(npart_tot, numpy.nan, dtype=F16)
    # These are the data we read along with types
    dtype = {"names": pars_extract,
             "formats": [forder[fnames.index(p)][1] for p in pars_extract]}
    # Allocate the output structured array
    out = numpy.full(npart_tot, numpy.nan, dtype)
    start_ind = nparts_to_start_ind((nparts))
    iters = tqdm(range(ncpu)) if verbose else range(ncpu)
    for cpu in iters:
        i = start_ind[cpu]
        j = nparts[cpu]
        for (fname, fdtype) in zip(fnames, fdtypes):
            if fname in pars_extract:
                out[fname][i:i + j] = read_sp(fdtype, partfiles[cpu])
            else:
                dum[i:i + j] = read_sp(fdtype, partfiles[cpu])

    return out


def open_unbinding(cpu, n, simpath):
    """
    Open particle files to a given CSiBORG simulation. Note that to be consistent CPU is incremented by 1.

    Parameters
    ----------
    cpu : int
        The CPU index.
    n : int
        The index of a redshift snapshot.
    simpath : str
        The complete path to the CSiBORG simulation.

    Returns
    -------
    unbinding : `scipy.io.FortranFile`
        The opened unbinding FortranFile.
    """
    nout = str(n).zfill(5)
    cpu = str(cpu + 1).zfill(5)
    fpath = join(simpath, "output_{}".format(nout),
                 "unbinding_{}.out{}".format(nout, cpu))

    return FortranFile(fpath)


def read_clumpid(n, simpath, verbose=True):
    """
    Read clump IDs from unbinding files.

    Parameters
    ----------
    n : int
        The index of a redshift snapshot.
    simpath : str
        The complete path to the CSiBORG simulation.
    verbose : bool, optional
        Verbosity flag while for reading the CPU outputs.

    Returns
    -------
    clumpid : 1-dimensional array
        The array of clump IDs.
    """
    nparts, __ = open_particle(n, simpath, verbose)
    start_ind = nparts_to_start_ind(nparts)
    ncpu = nparts.size

    clumpid = numpy.full(numpy.sum(nparts), numpy.nan)
    iters = tqdm(range(ncpu)) if verbose else range(ncpu)
    for cpu in iters:
        i = start_ind[cpu]
        j = nparts[cpu]
        ff = open_unbinding(cpu, n, simpath)
        clumpid[i:i + j] = ff.read_ints()

    return clumpid


def read_clumps(n, simpath):
    """
    Read in a precomputed clump file `clump_N.dat`.

    Parameters
    ----------
    n : int
        The index of a redshift snapshot.
    simpath : str
        The complete path to the CSiBORG simulation.

    Returns
    -------
    out : structured array
        Structured array of the clumps.
    """
    n = str(n).zfill(5)
    fname = join(simpath, "output_{}".format(n), "clump_{}.dat".format(n))
    # Check the file exists.
    if not isfile(fname):
        raise FileExistsError("Clump file `{}` does not exist.".format(fname))

    # Read in the clump array. This is how the columns must be written!
    arr = numpy.genfromtxt(fname)
    cols = [("index", I64), ("level", I64), ("parent", I64), ("ncell", F64),
            ("peak_x", F64), ("peak_y", F64), ("peak_z", F64),
            ("rho-", F64), ("rho+", F64), ("rho_av", F64),
            ("mass_cl", F64), ("relevance", F64)]
    # Write to a structured array
    dtype = {"names": [col[0] for col in cols],
            "formats": [col[1] for col in cols]}
    out = numpy.full(arr.shape[0], numpy.nan, dtype=dtype)
    for i, name in enumerate(dtype["names"]):
        out[name] = arr[:, i]
    return out


def read_mmain(n, srcdir, fname="Mmain_{}.npy"):
    """
    Read `mmain` numpy arrays of central halos whose mass contains their
    substracture contribution.

    Parameters
    ----------
    n : int
        The index of the initial conditions (IC) realisation.
    srcdir : str
        The path to the folder containing the files.
    fname : str, optional
        The file name convention.  By default `Mmain_{}.npy`, where the
        substituted value is `n`.

    Returns
    -------
    out : structured array
        Array with the central halo information.
    """
    fpath = join(srcdir, fname.format(n))

    arr = numpy.load(fpath)


    cols = [("index", I64), ("peak_x", F64), ("peak_y", F64),
            ("peak_z", F64), ("mass_cl", F64), ("sub_frac", F64)]
    out = cols_to_structured(arr.shape[0], cols)
    for i, name in enumerate(out.dtype.names):
        out[name] = arr[:, i]

    return out



def convert_mass_cols(arr, cols):
    """
    Convert mass columns from box units to :math:`M_{odot}`. `arr` is passed by
    reference and is not explicitly returned back.

    Parameters
    ----------
    arr : structured array
        The array whose columns are to be converted.
    cols : str or list of str
        The mass columns to be converted.

    Returns
    -------
    None
    """
    cols = [cols] if isinstance(cols, str) else cols
    for col in cols:
        arr[col] *= BOXMASS


def convert_position_cols(arr, cols, zero_centered=True):
    """
    Convert position columns from box units to :math:`\mathrm{Mpc}`. `arr` is
    passed by reference and is not explicitly returned back.

    Parameters
    ----------
    arr : structured array
        The array whose columns are to be converted.
    cols : str or list of str
        The mass columns to be converted.
    zero_centered : bool, optional
        Whether to translate the well-resolved origin in the centre of the
        simulation to the :math:`(0, 0 , 0)` point. By default `True`.

    Returns
    -------
    None
    """
    cols = [cols] if isinstance(cols, str) else cols
    for col in cols:
        arr[col] *= BOXSIZE
        if zero_centered:
            arr[col] -= BOXSIZE / 2


def flip_cols(arr, col1, col2):
    """
    Flip values in columns `col1` and `col2`. `arr` is passed by reference and
    is not explicitly returned back.


    Parameters
    ----------
    arr : structured array
        The array whose columns are to be converted.
    col1 : str
        The first column name.
    col2 : str
        The second column name.

    Returns
    -------
    nothing
    """
    dum = numpy.copy(arr[col1])
    arr[col1] = arr[col2]
    arr[col2] = dum
