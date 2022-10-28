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
Functions to read in the particle and clump files.
"""

import numpy
from scipy.io import FortranFile
from os import listdir
from os.path import (join, isfile)
from glob import glob
from tqdm import tqdm

from ..utils import cols_to_structured


F16 = numpy.float16
F32 = numpy.float32
F64 = numpy.float64
I32 = numpy.int32
I64 = numpy.int64


def get_csiborg_ids(srcdir):
    """
    Get CSiBORG simulation IDs from the list of folders in `srcdir`.
    Assumes that the folders look like `ramses_out_X` and extract the `X`
    integer. Removes `5511` from the list of IDs.

    Parameters
    ----------
    srcdir : string
        The folder where CSiBORG simulations are stored.

    Returns
    -------
    ids : 1-dimensional array
        Array of CSiBORG simulation IDs.
    """
    files = glob(join(srcdir, "ramses_out*"))
    # Select only file names
    files = [f.split("/")[-1] for f in files]
    # Remove files with inverted ICs
    files = [f for f in files if "_inv" not in f]
    # Remove the filename with _old
    files = [f for f in files if "OLD" not in f]
    ids = [int(f.split("_")[-1]) for f in files]
    try:
        ids.remove(5511)
    except ValueError:
        pass
    return numpy.sort(ids)


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
        Path to the `n`th CSiBORG simulation.
    """
    return join(srcdir, fname.format(n))


def get_snapshots(simpath):
    """
    Get the list of snapshots for the given IC realisation.

    Parameters
    ----------
    simpath : str
        Path to the CSiBORG IC realisation.

    Returns
    -------
    snapshots : 1-dimensional array
        Array of snapshot IDs.
    """
    # Get all files in simpath that start with output_
    snaps = glob(join(simpath, "output_*"))
    # Take just the last _00XXXX from each file  and strip zeros
    snaps = [int(snap.split('_')[-1].lstrip('0')) for snap in snaps]
    return numpy.sort(snaps)


def get_snapshot_path(Nsnap, simpath):
    """
    Get a path to a CSiBORG IC realisation snapshot.

    Parameters
    ----------
    Nsnap : int
        Snapshot index.
    simpath : str
        Path to the CSiBORG IC realisation.

    Returns
    -------
    snappath : str
        Path to the CSiBORG IC realisation snapshot.
    """
    return join(simpath, "output_{}".format(str(Nsnap).zfill(5)))


def read_info(Nsnap, simpath):
    """
    Read CSiBORG simulation snapshot info.

    Parameters
    ----------
    Nsnap : int
        Snapshot index.
    simpath : str
        Path to the CSiBORG IC realisation.

    Returns
    -------
    info : dict
        Dictionary of info paramaters. Note that both keys and values are
        strings.
    """
    # Open the info file
    snappath = get_snapshot_path(Nsnap, simpath)
    filename = join(snappath, "info_{}.txt".format(str(Nsnap).zfill(5)))
    with open(filename, "r") as f:
        info = f.read().split()
    # Throw anything below ordering line out
    info = numpy.asarray(info[:info.index("ordering")])
    # Get indexes of lines with `=`. Indxs before/after be keys/vals
    eqindxs = numpy.asarray([i for i in range(info.size) if info[i] == '='])

    keys = info[eqindxs - 1]
    vals = info[eqindxs + 1]
    return {key: val for key, val in zip(keys, vals)}


def open_particle(Nsnap, simpath, verbose=True):
    """
    Open particle files to a given CSiBORG simulation.

    Parameters
    ----------
    Nsnap : int
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
    nout = str(Nsnap).zfill(5)
    snappath = get_snapshot_path(Nsnap, simpath)
    ncpu = int(read_info(Nsnap, simpath)["ncpu"])

    if verbose:
        print("Reading in output `{}` with ncpu = `{}`.".format(nout, ncpu))

    # Check whether the unbinding file exists.
    snapdirlist = listdir(snappath)
    unbinding_file = "unbinding_{}.out00001".format(nout)
    if unbinding_file not in snapdirlist:
        raise FileNotFoundError(
            "Couldn't find `{}` in `{}`. Use mergertreeplot.py -h or --help "
            "to print help message.".format(unbinding_file, snappath))

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
            infopath = join(snappath, "info_{}.txt".format(nout))
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
        del ndim, localseed, nstar_tot, mstar_tot, mstar_lost, nsink

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
    Convert `nparts` array to starting indices in a pre-allocated array for
    looping over the CPU number.

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


def read_particle(pars_extract, Nsnap, simpath, verbose=True):
    """
    Read particle files of a simulation at a given snapshot and return
    values of `pars_extract`.

    Parameters
    ----------
    pars_extract : list of str
        Parameters to be extacted.
    Nsnap : int
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
    nparts, partfiles = open_particle(Nsnap, simpath)
    if verbose:
        print("Opened {} particle files.".format(nparts.size))
    ncpu = nparts.size
    # Order in which the particles are written in the FortranFile
    forder = [("x", F32), ("y", F32), ("z", F32),
              ("vx", F32), ("vy", F32), ("vz", F32),
              ("M", F32), ("ID", I32), ("level", I32)]
    fnames = [fp[0] for fp in forder]
    fdtypes = [fp[1] for fp in forder]
    # Check there are no strange parameters
    if isinstance(pars_extract, str):
        pars_extract = [pars_extract]
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


def open_unbinding(cpu, Nsnap, simpath):
    """
    Open particle files to a given CSiBORG simulation. Note that to be
    consistent CPU is incremented by 1.

    Parameters
    ----------
    cpu : int
        The CPU index.
    Nsnap : int
        The index of a redshift snapshot.
    simpath : str
        The complete path to the CSiBORG simulation.

    Returns
    -------
    unbinding : `scipy.io.FortranFile`
        The opened unbinding FortranFile.
    """
    nout = str(Nsnap).zfill(5)
    cpu = str(cpu + 1).zfill(5)
    fpath = join(simpath, "output_{}".format(nout),
                 "unbinding_{}.out{}".format(nout, cpu))

    return FortranFile(fpath)


def read_clumpid(Nsnap, simpath, verbose=True):
    """
    Read clump IDs of halos from unbinding files.

    Parameters
    ----------
    Nsnap : int
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
    nparts, __ = open_particle(Nsnap, simpath, verbose)
    start_ind = nparts_to_start_ind(nparts)
    ncpu = nparts.size

    clumpid = numpy.full(numpy.sum(nparts), numpy.nan, dtype=I32)
    iters = tqdm(range(ncpu)) if verbose else range(ncpu)
    for cpu in iters:
        i = start_ind[cpu]
        j = nparts[cpu]
        ff = open_unbinding(cpu, Nsnap, simpath)
        clumpid[i:i + j] = ff.read_ints()

    return clumpid


def drop_zero_indx(clump_ids, particles):
    """
    Drop from `clump_ids` and `particles` entries whose clump index is 0.

    Parameters
    ----------
    clump_ids : 1-dimensional array
        Array of clump IDs.
    particles : structured array
        Array of the particle data.

    Returns
    -------
    clump_ids : 1-dimensional array
        The array of clump IDs after removing zero clump ID entries.
    particles : structured array
        The particle data after removing zero clump ID entries.
    """
    mask = clump_ids != 0
    return clump_ids[mask], particles[mask]


def read_clumps(Nsnap, simpath, cols=None):
    """
    Read in a clump file `clump_Nsnap.dat`.

    Parameters
    ----------
    Nsnap : int
        The index of a redshift snapshot.
    simpath : str
        The complete path to the CSiBORG simulation.
    cols : list of str, optional.
        Columns to extract. By default `None` and all columns are extracted.

    Returns
    -------
    out : structured array
        Structured array of the clumps.
    """
    Nsnap = str(Nsnap).zfill(5)
    fname = join(simpath, "output_{}".format(Nsnap),
                 "clump_{}.dat".format(Nsnap))
    # Check the file exists.
    if not isfile(fname):
        raise FileExistsError("Clump file `{}` does not exist.".format(fname))

    # Read in the clump array. This is how the columns must be written!
    data = numpy.genfromtxt(fname)
    clump_cols = [("index", I64), ("level", I64), ("parent", I64),
                  ("ncell", F64), ("peak_x", F64), ("peak_y", F64),
                  ("peak_z", F64), ("rho-", F64), ("rho+", F64),
                  ("rho_av", F64), ("mass_cl", F64), ("relevance", F64)]
    out0 = cols_to_structured(data.shape[0], clump_cols)
    for i, name in enumerate(out0.dtype.names):
        out0[name] = data[:, i]
    # If take all cols then return
    if cols is None:
        return out0
    # Make sure we have a list
    cols = [cols] if isinstance(cols, str) else cols
    # Get the indxs of clump_cols to output
    clump_names = [col[0] for col in clump_cols]
    indxs = [None] * len(cols)
    for i, col in enumerate(cols):
        if col not in clump_names:
            raise KeyError("...")
        indxs[i] = clump_names.index(col)
    # Make an array and fill it
    out = cols_to_structured(out0.size, [clump_cols[i] for i in indxs])
    for name in out.dtype.names:
        out[name] = out0[name]

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
