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
from os.path import (join, isfile, isdir)
from glob import glob
from tqdm import tqdm
from warnings import warn
from ..utils import (cols_to_structured)


###############################################################################
#                            Paths manager                                    #
###############################################################################


class CSiBORGPaths:
    """
    Paths manager for CSiBORG IC realisations.

    Parameters
    ----------
    n_sim : int, optional
        CSiBORG IC realisation index. By default not set.
    n_snap : int, optional
        Snapshot index. By default not set.
    srcdir : str, optional
        The file path to the folder where realisations of the ICs are stored.
        By default `/mnt/extraspace/hdesmond/`.
    dumpdir : str, optional
        Path to where files from `run_fit_halos` are stored. By default
        `/mnt/extraspace/rstiskalek/csiborg/`.
    mmain_path : str, optional
        Path to where mmain files are stored. By default
        `/mnt/zfsusers/hdesmond/Mmain`.
    initmatch_path : str, optional
        Path to where match between the first and final snapshot is stored. By
        default `/mnt/extraspace/rstiskalek/csiborg/initmatch/`.
    to_new : bool, optional
        Whether the paths should point to `new` files, for example
        `ramses_out_8452_new`.
    """
    _srcdir = None
    _n_sim = None
    _n_snap = None
    _dumpdir = None
    _mmain_path = None
    _initmatch_path = None
    _to_new = None

    def __init__(self, n_sim=None, n_snap=None, srcdir=None, dumpdir=None,
                 mmain_path=None, initmatch_path=None, to_new=False):
        if srcdir is None:
            srcdir = "/mnt/extraspace/hdesmond/"
        self.srcdir = srcdir
        if dumpdir is None:
            dumpdir = "/mnt/extraspace/rstiskalek/csiborg/"
        self.dumpdir = dumpdir
        if mmain_path is None:
            mmain_path = "/mnt/zfsusers/hdesmond/Mmain"
        self.mmain_path = mmain_path
        if initmatch_path is None:
            initmatch_path = "/mnt/extraspace/rstiskalek/csiborg/initmatch/"
        self.initmatch_path = initmatch_path
        self.to_new = to_new
        if n_sim is not None and n_snap is not None:
            self.set_info(n_sim, n_snap)
        if n_sim is not None:
            self.n_sim = n_sim

    @property
    def srcdir(self):
        """
        Folder where CSiBORG simulations are stored.

        Returns
        -------
        srcdir : int
        """
        return self._srcdir

    @srcdir.setter
    def srcdir(self, srcdir):
        """
        Set `srcdir`, check that the directory exists.
        """
        if not isdir(srcdir):
            raise IOError("Invalid directory `{}`!".format(srcdir))
        self._srcdir = srcdir

    @property
    def dumpdir(self):
        """
        Folder where files from `run_fit_halos` are stored.

        Returns
        -------
        dumpdir : str
        """
        return self._dumpdir

    @property
    def temp_dumpdir(self):
        """
        Temporary dumping directory.

        Returns
        -------
        temp_dumpdir : str
        """
        fpath = join(self.dumpdir, "temp")
        if not isdir(fpath):
            raise IOError("Invalid directory `{}`!".format(fpath))
        return fpath

    @dumpdir.setter
    def dumpdir(self, dumpdir):
        """
        Set `dumpdir`, check that the directory exists.
        """
        if not isdir(dumpdir):
            raise IOError("Invalid directory `{}`!".format(dumpdir))
        self._dumpdir = dumpdir

    @property
    def mmain_path(self):
        """
        Path where mmain files are stored.

        Returns
        -------
        mmain_path : str
        """
        return self._mmain_path

    @mmain_path.setter
    def mmain_path(self, mmain_path):
        """
        Set `mmain_path`, check that the directory exists.
        """
        if not isdir(mmain_path):
            raise IOError("Invalid directory `{}`!".format(mmain_path))
        self._mmain_path = mmain_path

    @property
    def initmatch_path(self):
        """
        Path to where match between the first and final snapshot is stored.

        Returns
        -------
        initmach_path : str
        """
        return self._initmatch_path

    @initmatch_path.setter
    def initmatch_path(self, initmatch_path):
        """
        Set `initmatch_path`, check that the directory exists.
        """
        if not isdir(initmatch_path):
            raise IOError("Invalid directory `{}`!".format(initmatch_path))
        self._initmatch_path = initmatch_path

    @property
    def to_new(self):
        """
        Flag whether paths should point to `new` files, for example
        `ramses_out_8452_new`.

        Returns
        -------
        to_new : bool
        """
        return self._to_new

    @to_new.setter
    def to_new(self, to_new):
        """Set `to_new`."""
        if not isinstance(to_new, bool):
            raise TypeError("`to_new` must be be a bool")
        self._to_new = to_new

    @property
    def n_sim(self):
        """
        The IC realisation index set by the user.

        Returns
        -------
        n_sim : int
        """
        if self._n_sim is None:
            raise ValueError(
                "`self.n_sim` is not set! Either provide a value directly  "
                "or set it using `self.set_info(...)`")
        return self._n_sim

    @n_sim.setter
    def n_sim(self, n_sim):
        """Set `n_sim`, ensure it is a valid simulation index."""
        if n_sim not in self.ic_ids:
            raise ValueError(
                "`{}` is not a valid IC realisation index.".format(n_sim))
        self._n_sim = n_sim

    @property
    def n_snap(self):
        """
        The snapshot index of a IC realisation set by the user.

        Returns
        -------
        n_snap: int
        """
        if self._n_snap is None:
            raise ValueError(
                "`self.n_sim` is not set! Either provide a value directly  "
                "or set it using `self.set_info(...)`")
        return self._n_snap

    @n_snap.setter
    def n_snap(self, n_snap):
        """Set `n_snap`."""
        self._n_snap = n_snap

    def set_info(self, n_sim, n_snap):
        """
        Convenience function for setting `n_sim` and `n_snap`.

        Parameters
        ----------
        n_sim : int
            CSiBORG IC realisation index.
        n_snap : int
            Snapshot index.
        """
        self.n_sim = n_sim
        if n_snap not in self.get_snapshots(n_sim):
            raise ValueError(
                "Invalid snapshot number `{}` for IC realisation `{}`."
                .format(n_snap, n_sim))
        self.n_snap = n_snap

    def reset_info(self):
        """
        Reset `self.n_sim` and `self.n_snap`.
        """
        self._n_sim = None
        self._n_snap = None

    def get_n_sim(self, n_sim):
        """
        Get `n_sim`. If `self.n_sim` return it, otherwise returns `n_sim`.
        """
        if n_sim is None:
            return self.n_sim
        return n_sim

    def get_n_snap(self, n_snap):

        """
        Get `n_snap`. If `self.n_snap` return it, otherwise returns `n_snap`.
        """
        if n_snap is None:
            return self.n_snap
        return n_snap

    @property
    def ic_ids(self):
        """
        CSiBORG initial condition (IC) simulation IDs from the list of folders
        in `self.srcdir`.

        Returns
        -------
        ids : 1-dimensional array
            Array of CSiBORG simulation IDs.
        """
        if self.to_new:
            return self._ic_ids_new
        return self._ic_ids

    @property
    def _ic_ids(self):
        """
        IC simulation IDs.

        Returns
        -------
        ids : 1-dimensional array
        """
        files = glob(join(self.srcdir, "ramses_out*"))
        # Select only file names
        files = [f.split("/")[-1] for f in files]
        # Remove files with inverted ICs
        files = [f for f in files if "_inv" not in f]
        # Remove the new files with z = 70 only
        files = [f for f in files if "_new" not in f]
        # Remove the filename with _old
        files = [f for f in files if "OLD" not in f]
        ids = [int(f.split("_")[-1]) for f in files]
        try:
            ids.remove(5511)
        except ValueError:
            pass
        return numpy.sort(ids)

    @property
    def _ic_ids_new(self):
        """
        ICs simulation IDs denoted as `new` with recoved :math:`z = 70`
        particle information.

        Returns
        -------
        ids : 1-dimensional array
        """
        files = glob(join(self.srcdir, "ramses_out*"))
        # Select only file names
        files = [f.split("/")[-1] for f in files]
        # Only _new files
        files = [f for f in files if "_new" in f]
        # Take the ICs
        ids = [int(f.split("_")[2]) for f in files]
        return numpy.sort(ids)

    def ic_path(self, n_sim=None):
        """
        Path to `n_sim`th CSiBORG IC realisation.

        Parameters
        ----------
        n_sim : int, optional
            The index of the initial conditions (IC) realisation. By default
            `None` and the set value is attempted to be used.

        Returns
        -------
        path : str
        """
        n_sim = self.get_n_sim(n_sim)
        fname = "ramses_out_{}"
        if self.to_new:
            fname += "_new"
        return join(self.srcdir, fname.format(n_sim))

    def get_snapshots(self, n_sim=None):
        """
        List of snapshots for the `n_sim`th IC realisation.

        Parameters
        ----------
        n_sim : int
            The index of the initial conditions (IC) realisation. By default
            `None` and the set value is attempted to be used.

        Returns
        -------
        snapshots : 1-dimensional array
        """
        n_sim = self.get_n_sim(n_sim)
        simpath = self.ic_path(n_sim)
        # Get all files in simpath that start with output_
        snaps = glob(join(simpath, "output_*"))
        # Take just the last _00XXXX from each file  and strip zeros
        snaps = [int(snap.split('_')[-1].lstrip('0')) for snap in snaps]
        return numpy.sort(snaps)

    def get_maximum_snapshot(self, n_sim=None):
        """
        Return the maximum snapshot of an IC realisation.

        Parameters
        ----------
        n_sim : int
            The index of the initial conditions (IC) realisation. By default
            `None` and the set value is attempted to be used.

        Returns
        -------
        maxsnap : float
        """
        n_sim = self.get_n_sim(n_sim)
        return max(self.get_snapshots(n_sim))

    def get_minimum_snapshot(self, n_sim=None):
        """
        Return the maximum snapshot of an IC realisation.

        Parameters
        ----------
        n_sim : int
            The index of the initial conditions (IC) realisation. By default
            `None` and the set value is attempted to be used.

        Returns
        -------
        minsnap : float
        """
        n_sim = self.get_n_sim(n_sim)
        return min(self.get_snapshots(n_sim))

    def clump0_path(self, nsim):
        """
        Path to a single dumped clump's particles. This is expected to point
        to a dictonary whose keys are the clump indices and items structured
        arrays with the clump's particles in the initial snapshot.

        Parameters
        ----------
        nsim : int
            Index of the initial conditions (IC) realisation.

        Returns
        -------
        path : str
        """
        cdir = join(self.dumpdir, "initmatch")
        return join(cdir, "clump_{}_{}.npy".format(nsim, "particles"))

    def snapshot_path(self, n_snap=None, n_sim=None):
        """
        Path to a CSiBORG IC realisation snapshot.

        Parameters
        ----------
        n_snap : int
            Snapshot index. By default `None` and the set value is attempted
            to be used.
        n_sim : str
            Corresponding CSiBORG IC realisation index. By default `None` and
            the set value is attempted to be used.

        Returns
        -------
        snappath : str
        """
        n_snap = self.get_n_snap(n_snap)
        n_sim = self.get_n_sim(n_sim)
        simpath = self.ic_path(n_sim)
        return join(simpath, "output_{}".format(str(n_snap).zfill(5)))


###############################################################################
#                          Fortran readers                                    #
###############################################################################


class ParticleReader:
    """
    Tools to read in particle files alon with their corresponding clumps.

    Parameters
    ----------
    paths : py:class`csiborgtools.read.CSiBORGPaths`
        CSiBORG paths-handling object with set `n_sim` and `n_snap`.
    """
    _paths = None

    def __init__(self, paths):
        self.paths = paths

    @property
    def paths(self):
        """
        The paths-handling object.

        Returns
        -------
        paths : :py:class:`csiborgtools.read.CSiBORGPaths`
        """
        return self._paths

    @paths.setter
    def paths(self, paths):
        """
        Set `paths`. Makes sure it is the right object and `n_sim` and `n_snap`
        are both set.
        """
        if not isinstance(paths, CSiBORGPaths):
            raise TypeError("`paths` must be of type `CSiBORGPaths`.")
        if paths.n_sim is None or paths.n_snap is None:
            raise ValueError(
                "`paths` must have set both `n_sim` and `n_snap`!")
        self._paths = paths

    def read_info(self):
        """
        Read CSiBORG simulation snapshot info.

        Returns
        -------
        info : dict
            Dictionary of info paramaters. Note that both keys and values are
            strings.
        """
        # Open the info file
        n_snap = self.paths.n_snap
        snappath = self.paths.snapshot_path()
        filename = join(snappath, "info_{}.txt".format(str(n_snap).zfill(5)))
        with open(filename, "r") as f:
            info = f.read().split()
        # Throw anything below ordering line out
        info = numpy.asarray(info[:info.index("ordering")])
        # Get indexes of lines with `=`. Indxs before/after be keys/vals
        eqs = numpy.asarray([i for i in range(info.size) if info[i] == '='])

        keys = info[eqs - 1]
        vals = info[eqs + 1]
        return {key: val for key, val in zip(keys, vals)}

    def open_particle(self, verbose=True):
        """
        Open particle files to a given CSiBORG simulation.

        Parameters
        ----------
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        nparts : 1-dimensional array
            Number of parts assosiated with each CPU.
        partfiles : list of `scipy.io.FortranFile`
            Opened part files.
        """
        n_snap = self.paths.n_snap
        # Zeros filled snapshot number and the snapshot path
        nout = str(n_snap).zfill(5)
        snappath = self.paths.snapshot_path()
        ncpu = int(self.read_info()["ncpu"])

        if verbose:
            print("Reading in output `{}` with ncpu = `{}`."
                  .format(nout, ncpu))

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
                raise ValueError(
                    "`ncpu = {}` of `{}` disagrees with `ncpu = {}` "
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

    @staticmethod
    def read_sp(dtype, partfile):
        """
        Utility function to read a single particle file, depending on its
        dtype.

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
        if dtype in [numpy.float16, numpy.float32, numpy.float64]:
            return partfile.read_reals('d')
        elif dtype in [numpy.int32]:
            return partfile.read_ints()
        else:
            raise TypeError("Unexpected dtype `{}`.".format(dtype))

    @staticmethod
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

    def read_particle(self, pars_extract, verbose=True):
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
        nparts, partfiles = self.open_particle(verbose=verbose)
        if verbose:
            print("Opened {} particle files.".format(nparts.size))
        ncpu = nparts.size
        # Order in which the particles are written in the FortranFile
        forder = [("x", numpy.float32), ("y", numpy.float32),
                  ("z", numpy.float32), ("vx", numpy.float32),
                  ("vy", numpy.float32), ("vz", numpy.float32),
                  ("M", numpy.float32), ("ID", numpy.int32),
                  ("level", numpy.int32)]
        fnames = [fp[0] for fp in forder]
        fdtypes = [fp[1] for fp in forder]
        # Check there are no strange parameters
        if isinstance(pars_extract, str):
            pars_extract = [pars_extract]
        for p in pars_extract:
            if p not in fnames:
                raise ValueError(
                    "Undefined parameter `{}`. Must be one of `{}`."
                    .format(p, fnames))

        npart_tot = numpy.sum(nparts)
        # A dummy array is necessary for reading the fortran files.
        dum = numpy.full(npart_tot, numpy.nan, dtype=numpy.float16)
        # These are the data we read along with types
        dtype = {"names": pars_extract,
                 "formats": [forder[fnames.index(p)][1] for p in pars_extract]}
        # Allocate the output structured array
        out = numpy.full(npart_tot, numpy.nan, dtype)
        start_ind = self.nparts_to_start_ind(nparts)
        iters = tqdm(range(ncpu)) if verbose else range(ncpu)
        for cpu in iters:
            i = start_ind[cpu]
            j = nparts[cpu]
            for (fname, fdtype) in zip(fnames, fdtypes):
                if fname in pars_extract:
                    out[fname][i:i + j] = self.read_sp(fdtype, partfiles[cpu])
                else:
                    dum[i:i + j] = self.read_sp(fdtype, partfiles[cpu])
        # Close the fortran files
        for partfile in partfiles:
            partfile.close()

        return out

    def open_unbinding(self, cpu):
        """
        Open particle files to a given CSiBORG simulation. Note that to be
        consistent CPU is incremented by 1.

        Parameters
        ----------
        cpu : int
            The CPU index.

        Returns
        -------
        unbinding : `scipy.io.FortranFile`
            The opened unbinding FortranFile.
        """
        nout = str(self.paths.n_snap).zfill(5)
        cpu = str(cpu + 1).zfill(5)
        fpath = join(self.paths.ic_path(), "output_{}".format(nout),
                     "unbinding_{}.out{}".format(nout, cpu))
        return FortranFile(fpath)

    def read_clumpid(self, verbose=True):
        """
        Read clump IDs of particles from unbinding files.

        Parameters
        ----------
        verbose : bool, optional
            Verbosity flag while for reading the CPU outputs.

        Returns
        -------
        clumpid : 1-dimensional array
            The array of clump IDs.
        """
        nparts, __ = self.open_particle(verbose)
        start_ind = self.nparts_to_start_ind(nparts)
        ncpu = nparts.size

        clumpid = numpy.full(numpy.sum(nparts), numpy.nan, dtype=numpy.int32)
        iters = tqdm(range(ncpu)) if verbose else range(ncpu)
        for cpu in iters:
            i = start_ind[cpu]
            j = nparts[cpu]
            ff = self.open_unbinding(cpu)
            clumpid[i:i + j] = ff.read_ints()
            # Close
            ff.close()

        return clumpid

    @staticmethod
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

    def read_clumps(self, cols=None):
        """
        Read in a clump file `clump_Nsnap.dat`.

        Parameters
        ----------
        cols : list of str, optional.
            Columns to extract. By default `None` and all columns are
            extracted.

        Returns
        -------
        out : structured array
            Structured array of the clumps.
        """
        n_snap = str(self.paths.n_snap).zfill(5)
        fname = join(self.paths.ic_path(), "output_{}".format(n_snap),
                     "clump_{}.dat".format(n_snap))
        # Check the file exists.
        if not isfile(fname):
            raise FileExistsError(
                "Clump file `{}` does not exist.".format(fname))

        # Read in the clump array. This is how the columns must be written!
        data = numpy.genfromtxt(fname)
        clump_cols = [("index", numpy.int64), ("level", numpy.int64),
                      ("parent", numpy.int64), ("ncell", numpy.float64),
                      ("peak_x", numpy.float64), ("peak_y", numpy.float64),
                      ("peak_z", numpy.float64), ("rho-", numpy.float64),
                      ("rho+", numpy.float64), ("rho_av", numpy.float64),
                      ("mass_cl", numpy.float64), ("relevance", numpy.float64)]
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

    cols = [("index", numpy.int64), ("peak_x", numpy.float64),
            ("peak_y", numpy.float64), ("peak_z", numpy.float64),
            ("mass_cl", numpy.float64), ("sub_frac", numpy.float64)]
    out = cols_to_structured(arr.shape[0], cols)
    for i, name in enumerate(out.dtype.names):
        out[name] = arr[:, i]

    return out


def read_initcm(n, srcdir, fname="clump_{}_cm.npy"):
    """
    Read `clump_cm`, i.e. the center of mass of a clump at redshift z = 70.
    If the file does not exist returns `None`.

    Parameters
    ----------
    n : int
        The index of the initial conditions (IC) realisation.
    srcdir : str
        The path to the folder containing the files.
    fname : str, optional
        The file name convention.  By default `clump_cm_{}.npy`, where the
        substituted value is `n`.

    Returns
    -------
    out : structured array
    """
    fpath = join(srcdir, fname.format(n))
    try:
        return numpy.load(fpath)
    except FileNotFoundError:
        warn("File {} does not exist.".format(fpath))
        return None


def halfwidth_select(hw, particles):
    """
    Select particles that in a cube of size `2 hw`, centered at the origin.
    Note that this directly modifies the original array and throws away
    particles outside the central region.

    Parameters
    ----------
    hw : float
        Central region halfwidth.
    particles : structured array
        Particle array with keys `x`, `y`, `z`.

    Returns
    -------
    particles : structured array
        The modified particle array.
    """
    assert 0 < hw < 0.5
    mask = ((0.5 - hw < particles['x']) & (particles['x'] < 0.5 + hw)
            & (0.5 - hw < particles['y']) & (particles['y'] < 0.5 + hw)
            & (0.5 - hw < particles['z']) & (particles['z'] < 0.5 + hw))
    # Subselect the particles
    particles = particles[mask]
    # Rescale to range [0, 1]
    for p in ('x', 'y', 'z'):
        particles[p] = (particles[p] - 0.5 + hw) / (2 * hw)
    return particles
