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
from os.path import (join, isfile, isdir)
from glob import glob
from warnings import warn
import numpy
from scipy.io import FortranFile
from tqdm import tqdm
from ..utils import (cols_to_structured)


###############################################################################
#                            Paths manager                                    #
###############################################################################


class CSiBORGPaths:
    """
    Paths manager for CSiBORG IC realisations.

    Parameters
    ----------
    srcdir : str, optional
        Path to the folder where CSiBORG simulations are stored.
    dumpdir : str, optional
        Path to the folder where files from `run_fit_halos` are stored.
    mmain_path : str, optional
        Path to folder where mmain files are stored.
    initmatch_path : str, optional
        Path to the folder where particle ID match between the first and final
        snapshot is stored.
    """
    _srcdir = None
    _dumpdir = None
    _mmain_path = None
    _initmatch_path = None

    def __init__(self, srcdir="/mnt/extraspace/hdesmond/",
                 dumpdir="/mnt/extraspace/rstiskalek/csiborg/",
                 mmain_path="/mnt/zfsusers/hdesmond/Mmain",
                 initmatch_path="/mnt/extraspace/rstiskalek/csiborg/initmatch/"):  # noqa
        for path in [srcdir, dumpdir, mmain_path, initmatch_path]:
            self._check_directory(path)
        self._srcdir = srcdir
        self._dumpdir = dumpdir
        self._mmain_path = mmain_path
        self._initmatch_path = initmatch_path

    @staticmethod
    def _check_directory(path):
        if not isdir(path):
            raise IOError("Invalid directory `{}`!".format(path))

    @property
    def srcdir(self):
        """
        Path to the folder where CSiBORG simulations are stored.

        Returns
        -------
        path : str
        """
        return self._srcdir

    @property
    def dumpdir(self):
        """
        Path to the folder where files from `run_fit_halos` are stored.

        Returns
        -------
        path : str
        """
        return self._dumpdir

    @property
    def temp_dumpdir(self):
        """
        Path to a temporary dumping folder.

        Returns
        -------
        path : str
        """
        fpath = join(self.dumpdir, "temp")
        if not isdir(fpath):
            raise IOError("Invalid directory `{}`.".format(fpath))
        return fpath

    @property
    def mmain_path(self):
        """
        Path to the folder where mmain files are stored.

        Returns
        -------
        path : str
        """
        return self._mmain_path

    @property
    def initmatch_path(self):
        """
        Path to the folder where the match between the first and final
        snapshot is stored.

        Returns
        -------
        path : str
        """
        return self._initmatch_path

    def ic_ids(self, tonew):
        """
        CSiBORG IC realisation IDs from the list of folders in `self.srcdir`.

        Parameters
        ----------
        tonew : bool
            If `True`, path to the '_new' ICs is returned.
        Returns
        -------
        ids : 1-dimensional array
        """
        files = glob(join(self.srcdir, "ramses_out*"))
        files = [f.split("/")[-1] for f in files]  # Select only file names
        if tonew:
            files = [f for f in files if "_new" in f]
            ids = [int(f.split("_")[2]) for f in files]  # Take the IC IDs
        else:
            files = [f for f in files if "_inv" not in f]  # Remove inv. ICs
            files = [f for f in files if "_new" not in f]  # Remove _new
            files = [f for f in files if "OLD" not in f]   # Remove _old
            ids = [int(f.split("_")[-1]) for f in files]
            try:
                ids.remove(5511)
            except ValueError:
                pass
        return numpy.sort(ids)

    def ic_path(self, nsim, tonew=False):
        """
        Path to a CSiBORG IC realisation folder.

        Parameters
        ----------
        nsim : int
            IC realisation index.
        tonew : bool, optional
            Whether to return the path to the '_new' IC realisation.

        Returns
        -------
        path : str
        """
        fname = "ramses_out_{}"
        if tonew:
            fname += "_new"
        return join(self.srcdir, fname.format(nsim))

    def get_snapshots(self, nsim):
        """
        List of available snapshots of a CSiBORG IC realisation.

        Parameters
        ----------
        nsim : int
            IC realisation index.

        Returns
        -------
        snapshots : 1-dimensional array
        """
        simpath = self.ic_path(nsim, tonew=False)
        # Get all files in simpath that start with output_
        snaps = glob(join(simpath, "output_*"))
        # Take just the last _00XXXX from each file  and strip zeros
        snaps = [int(snap.split('_')[-1].lstrip('0')) for snap in snaps]
        return numpy.sort(snaps)

    def clump0_path(self, nsim):
        """
        Path to a single dumped clump's particles. Expected to point to a
        dictonary whose keys are the clump indices and items structured
        arrays with the clump's particles in the initial snapshot.

        Parameters
        ----------
        nsim : int
            IC realisation index.

        Returns
        -------
        path : str
        """
        cdir = join(self.dumpdir, "initmatch")
        return join(cdir, "clump_{}_{}.npy".format(nsim, "particles"))

    def snapshot_path(self, nsnap, nsim, tonew=False):
        """
        Path to a CSiBORG IC realisation snapshot.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.
        tonew : bool, optional
            Whether to return the path to the '_new' IC realisation.

        Returns
        -------
        snappath : str
        """
        simpath = self.ic_path(nsim, tonew=tonew)
        return join(simpath, "output_{}".format(str(nsnap).zfill(5)))


###############################################################################
#                          Fortran readers                                    #
###############################################################################


class ParticleReader:
    """
    Shortcut to read in particle files along with their corresponding clumps.

    Parameters
    ----------
    paths : py:class`csiborgtools.read.CSiBORGPaths`
    """
    _paths = None

    def __init__(self, paths):
        assert isinstance(paths, CSiBORGPaths)
        self._paths = paths

    @property
    def paths(self):
        return self._paths

    def read_info(self, nsnap, nsim):
        """
        Read CSiBORG simulation snapshot info.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.

        Returns
        -------
        info : dict
            Dictionary of information paramaters. Note that both keys and
            values are strings.
        """
        snappath = self.paths.snapshot_path(nsnap, nsim)
        filename = join(snappath, "info_{}.txt".format(str(nsnap).zfill(5)))
        with open(filename, "r") as f:
            info = f.read().split()
        # Throw anything below ordering line out
        info = numpy.asarray(info[:info.index("ordering")])
        # Get indexes of lines with `=`. Indxs before/after be keys/vals
        eqs = numpy.asarray([i for i in range(info.size) if info[i] == '='])

        keys = info[eqs - 1]
        vals = info[eqs + 1]
        return {key: val for key, val in zip(keys, vals)}

    def open_particle(self, nsnap, nsim, verbose=True):
        """
        Open particle files to a given CSiBORG simulation.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        nparts : 1-dimensional array
            Number of parts assosiated with each CPU.
        partfiles : list of `scipy.io.FortranFile`
            Opened part files.
        """
        snappath = self.paths.snapshot_path(nsnap, nsim)
        ncpu = int(self.read_info()["ncpu"])
        nsnap = str(nsnap).zfill(5)
        if verbose:
            print("Reading in output `{}` with ncpu = `{}`."
                  .format(nsnap, ncpu))

        # First read the headers. Reallocate arrays and fill them.
        nparts = numpy.zeros(ncpu, dtype=int)
        partfiles = [None] * ncpu
        for cpu in range(ncpu):
            cpu_str = str(cpu + 1).zfill(5)
            fpath = join(snappath, "part_{}.out{}".format(nsnap, cpu_str))

            f = FortranFile(fpath)
            # Read in this order
            ncpuloc = f.read_ints()
            if ncpuloc != ncpu:
                infopath = join(snappath, "info_{}.txt".format(nsnap))
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
        Utility function to read a single particle file.

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
        looping over the CPU number. The starting indices calculated as a
        cumulative sum starting at 0.

        Parameters
        ----------
        nparts : 1-dimensional array
            Number of parts assosiated with each CPU.

        Returns
        -------
        start_ind : 1-dimensional array
        """
        return numpy.hstack([[0], numpy.cumsum(nparts[:-1])])

    def read_particle(self, nsnap, nsim, pars_extract, verbose=True):
        """
        Read particle files of a simulation at a given snapshot and return
        values of `pars_extract`.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.
        pars_extract : list of str
            Parameters to be extacted.
        verbose : bool, optional
            Verbosity flag while for reading the CPU outputs.

        Returns
        -------
        out : structured array
        """
        # Open the particle files
        nparts, partfiles = self.open_particle(nsnap, nsim, verbose=verbose)
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

    def open_unbinding(self, nsnap, nsim, cpu):
        """
        Open particle files to a given CSiBORG simulation. Note that to be
        consistent CPU is incremented by 1.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.
        cpu : int
            The CPU index.

        Returns
        -------
        unbinding : `scipy.io.FortranFile`
            The opened unbinding FortranFile.
        """
        nsnap = str(nsnap).zfill(5)
        cpu = str(cpu + 1).zfill(5)
        fpath = join(self.paths.ic_path(nsim, to_new=False),
                     "output_{}".format(nsnap),
                     "unbinding_{}.out{}".format(nsnap, cpu))
        return FortranFile(fpath)

    def read_clumpid(self, nsnap, nsim, verbose=True):
        """
        Read clump IDs of particles from unbinding files.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.
        verbose : bool, optional
            Verbosity flag while for reading the CPU outputs.

        Returns
        -------
        clumpid : 1-dimensional array
            The array of clump IDs.
        """
        nparts, __ = self.open_particle(nsnap, nsim, verbose)
        start_ind = self.nparts_to_start_ind(nparts)
        ncpu = nparts.size

        clumpid = numpy.full(numpy.sum(nparts), numpy.nan, dtype=numpy.int32)
        iters = tqdm(range(ncpu)) if verbose else range(ncpu)
        for cpu in iters:
            i = start_ind[cpu]
            j = nparts[cpu]
            ff = self.open_unbinding(nsnap, nsim, cpu)
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

    def read_clumps(self, nsnap, nsim, cols=None):
        """
        Read in a clump file `clump_Nsnap.dat`.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.

        cols : list of str, optional.
            Columns to extract. By default `None` and all columns are
            extracted.

        Returns
        -------
        out : structured array
            Structured array of the clumps.
        """
        nsnap = str(nsnap).zfill(5)
        fname = join(self.paths.ic_path(nsim, to_new=False),
                     "output_{}".format(nsnap),
                     "clump_{}.dat".format(nsnap))
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


###############################################################################
#                       Supplementary reading functions                       #
###############################################################################


def read_mmain(nsim, srcdir, fname="Mmain_{}.npy"):
    """
    Read `mmain` numpy arrays of central halos whose mass contains their
    substracture contribution.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    srcdir : str
        Path to the folder containing the files.
    fname : str, optional
        File name convention.  By default `Mmain_{}.npy`, where the
        substituted value is `n`.

    Returns
    -------
    out : structured array
        Array with the central halo information.
    """
    fpath = join(srcdir, fname.format(nsim))
    arr = numpy.load(fpath)

    cols = [("index", numpy.int64), ("peak_x", numpy.float64),
            ("peak_y", numpy.float64), ("peak_z", numpy.float64),
            ("mass_cl", numpy.float64), ("sub_frac", numpy.float64)]
    out = cols_to_structured(arr.shape[0], cols)
    for i, name in enumerate(out.dtype.names):
        out[name] = arr[:, i]

    return out


def read_initcm(nsim, srcdir, fname="clump_{}_cm.npy"):
    """
    Read `clump_cm`, i.e. the center of mass of a clump at redshift z = 70.
    If the file does not exist returns `None`.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    srcdir : str
        Path to the folder containing the files.
    fname : str, optional
        File name convention.  By default `clump_cm_{}.npy`, where the
        substituted value is `nsim`.

    Returns
    -------
    out : structured array
    """
    fpath = join(srcdir, fname.format(nsim))
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
