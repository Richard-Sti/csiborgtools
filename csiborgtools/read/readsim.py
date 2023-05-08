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
from os.path import isfile, join
from warnings import warn

import numpy
from scipy.io import FortranFile
from tqdm import tqdm, trange

from .paths import CSiBORGPaths
from .utils import cols_to_structured

###############################################################################
#                       Fortran particle reader                               #
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
        self.paths = paths

    @property
    def paths(self):
        """
        Paths manager.

        Parameters
        ----------
        paths : py:class`csiborgtools.read.CSiBORGPaths`
        """
        return self._paths

    @paths.setter
    def paths(self, paths):
        assert isinstance(paths, CSiBORGPaths)
        self._paths = paths

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
        # trunk-ignore(ruff/B905)
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
        ncpu = int(self.read_info(nsnap, nsim)["ncpu"])
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
        Read a single particle file.

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

    def read_particle(self, nsnap, nsim, pars_extract, return_structured=True,
                      verbose=True):
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
        return_structured : bool, optional
            Whether to return a structured array or a 2-dimensional array. If
            the latter, then the order of the columns is the same as the order
            of `pars_extract`. However, enforces single-precision floating
            point format for all columns.
        verbose : bool, optional
            Verbosity flag while for reading the CPU outputs.

        Returns
        -------
        out : structured array or 2-dimensional array
            Particle information.
        pids : 1-dimensional array
            Particle IDs.
        """
        # Open the particle files
        nparts, partfiles = self.open_particle(nsnap, nsim, verbose=verbose)
        if verbose:
            print(f"Opened {nparts.size} particle files.")
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
        if "ID" in pars_extract:
            pars_extract.remove("ID")
        for p in pars_extract:
            if p not in fnames:
                raise ValueError(f"Undefined parameter `{p}`.")

        npart_tot = numpy.sum(nparts)
        # A dummy array is necessary for reading the fortran files.
        dum = numpy.full(npart_tot, numpy.nan, dtype=numpy.float16)
        # We allocate the output structured/2D array
        if return_structured:
            # These are the data we read along with types
            formats = [forder[fnames.index(p)][1] for p in pars_extract]
            dtype = {"names": pars_extract, "formats": formats}
            out = numpy.full(npart_tot, numpy.nan, dtype)
        else:
            par2arrpos = {par: i for i, par in enumerate(pars_extract)}
            out = numpy.full((npart_tot, len(pars_extract)), numpy.nan,
                             dtype=numpy.float32)
        pids = numpy.full(npart_tot, numpy.nan, dtype=numpy.int32)

        start_ind = self.nparts_to_start_ind(nparts)
        iters = tqdm(range(ncpu)) if verbose else range(ncpu)
        for cpu in iters:
            i = start_ind[cpu]
            j = nparts[cpu]
            for (fname, fdtype) in zip(fnames, fdtypes):
                single_part = self.read_sp(fdtype, partfiles[cpu])
                if fname == "ID":
                    pids[i:i + j] = single_part
                elif fname in pars_extract:
                    if return_structured:
                        out[fname][i:i + j] = single_part
                    else:
                        out[i:i + j, par2arrpos[fname]] = single_part
                else:
                    dum[i:i + j] = single_part
        # Close the fortran files
        for partfile in partfiles:
            partfile.close()

        return out, pids

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
        fpath = join(self.paths.ic_path(nsim, tonew=False), f"output_{nsnap}",
                     f"unbinding_{nsnap}.out{cpu}")
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

            ff.close()

        return clumpid

    def read_clumps(self, nsnap, nsim, cols=None):
        """
        Read in a clump file `clump_xxXXX.dat`.

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
        """
        nsnap = str(nsnap).zfill(5)
        fname = join(self.paths.ic_path(nsim, tonew=False),
                     "output_{}".format(nsnap),
                     "clump_{}.dat".format(nsnap))
        if not isfile(fname):
            raise FileExistsError("Clump file `{}` does not exist."
                                  .format(fname))
        data = numpy.genfromtxt(fname)
        # How the data is stored in the clump file.
        clump_cols = {"index": (0, numpy.int32),
                      "level": (1, numpy.int32),
                      "parent": (2, numpy.int32),
                      "ncell": (3, numpy.float32),
                      "x": (4, numpy.float32),
                      "y": (5, numpy.float32),
                      "z": (6, numpy.float32),
                      "rho-": (7, numpy.float32),
                      "rho+": (8, numpy.float32),
                      "rho_av": (9, numpy.float32),
                      "mass_cl": (10, numpy.float32),
                      "relevance": (11, numpy.float32),
                      }
        # Return the requested columns.
        cols = [cols] if isinstance(cols, str) else cols
        cols = list(clump_cols.keys()) if cols is None else cols

        dtype = [(col, clump_cols[col][1]) for col in cols]
        out = cols_to_structured(data.shape[0], dtype)
        for col in cols:
            out[col] = data[:, clump_cols[col][0]]
        return out


###############################################################################
#                    Summed substructure catalogue                            #
###############################################################################


class MmainReader:
    """
    Object to generate the summed substructure catalogue.

    Parameters
    ----------
    paths : :py:class:`csiborgtools.read.CSiBORGPaths`
        Paths objects.
    """
    _paths = None

    def __init__(self, paths):
        assert isinstance(paths, CSiBORGPaths)
        self._paths = paths

    @property
    def paths(self):
        return self._paths

    def find_parents(self, clumparr, verbose=False):
        """
        Find ultimate parent haloes for every clump in a final snapshot.

        Parameters
        ----------
        clumparr : structured array
            Clump array. Read from `ParticleReader.read_clumps`. Must contain
            `index` and `parent` columns.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        parent_arr : 1-dimensional array of shape `(nclumps, )`
            The ultimate parent halo index for every clump, i.e. referring to
            its ultimate parent clump.
        """
        clindex = clumparr["index"]
        parindex = clumparr["parent"]

        # The ultimate parent for every clump
        parent_arr = numpy.zeros(clindex.size, dtype=numpy.int32)
        for i in trange(clindex.size) if verbose else range(clindex.size):
            tocont = clindex[i] != parindex[i]  # Continue if not a main halo
            par = parindex[i]  # First we try the parent of this clump
            while tocont:
                # The element of the array corresponding to the parent clump to
                # the one we're looking at
                element = numpy.where(clindex == par)[0][0]
                # We stop if the parent is its own parent, so a main halo. Else
                # move onto the parent of the parent. Eventually this is its
                # own parent and we stop, with ultimate parent=par
                if clindex[element] == clindex[element]:
                    tocont = False
                else:
                    par = parindex[element]
            parent_arr[i] = par

        return parent_arr

    def make_mmain(self, nsim, verbose=False):
        """
        Make the summed substructure catalogue for a final snapshot. Includes
        the position of the parent, the summed mass and the fraction of mass in
        substructure.

        Parameters
        ----------
        nsim : int
            IC realisation index.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        mmain : structured array
            The `mmain` catalogue.
        ultimate_parent : 1-dimensional array of shape `(nclumps,)`
            The ultimate parent halo index for every clump, i.e. referring to
            its ultimate parent clump.
        """
        nsnap = max(self.paths.get_snapshots(nsim))
        partreader = ParticleReader(self.paths)
        cols = ["index", "parent", "mass_cl", 'x', 'y', 'z']
        clumparr = partreader.read_clumps(nsnap, nsim, cols)

        ultimate_parent = self.find_parents(clumparr, verbose=verbose)
        mask_main = clumparr["index"] == clumparr["parent"]
        nmain = numpy.sum(mask_main)
        # Preallocate already the output array
        out = cols_to_structured(
            nmain, [("index", numpy.int32), ("x", numpy.float32),
                    ("y", numpy.float32), ("z", numpy.float32),
                    ("M", numpy.float32), ("subfrac", numpy.float32)])
        out["index"] = clumparr["index"][mask_main]
        # Because for these index == parent
        for p in ('x', 'y', 'z'):
            out[p] = clumparr[p][mask_main]
        # We want a total mass for each halo in ID_main
        for i in range(nmain):
            # Should include the main halo itself, i.e. its own ultimate parent
            out["M"][i] = numpy.sum(
                clumparr["mass_cl"][ultimate_parent == out["index"][i]])

        out["subfrac"] = 1 - clumparr["mass_cl"][mask_main] / out["M"]
        return out, ultimate_parent

###############################################################################
#                       Supplementary reading functions                       #
###############################################################################


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
        warn("File {} does not exist.".format(fpath), UserWarning,
             stacklevel=1)
        return None


def halfwidth_mask(pos, hw):
    """
    Mask of particles in a region of width `2 hw, centered at the origin.

    Parameters
    ----------
    pos : 2-dimensional array of shape `(nparticles, 3)`
        Particle positions, in box units.
    hw : float
        Central region half-width.

    Returns
    -------
    mask : 1-dimensional boolean array of shape `(nparticles, )`
    """
    assert 0 < hw < 0.5
    return numpy.all((0.5 - hw < pos) & (pos < 0.5 + hw), axis=1)


def load_clump_particles(clid, particles, clump_map, clid2map):
    """
    Load a clump's particles from a particle array. If it is not there, i.e
    clump has no associated particles, return `None`.

    Parameters
    ----------
    clid : int
        Clump ID.
    particles : 2-dimensional array
        Array of particles.
    clump_map : 2-dimensional array
        Array containing start and end indices in the particle array
        corresponding to each clump.
    clid2map : dict
        Dictionary mapping clump IDs to `clump_map` array positions.

    Returns
    -------
    clump_particles : 2-dimensional array
        Particle array of this clump.
    """
    try:
        k0, kf = clump_map[clid2map[clid], 1:]
        return particles[k0:kf + 1, :]
    except KeyError:
        return None


def load_parent_particles(hid, particles, clump_map, clid2map, clumps_cat):
    """
    Load a parent halo's particles from a particle array. If it is not there,
    return `None`.

    Parameters
    ----------
    hid : int
        Halo ID.
    particles : 2-dimensional array
        Array of particles.
    clump_map : 2-dimensional array
        Array containing start and end indices in the particle array
        corresponding to each clump.
    clid2map : dict
        Dictionary mapping clump IDs to `clump_map` array positions.
    clumps_cat : :py:class:`csiborgtools.read.ClumpsCatalogue`
        Clumps catalogue.

    Returns
    -------
    halo : 2-dimensional array
        Particle array of this halo.
    """
    clids = clumps_cat["index"][clumps_cat["parent"] == hid]
    # We first load the particles of each clump belonging to this parent
    # and then concatenate them for further analysis.
    clumps = []
    for clid in clids:
        parts = load_clump_particles(clid, particles, clump_map, clid2map)
        if parts is not None:
            clumps.append(parts)
    if len(clumps) == 0:
        return None
    return numpy.concatenate(clumps)
