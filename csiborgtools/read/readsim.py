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
from abc import ABC, abstractmethod
from datetime import datetime
from gc import collect
from os.path import isfile, join
from warnings import warn

import numpy
import readfof
import readgadget
from scipy.io import FortranFile
from tqdm import tqdm, trange

from .paths import Paths
from .utils import cols_to_structured


class BaseReader(ABC):
    """
    Base class for all readers.
    """
    _paths = None

    @property
    def paths(self):
        """
        Paths manager.

        Parameters
        ----------
        paths : py:class`csiborgtools.read.Paths`
        """
        return self._paths

    @paths.setter
    def paths(self, paths):
        assert isinstance(paths, Paths)
        self._paths = paths

    @abstractmethod
    def read_info(self, nsnap, nsim):
        """
        Read simulation snapshot info.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.

        Returns
        -------
        info : dict
            Dictionary of information paramaters.
        """
        pass

    @abstractmethod
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
            Parameters to be extracted.
        return_structured : bool, optional
            Whether to return a structured array or a 2-dimensional array. If
            the latter, then the order of the columns is the same as the order
            of `pars_extract`. However, enforces single-precision floating
            point format for all columns.
        verbose : bool, optional
            Verbosity flag while for reading in the files.

        Returns
        -------
        out : structured array or 2-dimensional array
            Particle information.
        pids : 1-dimensional array
            Particle IDs.
        """
        pass


###############################################################################
#                       CSiBORG particle reader                               #
###############################################################################


class CSiBORGReader:
    """
    Object to read in CSiBORG snapshots from the binary files and halo
    catalogues.

    Parameters
    ----------
    paths : py:class`csiborgtools.read.Paths`
    """

    def __init__(self, paths):
        self.paths = paths

    def read_info(self, nsnap, nsim):
        snappath = self.paths.snapshot(nsnap, nsim, "csiborg")
        filename = join(snappath, "info_{}.txt".format(str(nsnap).zfill(5)))
        with open(filename, "r") as f:
            info = f.read().split()
        # Throw anything below ordering line out
        info = numpy.asarray(info[:info.index("ordering")])
        # Get indexes of lines with `=`. Indxs before/after be keys/vals
        eqs = numpy.asarray([i for i in range(info.size) if info[i] == '='])

        keys = info[eqs - 1]
        vals = info[eqs + 1]
        return {key: convert_str_to_num(val) for key, val in zip(keys, vals)}

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
        snappath = self.paths.snapshot(nsnap, nsim, "csiborg")
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
        Open particle files of a given CSiBORG simulation. Note that to be
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
        fpath = join(self.paths.snapshots(nsim, "csiborg", tonew=False),
                     f"output_{nsnap}", f"unbinding_{nsnap}.out{cpu}")
        return FortranFile(fpath)

    def read_phew_clumpid(self, nsnap, nsim, verbose=True):
        """
        Read PHEW clump IDs of particles from unbinding files. This halo finder
        was used when running the catalogue.

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

    def read_phew_clups(self, nsnap, nsim, cols=None):
        """
        Read in a PHEW clump file `clump_xxXXX.dat`.

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
        fname = join(self.paths.snapshots(nsim, "csiborg", tonew=False),
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

    def read_fof_hids(self, nsim):
        """
        Read in the FoF particle halo membership IDs that are sorted to match
        the PHEW output.

        Parameters
        ----------
        nsim : int
            IC realisation index.

        Returns
        -------
        hids : 1-dimensional array
            Halo IDs of particles.
        """
        return numpy.load(self.paths.fof_membership(nsim, "csiborg",
                                                    sorted=True))

    def read_fof_halos(self, nsim):
        """
        Read in the FoF halo catalogue.

        Parameters
        ----------
        nsim : int
            IC realisation index.

        Returns
        -------
        cat : structured array
        """
        fpath = self.paths.fof_cat(nsim, "csiborg")
        hid = numpy.genfromtxt(fpath, usecols=0, dtype=numpy.int32)
        pos = numpy.genfromtxt(fpath, usecols=(1, 2, 3), dtype=numpy.float32)
        totmass = numpy.genfromtxt(fpath, usecols=4, dtype=numpy.float32)
        m200c = numpy.genfromtxt(fpath, usecols=5, dtype=numpy.float32)

        dtype = {"names": ["index", "x", "y", "z", "fof_totpartmass",
                           "fof_m200c"],
                 "formats": [numpy.int32] + [numpy.float32] * 5}
        out = numpy.full(hid.size, numpy.nan, dtype=dtype)
        out["index"] = hid
        out["x"] = pos[:, 0]
        out["y"] = pos[:, 1]
        out["z"] = pos[:, 2]
        out["fof_totpartmass"] = totmass * 1e11
        out["fof_m200c"] = m200c * 1e11
        return out


###############################################################################
#                 Summed substructure PHEW catalogue for CSiBORG              #
###############################################################################


class MmainReader:
    """
    Object to generate the summed substructure CSiBORG PHEW catalogue.

    Parameters
    ----------
    paths : :py:class:`csiborgtools.read.Paths`
        Paths objects.
    """
    _paths = None

    def __init__(self, paths):
        assert isinstance(paths, Paths)
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
            Clump array. Read from `CSiBORGReader.read_phew_clups`. Must
            contain `index` and `parent` columns.
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
        substructure. Corresponds to the PHEW Halo finder.

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
        nsnap = max(self.paths.get_snapshots(nsim, "csiborg"))
        partreader = CSiBORGReader(self.paths)
        cols = ["index", "parent", "mass_cl", 'x', 'y', 'z']
        clumparr = partreader.read_phew_clups(nsnap, nsim, cols)

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
#                         Quijote particle reader                             #
###############################################################################


class QuijoteReader:
    """
    Object to read in Quijote snapshots from the binary files.

    Parameters
    ----------
    paths : py:class`csiborgtools.read.Paths`
    """

    def __init__(self, paths):
        self.paths = paths

    def read_info(self, nsnap, nsim):
        snapshot = self.paths.snapshot(nsnap, nsim, "quijote")
        header = readgadget.header(snapshot)
        out = {"BoxSize": header.boxsize / 1e3,       # Mpc/h
               "Nall": header.nall[1],                # Tot num of particles
               "PartMass": header.massarr[1] * 1e10,  # Part mass in Msun/h
               "Omega_m": header.omega_m,
               "Omega_l": header.omega_l,
               "h": header.hubble,
               "redshift": header.redshift,
               }
        out["Hubble"] = (100.0 * numpy.sqrt(
            header.omega_m * (1.0 + header.redshift)**3 + header.omega_l))
        return out

    def read_particle(self, nsnap, nsim, pars_extract=None,
                      return_structured=True, verbose=True):
        assert pars_extract in [None, "pids"]
        snapshot = self.paths.snapshot(nsnap, nsim, "quijote")
        info = self.read_info(nsnap, nsim)
        ptype = [1]  # DM in Gadget speech

        if verbose:
            print(f"{datetime.now()}: reading particle IDs.")
        pids = readgadget.read_block(snapshot, "ID  ", ptype)

        if pars_extract == "pids":
            return None, pids

        if return_structured:
            dtype = {"names": ['x', 'y', 'z', 'vx', 'vy', 'vz', 'M'],
                     "formats": [numpy.float32] * 7}
            out = numpy.full(info["Nall"], numpy.nan, dtype=dtype)
        else:
            out = numpy.full((info["Nall"], 7), numpy.nan, dtype=numpy.float32)

        if verbose:
            print(f"{datetime.now()}: reading particle positions.")
        pos = readgadget.read_block(snapshot, "POS ", ptype) / 1e3  # Mpc/h
        pos /= info["BoxSize"]  # Box units

        for i, p in enumerate(['x', 'y', 'z']):
            if return_structured:
                out[p] = pos[:, i]
            else:
                out[:, i] = pos[:, i]
        del pos
        collect()

        if verbose:
            print(f"{datetime.now()}: reading particle velocities.")
        # NOTE convert to box units.
        vel = readgadget.read_block(snapshot, "VEL ", ptype)  # km/s
        vel *= (1 + info["redshift"])

        for i, v in enumerate(['vx', 'vy', 'vz']):
            if return_structured:
                out[v] = vel[:, i]
            else:
                out[:, i + 3] = vel[:, i]
        del vel
        collect()

        if verbose:
            print(f"{datetime.now()}: reading particle masses.")
        if return_structured:
            out["M"] = info["PartMass"]
        else:
            out[:, 6] = info["PartMass"]

        return out, pids

    def read_fof_hids(self, nsnap, nsim, verbose=True):
        """
        Read the FoF group membership of particles. Unassigned particles have
        FoF group ID 0.

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
        out : 2-dimensional array of shape `(nparticles, 2)`
            The first column is the group index and the second is the
            particle ID.
        """
        redshift = {4: 0.0, 3: 0.5, 2: 1.0, 1: 2.0, 0: 3.0}.get(nsnap, None)
        if redshift is None:
            raise ValueError(f"Redshift of snapshot {nsnap} is not known.")
        path = self.paths.fof_cat(nsim, "quijote")
        cat = readfof.FoF_catalog(path, nsnap)

        # Read the particle IDs of the snapshot
        __, pids = self.read_particle(nsnap, nsim, pars_extract="pids",
                                      verbose=verbose)

        # Read the FoF particle membership. These are only assigned particles.
        if verbose:
            print(f"{datetime.now()}: reading the FoF particle membership.",
                  flush=True)
        group_pids = cat.GroupIDs
        group_len = cat.GroupLen

        # Create a mapping from particle ID to FoF group ID.
        if verbose:
            print(f"{datetime.now()}: creating the particle to FoF ID to map.",
                  flush=True)
        ks = numpy.insert(numpy.cumsum(group_len), 0, 0)
        pid2hid = numpy.full((group_pids.size, 2), numpy.nan,
                             dtype=numpy.uint32)
        for i, (k0, kf) in enumerate(zip(ks[:-1], ks[1:])):
            pid2hid[k0:kf, 0] = i + 1
            pid2hid[k0:kf, 1] = group_pids[k0:kf]
        pid2hid = {pid: hid for hid, pid in pid2hid}

        # Create the final array of hids matchign the snapshot array.
        # Unassigned particles have hid 0.
        if verbose:
            print(f"{datetime.now()}: creating the final hid array.",
                  flush=True)
        hids = numpy.full(pids.size, 0, dtype=numpy.uint32)
        for i in trange(pids.size) if verbose else range(pids.size):
            hids[i] = pid2hid.get(pids[i], 0)

        return numpy.vstack([hids, pids]).T


###############################################################################
#                       Supplementary reading functions                       #
###############################################################################


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


def load_halo_particles(hid, particles, halo_map, hid2map):
    """
    Load a halo's particles from a particle array. If it is not there, i.e
    halo has no associated particles, return `None`.

    Parameters
    ----------
    hid : int
        Halo ID.
    particles : 2-dimensional array
        Array of particles.
    halo_map : 2-dimensional array
        Array containing start and end indices in the particle array
        corresponding to each halo.
    hid2map : dict
        Dictionary mapping halo IDs to `halo_map` array positions.

    Returns
    -------
    halo_particles : 2-dimensional array
        Particle array of this halo.
    """
    try:
        k0, kf = halo_map[hid2map[hid], 1:]
        return particles[k0:kf + 1, :]
    except KeyError:
        return None


def convert_str_to_num(s):
    """
    Convert a string representation of a number to its appropriate numeric type
    (int or float).

    Parameters
    ----------
    s : str
        The string representation of the number.

    Returns
    -------
    num : int or float
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            warn(f"Cannot convert string '{s}' to number", UserWarning)
            return s
