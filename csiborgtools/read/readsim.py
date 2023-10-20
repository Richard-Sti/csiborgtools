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
from os.path import isfile, join, getsize
from warnings import warn

import numpy
import pynbody
from scipy.io import FortranFile

import readfof
import readgadget
from readfof import FoF_catalog
from tqdm import trange

from ..utils import fprint
from .paths import Paths
from .utils import add_columns, cols_to_structured, flip_cols


class BaseReader(ABC):
    """
    Base class for all readers.
    """
    _paths = None

    @property
    def paths(self):
        """Paths manager."""
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
    def read_snapshot(self, nsnap, nsim, kind, sort_like_final=False):
        """
        Read snapshot.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.
        kind : str
            Information to read. Can be `pid`, `pos`, `vel`, or `mass`.
        sort_like_final : bool, optional
            Whether to sort the particles like the final snapshot.

        Returns
        -------
        n-dimensional array
        """

    @abstractmethod
    def read_halo_id(self, nsnap, nsim, halo_finder, verbose=True):
        """
        Read the (sub) halo membership of particles.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.
        halo_finder : str
            Halo finder used when running the catalogue.

        Returns
        -------
        out : 1-dimensional array of shape `(nparticles, )`
        """

    def read_catalogue(self, nsnap, nsim, halo_finder):
        """
        Read in the halo catalogue.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.
        halo_finder : str
            Halo finder used when running the catalogue.

        Returns
        -------
        structured array
        """


###############################################################################
#                       CSiBORG particle reader                               #
###############################################################################


class CSiBORGReader(BaseReader):
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

    def read_snapshot(self, nsnap, nsim, kind):
        sim = pynbody.load(self.paths.snapshot(nsnap, nsim, "csiborg"))
        if kind == "pid":
            x = numpy.array(sim["iord"], dtype=numpy.uint64)
        elif kind in ["pos", "vel", "mass"]:
            x = numpy.array(sim[kind], dtype=numpy.float32)
        else:
            raise ValueError(f"Unknown kind `{kind}`.")

        # Because of a RAMSES bug x and z are flipped.
        if x in ["pos", "vel"]:
            x[:, [0, 2]] = x[:, [2, 0]]

        return x

    def read_halo_id(self, nsnap, nsim, halo_finder, verbose=True):
        if halo_finder == "PHEW":
            ids = self.read_phew_id(nsnap, nsim, verbose)
        elif halo_finder in ["FOF"]:
            ids = self.read_halomaker_id(nsnap, nsim, halo_finder, verbose)
        else:
            raise ValueError(f"Unknown halo finder `{halo_finder}`.")
        return ids

    def open_particle(self, nsnap, nsim, verbose=True):
        """Open particle files to a given CSiBORG simulation."""
        snappath = self.paths.snapshot(nsnap, nsim, "csiborg")
        ncpu = int(self.read_info(nsnap, nsim)["ncpu"])
        nsnap = str(nsnap).zfill(5)
        fprint(f"reading in output `{nsnap}` with ncpu = `{ncpu}`.", verbose)

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
                infopath = join(snappath, f"info_{nsnap}.txt")
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

    def open_unbinding(self, nsnap, nsim, cpu):
        """Open PHEW unbinding files."""
        nsnap = str(nsnap).zfill(5)
        cpu = str(cpu + 1).zfill(5)
        fpath = join(self.paths.snapshots(nsim, "csiborg", tonew=False),
                     f"output_{nsnap}", f"unbinding_{nsnap}.out{cpu}")
        return FortranFile(fpath)

    def read_phew_id(self, nsnap, nsim, verbose):
        nparts, __ = self.open_particle(nsnap, nsim)
        start_ind = numpy.hstack([[0], numpy.cumsum(nparts[:-1])])
        ncpu = nparts.size

        clumpid = numpy.full(numpy.sum(nparts), numpy.nan, dtype=numpy.int32)
        for cpu in trange(ncpu, disable=not verbose, desc="CPU"):
            i = start_ind[cpu]
            j = nparts[cpu]
            ff = self.open_unbinding(nsnap, nsim, cpu)
            clumpid[i:i + j] = ff.read_ints()
            ff.close()

        return clumpid

    def read_halomaker_id(self, nsnap, nsim, halo_finder, verbose):
        fpath = self.paths.halomaker_particle_membership(
            nsnap, nsim, halo_finder)
        fprint(f"loading particle membership `{fpath}`.", verbose)
        membership = numpy.genfromtxt(fpath, dtype=int)

        fprint("loading particle IDs from the snapshot.", verbose)
        pids = self.read_snapshot(nsnap, nsim, "pid")

        fprint("mapping particle IDs to their indices.", verbose)
        pids_idx = {pid: i for i, pid in enumerate(pids)}
        # Unassigned particle IDs are assigned a halo ID of 0.
        fprint("mapping HIDs to their array indices.", verbose)
        hids = numpy.zeros(pids.size, dtype=numpy.int32)
        for i in trange(membership.shape[0]):
            hid, pid = membership[i]
            hids[pids_idx[pid]] = hid

        return hids

    def read_catalogue(self, nsnap, nsim, halo_finder):
        if halo_finder == "PHEW":
            return self.read_phew_clumps(nsnap, nsim)
        elif halo_finder == "FOF":
            return self.read_fof_halos(nsnap, nsim)
        else:
            raise ValueError(f"Unknown halo finder `{halo_finder}`.")

    def read_fof_halos(self, nsnap, nsim):
        """
        Read in the FoF halo catalogue.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.

        Returns
        -------
        structured array
        """
        info = self.read_info(nsnap, nsim)
        h = info["H0"] / 100

        fpath = self.paths.fof_cat(nsnap, nsim, "csiborg")
        hid = numpy.genfromtxt(fpath, usecols=0, dtype=numpy.int32)
        pos = numpy.genfromtxt(fpath, usecols=(1, 2, 3), dtype=numpy.float32)
        totmass = numpy.genfromtxt(fpath, usecols=4, dtype=numpy.float32)
        m200c = numpy.genfromtxt(fpath, usecols=5, dtype=numpy.float32)

        dtype = {"names": ["index", "x", "y", "z", "totpartmass", "m200c"],
                 "formats": [numpy.int32] + [numpy.float32] * 5}
        out = numpy.full(hid.size, numpy.nan, dtype=dtype)
        out["index"] = hid
        out["x"] = pos[:, 0] * h + 677.7 / 2
        out["y"] = pos[:, 1] * h + 677.7 / 2
        out["z"] = pos[:, 2] * h + 677.7 / 2
        # Because of a RAMSES bug x and z are flipped.
        flip_cols(out, "x", "z")
        out["totpartmass"] = totmass * 1e11 * h
        out["m200c"] = m200c * 1e11 * h
        return out

    def read_phew_clumps(self, nsnap, nsim, find_ultimate_parent=True,
                         verbose=True):
        """
        Read in a PHEW clump file `clump_XXXXX.dat`.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.
        find_ultimate_parent : bool, optional
            Whether to find the ultimate parent halo of every clump.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        structured array
        """
        nsnap = str(nsnap).zfill(5)
        fname = join(self.paths.snapshots(nsim, "csiborg", tonew=False),
                     "output_{}".format(nsnap),
                     "clump_{}.dat".format(nsnap))

        if not isfile(fname) or getsize(fname) == 0:
            raise FileExistsError(f"Clump file `{fname}` does not exist.")

        data = numpy.genfromtxt(fname)

        if data.ndim == 1:
            raise FileExistsError(f"Invalid clump file `{fname}`.")

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

        cols = list(clump_cols.keys())
        dtype = [(col, clump_cols[col][1]) for col in cols]
        out = cols_to_structured(data.shape[0], dtype)
        for col in cols:
            out[col] = data[:, clump_cols[col][0]]

        # Convert to cMpc / h and Msun / h
        out['x'] *= 677.7
        out['y'] *= 677.7
        out['z'] *= 677.7
        # Because of a RAMSES bug x and z are flipped.
        flip_cols(out, "x", "z")
        out["mass_cl"] *= 2.6543271649678946e+19

        if find_ultimate_parent:
            ultimate_parent = self.find_parents(out, verbose)

            summed_mass = numpy.full(out.size, numpy.nan, dtype=numpy.float32)
            for i in range(out.size):
                mask = [ultimate_parent == out["index"][i]]
                summed_mass[i] = numpy.sum(out["mass_cl"][mask])

            add_columns(out,
                        [ultimate_parent, summed_mass],
                        ["ultimate_parent", "summed_mass"]
                        )

        return out

    def find_parents(self, clumparr, verbose=False):
        """
        Find ultimate parent haloes for every PHEW clump.

        Parameters
        ----------
        clumparr : structured array
            Clump array. Must contain `index` and `parent` columns.
        verbose : bool, optional
            Verbosity flag.

        Returns
        -------
        parent_arr : 1-dimensional array of shape `(nclumps, )`
            The ultimate parent halo index of every clump.
        """
        clindex = clumparr["index"]
        parindex = clumparr["parent"]

        # The ultimate parent for every clump
        parent_arr = numpy.zeros(clindex.size, dtype=numpy.int32)
        for i in trange(clindex.size, disable=not verbose,
                        desc="Ultimate clump"):
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


###############################################################################
#                         Quijote particle reader                             #
###############################################################################


class QuijoteReader(BaseReader):
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
        out["TotMass"] = out["Nall"] * out["PartMass"]
        out["Hubble"] = (100.0 * numpy.sqrt(
            header.omega_m * (1.0 + header.redshift)**3 + header.omega_l))
        return out

    def read_snapshot(self, nsnap, nsim, kind):
        snapshot = self.paths.snapshot(nsnap, nsim, "quijote")
        info = self.read_info(nsnap, nsim)
        ptype = [1]  # DM in Gadget speech

        if kind == "pid":
            return readgadget.read_block(snapshot, "ID  ", ptype)
        elif kind == "pos":
            pos = readgadget.read_block(snapshot, "POS ", ptype) / 1e3  # Mpc/h
            pos = pos.astype(numpy.float32)
            pos /= info["BoxSize"]  # Box units
            return pos
        elif kind == "vel":
            vel = readgadget.read_block(snapshot, "VEL ", ptype)
            vel = vel.astype(numpy.float32)
            vel *= (1 + info["redshift"])  # km / s
            return vel
        elif kind == "mass":
            return numpy.full(info["Nall"], info["PartMass"],
                              dtype=numpy.float32)
        else:
            raise ValueError(f"Unsupported kind `{kind}`.")

    def read_halo_id(self, nsnap, nsim, halo_finder, verbose=True):
        if halo_finder == "FOF":
            path = self.paths.fof_cat(nsnap, nsim, "quijote")
            cat = readfof.FoF_catalog(path, nsnap)
            pids = self.read_snapshot(nsnap, nsim, kind="pid")

            # Read the FoF particle membership.
            fprint("reading the FoF particle membership.")
            group_pids = cat.GroupIDs
            group_len = cat.GroupLen

            # Create a mapping from particle ID to FoF group ID.
            fprint("creating the particle to FoF ID to map.")
            ks = numpy.insert(numpy.cumsum(group_len), 0, 0)
            pid2hid = numpy.full(
                (group_pids.size, 2), numpy.nan, dtype=numpy.uint64)
            for i, (k0, kf) in enumerate(zip(ks[:-1], ks[1:])):
                pid2hid[k0:kf, 0] = i + 1
                pid2hid[k0:kf, 1] = group_pids[k0:kf]
            pid2hid = {pid: hid for hid, pid in pid2hid}

            # Create the final array of hids matchign the snapshot array.
            # Unassigned particles have hid 0.
            fprint("creating the final hid array.")
            hids = numpy.full(pids.size, 0, dtype=numpy.uint64)
            for i in trange(pids.size, disable=not verbose):
                hids[i] = pid2hid.get(pids[i], 0)

            return hids
        else:
            raise ValueError(f"Unknown halo finder `{halo_finder}`.")

    def read_catalogue(self, nsnap, nsim, halo_finder):
        if halo_finder == "FOF":
            return self.read_fof_halos(nsnap, nsim)
        else:
            raise ValueError(f"Unknown halo finder `{halo_finder}`.")

    def read_fof_halos(self, nsnap, nsim):
        """
        Read in the FoF halo catalogue.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.

        Returns
        -------
        structured array
        """
        fpath = self.paths.fof_cat(nsnap, nsim, "quijote", False)
        fof = FoF_catalog(fpath, nsnap, long_ids=False, swap=False,
                          SFR=False, read_IDs=False)

        cols = [("x", numpy.float32),
                ("y", numpy.float32),
                ("z", numpy.float32),
                ("vx", numpy.float32),
                ("vy", numpy.float32),
                ("vz", numpy.float32),
                ("group_mass", numpy.float32),
                ("npart", numpy.int32),
                ("index", numpy.int32)
                ]
        data = cols_to_structured(fof.GroupLen.size, cols)

        pos = fof.GroupPos / 1e3
        vel = fof.GroupVel * (1 + self.read_info(nsnap, nsim)["redshift"])
        for i, p in enumerate(["x", "y", "z"]):
            data[p] = pos[:, i]
            data[f"v{p}"] = vel[:, i]
        data["group_mass"] = fof.GroupMass * 1e10
        data["npart"] = fof.GroupLen
        # We want to start indexing from 1. Index 0 is reserved for
        # particles unassigned to any FoF group.
        data["index"] = 1 + numpy.arange(data.size, dtype=numpy.int32)
        return data


###############################################################################
#                         Supplementary functions                             #
###############################################################################


def make_halomap_dict(halomap):
    """
    Make a dictionary mapping halo IDs to their start and end indices in the
    snapshot particle array.
    """
    return {hid: (int(start), int(end)) for hid, start, end in halomap}


def load_halo_particles(hid, particles, hid2map):
    """
    Load a halo's particles from a particle array. If it is not there, i.e
    halo has no associated particles, return `None`.

    Parameters
    ----------
    hid : int
        Halo ID.
    particles : 2-dimensional array
        Array of particles.
    hid2map : dict
        Dictionary mapping halo IDs to `halo_map` array positions.

    Returns
    -------
    parts : 1- or 2-dimensional array
    """
    try:
        k0, kf = hid2map[hid]
        return particles[k0:kf + 1]
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
