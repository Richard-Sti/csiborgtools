# Copyright (C) 2023 Richard Stiskalek
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
Script to process simulation snapshots to sorted HDF5 files. Be careful
because reading the HDF5 file may require `hdf5plugin` package to be installed.
The snapshot particles are sorted by their halo ID, so that particles of a halo
can be accessed by slicing the array.

CSiBORG1 reader will complain unless it can find the halomaker FOF files
where it expects them:
    fdir = f"/mnt/extraspace/rstiskalek/csiborg1/chain_{self.nsim}/FOF"
"""
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import datetime
from gc import collect
from glob import glob, iglob
from os import makedirs
from os.path import basename, dirname, exists, join
from warnings import catch_warnings, filterwarnings, warn

import hdf5plugin
import numpy
import pynbody
import readgadget
from astropy import constants, units
from h5py import File
from numba import jit
from readfof import FoF_catalog
from tqdm import tqdm, trange

MSUNCGS = constants.M_sun.cgs.value
BLOSC_KWARGS = {"cname": "blosclz",
                "clevel": 9,
                "shuffle": hdf5plugin.Blosc.SHUFFLE,
                }


###############################################################################
#                               Utility functions                             #
###############################################################################


def now():
    """
    Return current time.
    """
    return datetime.now()


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


def cols_to_structured(N, cols):
    """
    Allocate a structured array from `cols`, a list of (name, dtype) tuples.
    """
    if not (isinstance(cols, list)
            and all(isinstance(c, tuple) and len(c) == 2 for c in cols)):
        raise TypeError("`cols` must be a list of (name, dtype) tuples.")

    names, formats = zip(*cols)
    dtype = {"names": names, "formats": formats}

    return numpy.full(N, numpy.nan, dtype=dtype)


def copy_hdf5_file(src_file, dest_file, exclude_headers=None):
    """
    Make a copy of an HDF5 file, excluding the specified headers.

    Parameters
    ----------
    src_file : str
        Path to the source file.
    dest_file : str
        Path to the destination file.
    exclude_headers : str or list of str
        Name of the headers to exclude, optional.

    Returns
    -------
    None
    """
    if exclude_headers is None:
        exclude_headers = []
    if isinstance(exclude_headers, str):
        exclude_headers = [exclude_headers]

    with File(src_file, 'r') as src, File(dest_file, 'w') as dest:
        # Copying all groups and datasets except the ones in exclude_headers
        for name, item in src.items():
            if name not in exclude_headers:
                src.copy(item, dest)


###############################################################################
#                       Base reader of snapshots                              #
###############################################################################


class BaseReader(ABC):
    """Base reader layout that every subsequent reader should follow."""
    @abstractmethod
    def read_info(self):
        pass

    @abstractmethod
    def read_snapshot(self, kind):
        pass

    @abstractmethod
    def read_halo_id(self, pids):
        pass

    @abstractmethod
    def read_halos(self):
        pass


###############################################################################
#                       CSiBORG particle reader                               #
###############################################################################


class CSiBORG1Reader:
    """
    Object to read in CSiBORG snapshots from the binary files and halo
    catalogues.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    which_snapshot : str
        Which snapshot to read. Options are `initial` or `final`.
    """
    def __init__(self, nsim, which_snapshot):
        self.nsim = nsim
        base_dir = "/mnt/extraspace/hdesmond/"

        if which_snapshot == "initial":
            self.nsnap = 1
            self.source_dir = f"/mnt/extraspace/rstiskalek/csiborg1/initial/ramses_out_{nsim}_new/output_00001"  # noqa
        elif which_snapshot == "final":
            sourcedir = join(base_dir, f"ramses_out_{nsim}")
            self.nsnap = max([int(basename(f).replace("output_", ""))
                              for f in glob(join(sourcedir, "output_*"))])
            self.source_dir = join(sourcedir,
                                   f"output_{str(self.nsnap).zfill(5)}")
        else:
            raise ValueError(f"Unknown snapshot option `{which_snapshot}`.")

        self.output_dir = f"/mnt/extraspace/rstiskalek/csiborg1/chain_{self.nsim}"  # noqa
        self.output_snap = join(self.output_dir,
                                f"snapshot_{str(self.nsnap).zfill(5)}.hdf5")
        self.output_cat = join(self.output_dir,
                               f"fof_{str(self.nsnap).zfill(5)}.hdf5")
        self.halomaker_dir = join(self.output_dir, "FOF")

        self.sph_file = f"/mnt/extraspace/rstiskalek/csiborg1/sph_temp/chain_{self.nsim}.hdf5"  # noqa

    def read_info(self):
        filename = glob(join(self.source_dir, "info_*.txt"))
        if len(filename) > 1:
            raise ValueError("Found too many `info` files.")
        filename = filename[0]

        with open(filename, "r") as f:
            info = f.read().split()
        # Throw anything below ordering line out
        info = numpy.asarray(info[:info.index("ordering")])
        # Get indexes of lines with `=`. Indxs before/after be keys/vals
        eqs = numpy.asarray([i for i in range(info.size) if info[i] == '='])

        keys = info[eqs - 1]
        vals = info[eqs + 1]
        return {key: convert_str_to_num(val) for key, val in zip(keys, vals)}

    def read_snapshot(self, kind):
        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            sim = pynbody.load(self.source_dir)

        info = self.read_info()

        if kind == "pid":
            x = numpy.array(sim["iord"], dtype=numpy.uint32)
        elif kind == "pos":
            x = numpy.array(sim[kind], dtype=numpy.float32)
            # Convert box units to Mpc / h
            box2mpc = (info["unit_l"] / units.kpc.to(units.cm) / info["aexp"]
                       * 1e-3 * info["H0"] / 100)
            x *= box2mpc
        elif kind == "mass":
            x = numpy.array(sim[kind], dtype=numpy.float32)
            # Convert box units to Msun / h
            box2msun = (info["unit_d"] * info["unit_l"]**3 / MSUNCGS
                        * info["H0"] / 100)
            x *= box2msun
        elif kind == "vel":
            x = numpy.array(sim[kind], dtype=numpy.float16)
            # Convert box units to km / s
            box2kms = (1e-2 * info["unit_l"] / info["unit_t"] / info["aexp"]
                       * 1e-3)
            x *= box2kms
        else:
            raise ValueError(f"Unknown kind `{kind}`. "
                             "Options are: `pid`, `pos`, `vel` or `mass`.")

        del sim
        collect()

        return x

    def read_halo_id(self, pids):
        fpath = join(self.halomaker_dir, "*particle_membership*")
        fpath = next(iglob(fpath, recursive=True), None)
        if fpath is None:
            raise FileNotFoundError(f"Found no Halomaker files in `{self.halomaker_dir}`.")  # noqa

        print(f"{now()}: mapping particle IDs to their indices.")
        pids_idx = {pid: i for i, pid in enumerate(pids)}

        # Unassigned particle IDs are assigned a halo ID of 0.
        print(f"{now()}: mapping HIDs to their array indices.")
        hids = numpy.zeros(pids.size, dtype=numpy.int32)

        # Read line-by-line to avoid loading the whole file into memory.
        with open(fpath, 'r') as file:
            for line in tqdm(file, desc="Reading membership"):
                hid, pid = map(int, line.split())
                hids[pids_idx[pid]] = hid

        del pids_idx
        collect()

        return hids

    def read_halos(self):
        info = self.read_info()
        h = info["H0"] / 100

        fpath = join(self.halomaker_dir, "fort.132")
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

        out["totpartmass"] = totmass * 1e11 * h
        out["m200c"] = m200c * 1e11 * h

        return out


###############################################################################
#                         CSiBORG2 particle reader                            #
###############################################################################


class CSiBORG2Reader(BaseReader):
    """
    Object to read in CSiBORG2 snapshots. Because this is Gadget4 the final
    snapshot is already sorted, however we still have to sort the initial
    snapshot.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    which_snapshot : str
        Which snapshot to read. Options are `initial` or `final`.
    """
    def __init__(self, nsim, which_snapshot, kind):
        self.nsim = nsim
        if kind not in ["main", "random", "varysmall"]:
            raise ValueError(f"Unknown kind `{kind}`.")
        self.base_dir = f"/mnt/extraspace/rstiskalek/csiborg2_{kind}"

        if kind == "varysmall":
            self.nsim = f"16417_{str(self.nsim).zfill(3)}"

        if which_snapshot == "initial":
            self.nsnap = 0
        elif which_snapshot == "final":
            self.nsnap = 99
        else:
            raise ValueError(f"Unknown snapshot option `{which_snapshot}`.")

        self.source_dir = join(
                self.base_dir, f"chain_{self.nsim}", "output",
                f"snapshot_{str(self.nsnap).zfill(3)}_full.hdf5")
        if which_snapshot == "initial":
            self.source_dir = self.source_dir.replace("_full.hdf5", ".hdf5")

        self.output_dir = join(self.base_dir, f"chain_{self.nsim}", "output")
        self.output_snap = join(
            self.output_dir,
            f"snapshot_{str(self.nsnap).zfill(3)}_sorted.hdf5")
        self.output_cat = None

        self.offset_path = join(
            self.base_dir, f"chain_{self.nsim}", "output",
            f"fof_subhalo_tab_{str(self.nsnap).zfill(3)}_full.hdf5")

    def read_info(self):
        fpath = join(dirname(self.source_dir), "snapshot_99_full.hdf5")

        with File(fpath, 'r') as f:
            header = f["Header"]
            params = f["Parameters"]

            out = {"BoxSize": header.attrs["BoxSize"],
                   "MassTable": header.attrs["MassTable"],
                   "NumPart_Total": header.attrs["NumPart_Total"],
                   "Omega_m": params.attrs["Omega0"],
                   "Omega_l": params.attrs["OmegaLambda"],
                   "Omega_b": params.attrs["OmegaBaryon"],
                   "h": params.attrs["HubbleParam"],
                   "redshift": header.attrs["Redshift"],
                   }
        return out

    def _get_particles(self, kind):
        with File(self.source_dir, "r") as f:
            if kind == "Masses":
                npart = f["Header"].attrs["NumPart_Total"][1]
                x_high = numpy.ones(npart, dtype=numpy.float32)
                x_high *= f["Header"].attrs["MassTable"][1]
            else:
                x_high = f[f"PartType1/{kind}"][...]

            x_low = f[f"PartType5/{kind}"][...]

        return x_high, x_low

    def read_snapshot(self, kind):
        if kind == "pid":
            x_high, x_low = self._get_particles("ParticleIDs")
        elif kind == "pos":
            x_high, x_low = self._get_particles("Coordinates")
        elif kind == "mass":
            x_high, x_low = self._get_particles("Masses")
        elif kind == "vel":
            x_high, x_low = self._get_particles("Velocities")
        else:
            raise ValueError(f"Unknown kind `{kind}`. "
                             "Options are: `pid`, `pos`, `vel` or `mass`.")

        return x_high, x_low

    def read_halo_id(self, pids):
        raise RuntimeError("TODO Not implemented.")

    def read_halos(self):
        raise RuntimeError("TODO Not implemented.")


###############################################################################
#                         Quijote particle reader                             #
###############################################################################


class QuijoteReader:
    """
    Object to read in Quijote snapshots from the binary files.

    Parameters
    ----------
    nsim : int
        IC realisation index.
    which_snapshot : str
        Which snapshot to read. Options are `initial` or `final`.
    """
    def __init__(self, nsim, which_snapshot):
        self.nsim = nsim
        quijote_dir = "/mnt/extraspace/rstiskalek/quijote"

        if which_snapshot == "initial":
            self.nsnap = -1
            snap_str = "ICs"
            self.source_dir = join(quijote_dir, "Snapshots_fiducial",
                                   str(nsim), "ICs", "ics")
        elif which_snapshot == "final":
            self.nsnap = 4
            snap_str = str(self.nsnap).zfill(3)
            self.source_dir = join(
                quijote_dir, "Snapshots_fiducial",
                str(nsim), f"snapdir_{snap_str}", f"snap_{snap_str}")
        else:
            raise ValueError(f"Unknown snapshot option `{which_snapshot}`.")

        self.fof_dir = join(quijote_dir, "Halos_fiducial", str(nsim))
        self.output_dir = f"/mnt/extraspace/rstiskalek/quijote/fiducial_processed/chain_{self.nsim}"  # noqa
        self.output_snap = join(self.output_dir, f"snapshot_{snap_str}.hdf5")
        self.output_cat = join(self.output_dir, f"fof_{snap_str}.hdf5")

    def read_info(self):
        header = readgadget.header(self.source_dir)
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

    def read_snapshot(self, kind):
        info = self.read_info()
        ptype = [1]  # DM

        if kind == "pid":
            return readgadget.read_block(self.source_dir, "ID  ", ptype)
        elif kind == "pos":
            pos = readgadget.read_block(self.source_dir, "POS ", ptype) / 1e3
            return pos.astype(numpy.float32)
        elif kind == "vel":
            vel = readgadget.read_block(self.source_dir, "VEL ", ptype)
            vel = vel.astype(numpy.float16)
            vel *= (1 + info["redshift"])  # km / s
            return vel
        elif kind == "mass":
            return numpy.full(info["Nall"], info["PartMass"],
                              dtype=numpy.float32)
        else:
            raise ValueError(f"Unknown kind `{kind}`. "
                             "Options are: `pid`, `pos`, `vel` or `mass`.")

    def read_halo_id(self, pids):
        cat = FoF_catalog(self.fof_dir, self.nsnap)

        group_pids = cat.GroupIDs
        group_len = cat.GroupLen

        # Create a mapping from particle ID to FoF group ID.
        print(f"{now()}: mapping particle IDs to their indices.")
        ks = numpy.insert(numpy.cumsum(group_len), 0, 0)
        with catch_warnings():
            # Ignore because we are casting NaN as integer.
            filterwarnings("ignore", category=RuntimeWarning)
            pid2hid = numpy.full((group_pids.size, 2), numpy.nan,
                                 dtype=numpy.uint64)
        for i, (k0, kf) in enumerate(zip(ks[:-1], ks[1:])):
            pid2hid[k0:kf, 0] = i + 1
            pid2hid[k0:kf, 1] = group_pids[k0:kf]
        pid2hid = {pid: hid for hid, pid in pid2hid}

        # Create the final array of hids matchign the snapshot array.
        # Unassigned particles have hid 0.
        print(f"{now()}: mapping HIDs to their array indices.")
        hids = numpy.full(pids.size, 0, dtype=numpy.uint32)
        for i in trange(pids.size):
            hids[i] = pid2hid.get(pids[i], 0)

        return hids

    def read_halos(self):
        fof = FoF_catalog(self.fof_dir, self.nsnap, long_ids=False, swap=False,
                          SFR=False, read_IDs=False)

        cols = [("x", numpy.float32),
                ("y", numpy.float32),
                ("z", numpy.float32),
                ("vx", numpy.float32),
                ("vy", numpy.float32),
                ("vz", numpy.float32),
                ("GroupMass", numpy.float32),
                ("npart", numpy.int32),
                ("index", numpy.int32)
                ]
        data = cols_to_structured(fof.GroupLen.size, cols)

        pos = fof.GroupPos / 1e3
        vel = fof.GroupVel * (1 + self.read_info()["redshift"])
        for i, p in enumerate(["x", "y", "z"]):
            data[p] = pos[:, i]
            data[f"v{p}"] = vel[:, i]
        data["GroupMass"] = fof.GroupMass * 1e10
        data["npart"] = fof.GroupLen
        # We want to start indexing from 1. Index 0 is reserved for
        # particles unassigned to any FoF group.
        data["index"] = 1 + numpy.arange(data.size, dtype=numpy.uint32)
        return data


###############################################################################
#                            Group Offsets                                    #
###############################################################################


@jit(nopython=True, boundscheck=False)
def minmax_halo(hid, halo_ids, start_loop=0):
    """
    Find the start and end index of a halo in a sorted array of halo IDs.
    This is much faster than using `numpy.where` and then `numpy.min` and
    `numpy.max`.
    """
    start = None
    end = None

    for i in range(start_loop, halo_ids.size):
        n = halo_ids[i]
        if n == hid:
            if start is None:
                start = i
            end = i
        elif n > hid:
            break
    return start, end


def make_offset_map(part_hids):
    """
    Make group offsets for a list of particles' halo IDs. This is a
    2-dimensional array, where the first column is the halo ID, the second
    column is the start index of the halo in the particle list, and the third
    index is the end index of the halo in the particle list. The start index is
    inclusive, while the end index is exclusive.
    """
    unique_halo_ids = numpy.unique(part_hids)
    unique_halo_ids = unique_halo_ids[unique_halo_ids != 0]
    with catch_warnings():
        filterwarnings("ignore", category=RuntimeWarning)
        halo_map = numpy.full((unique_halo_ids.size, 3), numpy.nan,
                              dtype=numpy.uint32)
    start_loop, niters = 0, unique_halo_ids.size
    for i in trange(niters):
        hid = unique_halo_ids[i]
        k0, kf = minmax_halo(hid, part_hids, start_loop=start_loop)
        halo_map[i, :] = hid, k0, kf
        start_loop = kf

    return halo_map, unique_halo_ids


###############################################################################
#                Process the final snapshot and sort it by groups             #
###############################################################################


def process_final_snapshot(nsim, simname):
    """
    Read in the snapshot particles, sort them by their halo ID and dump
    into a HDF5 file. Stores the first and last index of each halo in the
    particle array for fast slicing of the array to acces particles of a single
    halo.
    """
    if simname == "csiborg1":
        reader = CSiBORG1Reader(nsim, "final")
    elif simname == "quijote":
        reader = QuijoteReader(nsim, "final")
    else:
        raise RuntimeError(f"Simulation `{simname}` is not supported.")

    if not exists(reader.output_dir):
        makedirs(reader.output_dir)

    print("---- Processing Final Snapshot Information ----")
    print(f"Simulation index:      {nsim}")
    print(f"Simulation name:       {simname}")
    print(f"Output snapshot:       {reader.output_snap}")
    print(f"Output catalogue:      {reader.output_cat}")
    print("-----------------------------------------------")
    print(flush=True)

    # First off load the particle IDs from the raw data.
    pids = reader.read_snapshot("pid")

    # Then, load the halo ids and make sure their ordering is the same as the
    # particle IDs ordering.
    print(f"{now()}: loading HIDs.")
    halo_ids = reader.read_halo_id(pids)
    print(f"{now()}: sorting HIDs.")

    # Get a mask that sorts the halo ids and then write the information to
    # the data files sorted by it.
    sort_indxs = numpy.argsort(halo_ids)
    halo_ids = halo_ids[sort_indxs]

    with File(reader.output_snap, 'w') as f:
        print(f"{now()}: creating dataset `ParticleIDs`...",
              flush=True)
        f.create_dataset("ParticleIDs", data=pids[sort_indxs],
                         **hdf5plugin.Blosc(**BLOSC_KWARGS))
        del pids
        collect()

        print(f"{now()}: creating dataset `Coordinates`...",
              flush=True)
        f.create_dataset(
            "Coordinates", data=reader.read_snapshot("pos")[sort_indxs],
            **hdf5plugin.Blosc(**BLOSC_KWARGS))

        print(f"{now()}: creating dataset `Velocities`...",
              flush=True)
        f.create_dataset(
            "Velocities", data=reader.read_snapshot("vel")[sort_indxs],
            **hdf5plugin.Blosc(**BLOSC_KWARGS))

        print(f"{now()}: creating dataset `Masses`...",
              flush=True)
        f.create_dataset(
            "Masses", data=reader.read_snapshot("mass")[sort_indxs],
            **hdf5plugin.Blosc(**BLOSC_KWARGS))

        if simname == "csiborg1":
            header = f.create_dataset("Header", (0,))
            header.attrs["BoxSize"] = 677.7  # Mpc/h
            header.attrs["Omega0"] = 0.307
            header.attrs["OmegaBaryon"] = 0.0
            header.attrs["OmegaLambda"] = 0.693
            header.attrs["HubleParam"] = 0.6777
            header.attrs["Redshift"] = 0.0
        elif simname == "quijote":
            info = reader.read_info()

            header = f.create_dataset("Header", (0,))
            header.attrs["BoxSize"] = info["BoxSize"]
            header.attrs["Omega0"] = info["Omega_m"]
            header.attrs["OmegaLambda"] = info["Omega_l"]
            header.attrs["OmegaBaryon"] = 0.0
            header.attrs["HubleParam"] = info["h"]
            header.attrs["Redshift"] = info["redshift"]
        else:
            raise ValueError(f"Unknown simname `{simname}`.")

        print(f"{now()}: done with `{reader.output_snap}`.",
              flush=True)

        # Lastly, create the halo mapping and default catalogue.
        print(f"{now()}: writing `{reader.output_cat}`.")
        print(f"{datetime.now()}: creating `GroupOffset`...")
        halo_map, unique_halo_ids = make_offset_map(halo_ids)
        # Dump the halo mapping.
        with File(reader.output_cat, "w") as f:
            f.create_dataset("GroupOffset", data=halo_map)

        # Add the halo finder catalogue
        print(f"{now()}: adding the halo finder catalogue.")
        with File(reader.output_cat, "r+") as f:
            cat = reader.read_halos()
            hid2pos = {hid: i for i, hid in enumerate(unique_halo_ids)}

            for key in cat.dtype.names:
                x = numpy.full(unique_halo_ids.size, numpy.nan,
                               dtype=cat[key].dtype)

                for i in range(len(cat)):
                    j = hid2pos[cat["index"][i]]
                    x[j] = cat[key][i]
                f.create_dataset(key, data=x)


###############################################################################
#               Sort the initial snapshot like the final snapshot             #
###############################################################################


def process_initial_snapshot(nsim, simname):
    """
    Sort the initial snapshot particles according to their final snapshot and
    add them to the final snapshot's HDF5 file.

    Note that there is a specific function for CSiBORG2 because of its Gadget4
    formatting.
    """
    if simname == "csiborg1":
        reader = CSiBORG1Reader(nsim, "initial")
        output_snap_final = CSiBORG1Reader(nsim, "final").output_snap
    elif simname == "quijote":
        reader = QuijoteReader(nsim, "initial")
        output_snap_final = QuijoteReader(nsim, "final").output_snap
    elif "csiborg2" in simname:
        return process_initial_snapshot_csiborg2(nsim, simname)
    else:
        raise RuntimeError(f"Simulation `{simname}` is not supported.")

    print("---- Processing Initial Snapshot Information ----")
    print(f"Simulation index:      {nsim}")
    print(f"Simulation name:       {simname}")
    print(f"Output snapshot:       {reader.output_snap}")
    print(f"Output catalogue:      {reader.output_cat}")
    print("-----------------------------------------------")
    print(flush=True)

    print(f"{now()}: loading and sorting the initial PID.")
    sort_indxs = numpy.argsort(reader.read_snapshot("pid"))

    print(f"{now()}: loading the final particles.")
    with File(output_snap_final, "r") as f:
        sort_indxs_final = f["ParticleIDs"][:]
        f.close()

    print(f"{now()}: sorting the particles according to the final snapshot.")
    sort_indxs_final = numpy.argsort(numpy.argsort(sort_indxs_final))
    sort_indxs = sort_indxs[sort_indxs_final]

    del sort_indxs_final
    collect()

    print(f"{now()}: loading and sorting the initial particle information.")
    pos = reader.read_snapshot("pos")[sort_indxs]
    mass = reader.read_snapshot("mass")[sort_indxs]

    del sort_indxs
    collect()

    # In Quijote some particles are positioned precisely at the edge of the
    # box. Move them to be just inside.
    if simname == "quijote":
        boxsize = reader.read_info()["BoxSize"]
        mask = pos >= boxsize
        if numpy.any(mask):
            spacing = numpy.spacing(pos[mask])
            assert numpy.max(spacing) <= 1e-3
            pos[mask] -= spacing

    print(f"{now()}: dumping particles `{reader.output_snap}`.")
    with File(reader.output_snap, 'w') as f:
        f.create_dataset("Coordinates", data=pos,
                         **hdf5plugin.Blosc(**BLOSC_KWARGS))
        f.create_dataset("Masses", data=mass,
                         **hdf5plugin.Blosc(**BLOSC_KWARGS))


def process_initial_snapshot_csiborg2(nsim, simname):
    """
    Sort the initial snapshot particles according to their final snapshot and
    add them to the final snapshot's HDF5 file.
    """
    if "csiborg2" not in simname:
        raise RuntimeError(f"Simulation `{simname}` is not supported in this CSiBORG2 reader.")  # noqa

    reader_initial = CSiBORG2Reader(nsim, "initial", simname.split("_")[1])
    reader_final = CSiBORG2Reader(nsim, "final", simname.split("_")[1])

    print("---- Processing Initial Snapshot Information ----")
    print(f"Simulation index:      {nsim}")
    print(f"Simulation name:       {simname}")
    print(f"Output snapshot:       {reader_initial.output_snap}")
    print("-------------------------------------------------")
    print(flush=True)

    print(f"{now()}: loading and sorting the initial PID.")
    pids_high, pids_low = reader_initial.read_snapshot("pid")
    sort_indxs_high = numpy.argsort(pids_high)
    sort_indxs_low = numpy.argsort(pids_low)
    del pids_high, pids_low
    collect()

    print(f"{now()}: loading the final particles.")
    with File(reader_final.source_dir, "r") as f:
        sort_indxs_final_high = f["PartType1/ParticleIDs"][:]
        sort_indxs_final_low = f["PartType5/ParticleIDs"][:]

    print(f"{now()}: sorting the particles according to the final snapshot.")
    sort_indxs_final_high = numpy.argsort(numpy.argsort(sort_indxs_final_high))
    sort_indxs_high = sort_indxs_high[sort_indxs_final_high]

    sort_indxs_final_low = numpy.argsort(numpy.argsort(sort_indxs_final_low))
    sort_indxs_low = sort_indxs_low[sort_indxs_final_low]

    del sort_indxs_final_high, sort_indxs_final_low
    collect()

    # Make a copy of the initial snapshot without copying the high- and low-
    # resolution particles.
    print(f"{now()}: loading, sorting and writing the initial particles.")
    src_fname = reader_initial.source_dir
    dest_fname = reader_initial.output_snap
    copy_hdf5_file(src_fname, dest_fname,
                   exclude_headers=["PartType1", "PartType5"])

    kinds = ["Coordinates", "ParticleIDs", "Velocities"]
    with File(dest_fname, 'r+') as dest, File(src_fname, 'r') as src:
        # Write and sort the high-resolution particles
        grp_dest = dest.create_group("PartType1")
        grp_source = src["PartType1"]

        for kind in kinds:
            grp_dest.create_dataset(
                kind, data=grp_source[kind][...][sort_indxs_high],
                **hdf5plugin.Blosc(**BLOSC_KWARGS))

        # Write and sort the low-resolution particles
        grp_dest = dest.create_group("PartType5")
        grp_source = src["PartType5"]

        # Read the data up to the specified index
        for kind in kinds + ["Masses"]:
            grp_dest.create_dataset(
                kind, data=grp_source[kind][...][sort_indxs_low],
                **hdf5plugin.Blosc(**BLOSC_KWARGS))


###############################################################################
#                         Command line interface                              #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser(description="Tool to manage the `raw` simulation data.")  # noqa
    parser.add_argument("--nsim", type=int, required=True,
                        help="Simulation index.")
    parser.add_argument("--simname", type=str, required=True,
                        choices=["csiborg1", "quijote", "csiborg2_main",
                                 "csiborg2_random", "csiborg2_varysmall"],
                        help="Simulation name.")
    parser.add_argument("--mode", type=int, required=True,
                        choices=[0, 1, 2],
                        help="0: process final snapshot, 1: process initial snapshot, 2: process both")  # noqa
    args = parser.parse_args()

    if "csiborg2" in args.simname and args.mode in [0, 2]:
        raise RuntimeError("Processing the final snapshot for CSiBORG2 is not supported.")  # noqa

    if args.simname != "csiborg1" and args.mode == 3:
        raise RuntimeError("Preparing for SPH is only supported for CSiBORG1.")

    if args.mode == 0:
        process_final_snapshot(args.nsim, args.simname)
    elif args.mode == 1:
        process_initial_snapshot(args.nsim, args.simname)
    else:
        process_final_snapshot(args.nsim, args.simname)
        process_initial_snapshot(args.nsim, args.simname)
