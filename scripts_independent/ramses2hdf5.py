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
Script to convert a RAMSES snapshot to a compressed HDF5 file. Be careful
because reading the HDF5 file requires `hdf5plugin` package to be installed.
If `halomaker_path` is provided, particles will be sorted by their halo ID to
allow for fast halo lookups.
"""
from argparse import ArgumentParser
from datetime import datetime
from gc import collect
from os.path import exists, join
from warnings import catch_warnings, filterwarnings

from glob import iglob

import hdf5plugin
import numpy
import pynbody
from h5py import File
from numba import jit
from tqdm import tqdm, trange
from glob import glob

from os import makedirs

from os.path import basename, join
from datetime import datetime


def now():
    return datetime.now()


def flip_cols(arr, col1, col2):
    """
    Flip values in columns `col1` and `col2` of a structured array `arr`.
    """
    if col1 not in arr.dtype.names or col2 not in arr.dtype.names:
        raise ValueError(f"Both `{col1}` and `{col2}` must exist in `arr`.")

    arr[col1], arr[col2] = numpy.copy(arr[col2]), numpy.copy(arr[col1])


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
    """
    def __init__(self, nsim):
        self.nsim = nsim

    def read_info(self, snapshot_path):
        filename = glob(join(snapshot_path, "info_*"))
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

    def read_snapshot(self, kind, which_snapshot):
        if which_snapshot == "initial":
            raise RuntimeError("TODO add support.")
        elif which_snapshot == "final":
            base_dir = "/mnt/extraspace/hdesmond/"
            sourcedir = join(base_dir, f"ramses_out_{self.nsim}")
            snap = max([int(basename(f).replace("output_", ""))
                        for f in glob(join(sourcedir, "output_*"))])
            snapshot_path = join(sourcedir, f"output_{str(snap).zfill(5)}")
            pass
        else:
            raise ValueError(f"`which_snapshot` must be either 'initial' or 'final'. Received {which_snapshot}")  # noqa

        with catch_warnings():
            filterwarnings("ignore", category=UserWarning)
            sim = pynbody.load(snapshot_path)

        if kind == "pid":
            x = numpy.array(sim["iord"], dtype=numpy.uint32)
        elif kind == "pos" or kind == "mass":
            x = numpy.array(sim[kind], dtype=numpy.float32)
        elif kind == "vel":
            x = numpy.array(sim[kind], dtype=numpy.float16)
        else:
            raise ValueError(f"Unknown kind `{kind}`. "
                             "Options are: `pid`, `pos`, `vel` or `mass`.")

        # Because of a RAMSES bug x and z are flipped.
        if kind in ["pos", "vel"]:
            print(f"For kind `{kind}` flipping x and z.")
            x[:, [0, 2]] = x[:, [2, 0]]

        del sim
        collect()

        return x

    def read_halo_id(self, pids):
        fdir = f"/mnt/extraspace/rstiskalek/csiborg1/chain_{self.nsim}/FOF"
        fpath = join(fdir, "*particle_membership*")
        fpath = next(iglob(fpath, recursive=True), None)
        if fpath is None:
            raise FileNotFoundError(f"Found no Halomaker files in `{fdir}`.")

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

    def read_halos(self, nsnap, nsim):
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

    # fdir = join(self.postdir, "halo_maker", f"ramses_{nsim}",
    #                     f"output_{str(nsnap).zfill(5)}", "FOF")
    #         try_create_directory(fdir)
    #         return join(fdir, "fort.132")


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


###############################################################################
#                         CSiBORG2 particle reader                             #
###############################################################################


# class CSiBORG2Reader(BaseReader):
#     """
#     Object to read in CSiBORG2 snapshots.
#
#     Parameters
#     ----------
#     paths : py:class`csiborgtools.read.Paths`
#     """
#     def __init__(self, paths):
#         self.paths = paths
#
#     @abstractmethod
#     def read_info(self, nsnap, nsim):
#         """
#         Read simulation snapshot info.
#         """
#         snapshot = self.paths.snapshot(nsnap, nsim, "quijote")
#
#         header = readgadget.header(snapshot)
#         out = {"BoxSize": header.boxsize / 1e3,       # Mpc/h
#                "Nall": header.nall[1],                # Tot num of particles
#                "PartMass": header.massarr[1] * 1e10,  # Part mass in Msun/h
#                "Omega_m": header.omega_m,
#                "Omega_l": header.omega_l,
#                "h": header.hubble,
#                "redshift": header.redshift,
#                }
#         out["TotMass"] = out["Nall"] * out["PartMass"]
#         out["Hubble"] = (100.0 * numpy.sqrt(
#             header.omega_m * (1.0 + header.redshift)**3 + header.omega_l))
#         return out
#
#
#
#     @abstractmethod
#     def read_snapshot(self, nsnap, nsim, kind, sort_like_final=False):
#         """
#         Read snapshot.
#         """
#
#     @abstractmethod
#     def read_halo_id(self, nsnap, nsim, halo_finder, verbose=True):
#         """
#         Read the (sub) halo membership of particles.
#         """
#
#     def read_catalogue(self, nsnap, nsim, halo_finder):
#         """
#         Read in the halo catalogue.
#
#         """


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
            cat = FoF_catalog(path, nsnap)
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


def fprint(msg, verbose=True):
    """Print and flush a message with a timestamp."""
    if verbose:
        print(f"{datetime.now()}:   {msg}", flush=True)


def load_snapshot(snapshot_path, kind):
    """
    Load a RAMSES snapshot and return the requested `kind` data.

    Parameters
    ----------
    snapshot_path : str
        Path to RAMSES snapshot folder.
    kind : str
        Kind of data to load. Options are: `pid`, `pos`, `vel` or `mass`.

    Returns
    -------
    x : np.ndarray
    """
    with catch_warnings():
        filterwarnings("ignore", category=UserWarning)
        sim = pynbody.load(snapshot_path)

    if kind == "pid":
        x = numpy.array(sim["iord"], dtype=numpy.uint32)
    elif kind == "pos" or kind == "mass":
        x = numpy.array(sim[kind], dtype=numpy.float32)
    elif kind == "vel":
        x = numpy.array(sim[kind], dtype=numpy.float16)
    else:
        raise ValueError(f"Unknown kind `{kind}`. "
                         "Options are: `pid`, `pos`, `vel` or `mass`.")

    return x


###############################################################################
#                           Halomaker support                                 #
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
    halo_map = numpy.full((unique_halo_ids.size, 3), numpy.nan,
                          dtype=numpy.uint64)
    start_loop, niters = 0, unique_halo_ids.size
    for i in trange(niters):
        hid = unique_halo_ids[i]
        k0, kf = minmax_halo(hid, part_hids, start_loop=start_loop)
        halo_map[i, :] = hid, k0, kf
        start_loop = kf

    return halo_map

###############################################################################
#                       Main conversion functions                             #
###############################################################################


def convert_to_hdf5(snapshot_path, output_path, halomaker_path=None):
    """
    Convert a RAMSES CSiBORG1 snapshot to a compressed HDF5 file.

    Parameters
    ----------
    snapshot_path : str
        Path to RAMSES snapshot folder.
    output_path : str, optional
        Path to output HDF5 file.
    halomaker_path : str, optional
        Path to HaloMaker particle membership file. Optional, if provided
        particles will be sorted by their halo membership.

    Returns
    -------
    None
    """
    blosc_kwargs = {"cname": "blosclz",
                    "clevel": 9,
                    "shuffle": hdf5plugin.Blosc.SHUFFLE,
                    }

    pids = load_snapshot(snapshot_path, "pid")

    if halomaker_path is not None:
        halo_ids = read_halomaker_id(halomaker_path, pids)
        fprint("sorting HIDs.")
        sort_indxs = numpy.argsort(halo_ids)
        halo_ids = halo_ids[sort_indxs]

    with File(output_path, 'w') as f:
        print(f"{datetime.now()}: creating dataset `ParticleIDs`...",
              flush=True)
        if halomaker_path is not None:
            pids = pids[sort_indxs]
        f.create_dataset("ParticleIDs", data=pids,
                         **hdf5plugin.Blosc(**blosc_kwargs))
        del pids
        collect()

        print(f"{datetime.now()}: creating dataset `Coordinates`...",
              flush=True)
        box2mpch = 677.7
        coord = load_snapshot(snapshot_path, "pos") * box2mpch
        if halomaker_path is not None:
            coord = coord[sort_indxs]
        f.create_dataset("Coordinates", data=coord,
                         **hdf5plugin.Blosc(**blosc_kwargs))
        del coord
        collect()

        print(f"{datetime.now()}: creating dataset `Velocities`...",
              flush=True)
        box2kms = 67682.75228061239
        vel = load_snapshot(snapshot_path, "vel") * box2kms
        if halomaker_path is not None:
            vel = vel[sort_indxs]
        f.create_dataset("Velocities", data=vel,
                         **hdf5plugin.Blosc(**blosc_kwargs))
        del vel
        collect()

        print(f"{datetime.now()}: creating dataset `Masses`...",
              flush=True)
        box2msunh = 2.6543271649678946e+19
        mass = load_snapshot(snapshot_path, "mass") * box2msunh
        if halomaker_path is not None:
            mass = mass[sort_indxs]
        f.create_dataset("Masses", data=mass,
                         **hdf5plugin.Blosc(**blosc_kwargs))

        header = f.create_dataset("Header", (0,))
        header.attrs["BoxSize"] = 677.7  # Mpc/h
        header.attrs["Omega0"] = 0.307
        header.attrs["OmegaBaryon"] = 0.0
        header.attrs["OmegaLambda"] = 0.693
        header.attrs["HubleParam"] = 0.6777

        print(f"{datetime.now()}: done with `{snapshot_path}`.", flush=True)

    if halomaker_path is not None:
        halo_map = make_offset_map(halo_ids)
        # Dump the halo mapping.
        with File(output_path, "r+") as f:
            dset = f["GroupOffset"].create_dataset("halo_map", data=halo_map)
            dset.attrs["header"] = """
            Halo to particle mapping. Columns are HID, start index, end index.
            """
            f.close()


def process_snapshot(nsim, which_snapshot, simname, halo_finder, verbose):
    """
    Read in the snapshot particles, sort them by their halo ID and dump
    into a HDF5 file. Stores the first and last index of each halo in the
    particle array for fast slicing of the array to acces particles of a single
    halo.
    """
    # Determine which simulation and where to read and save files
    if simname == "csiborg":
        partreader = CSiBORG1Reader(nsim)
        output_dir = f"/mnt/extraspace/rstiskalek/csiborg1/chain_{nsim}"
        if which_snapshot == "final":
            base_dir = "/mnt/extraspace/hdesmond/"
            sourcedir = join(base_dir, f"ramses_out_{nsim}")
            snap = max([int(basename(f).replace("output_", ""))
                        for f in glob(join(sourcedir, "output_*"))])
        else:
            raise RuntimeError("TODO")

        fname_out = join(output_dir, f"snapshot_{str(snap).zfill(5)}")
    elif simanme == "quijote":
        partreader = QuijoteReader(nsim)
        output_dir = None
        raise RuntimeError("Not implemented")
        fname_out = None
    elif "csiborg2" in simname:
        raise RuntimeError("CSiBORG2 simulations do not need to be processed.")
    else:
        raise RuntimeError(f"")
    if not exists(output_dir):
        makedirs(output_dir)


    fprint(f"loading HIDs of IC {nsim}.", verbose)
    hids = partreader.read_halo_id(nsnap, nsim, halo_finder, verbose)
    collect()

    fprint(f"sorting HIDs of IC {nsim}.")
    sort_indxs = numpy.argsort(hids)

    with h5py.File(fname, "w") as f:
        group = f.create_group("snapshot_final")
        group.attrs["header"] = "Snapshot data at z = 0."

        fprint("dumping halo IDs.", verbose)
        dset = group.create_dataset("halo_ids", data=hids[sort_indxs])
        dset.attrs["header"] = desc["hid"]
        del hids
        collect()

        fprint("reading, sorting and dumping the snapshot particles.", verbose)
        for kind in ["pos", "vel", "mass", "pid"]:
            x = partreader.read_snapshot(nsnap, nsim, kind)[sort_indxs]

            if simname == "csiborg" and kind == "vel":
                x = box.box2vel(x) if simname == "csiborg" else x

            if simname == "csiborg" and kind == "mass":
                x = box.box2solarmass(x) if simname == "csiborg" else x

            dset = f["snapshot_final"].create_dataset(kind, data=x)
            dset.attrs["header"] = desc[kind]
            del x
            collect()

    del sort_indxs
    collect()

    fprint(f"creating a halo map for IC {nsim}.")
    with h5py.File(fname, "r") as f:
        part_hids = f["snapshot_final"]["halo_ids"][:]
    # We loop over the unique halo IDs and remove the 0 halo ID
    unique_halo_ids = numpy.unique(part_hids)
    unique_halo_ids = unique_halo_ids[unique_halo_ids != 0]
    halo_map = numpy.full((unique_halo_ids.size, 3), numpy.nan,
                          dtype=numpy.uint64)
    start_loop, niters = 0, unique_halo_ids.size
    for i in trange(niters, disable=not verbose):
        hid = unique_halo_ids[i]
        k0, kf = minmax_halo(hid, part_hids, start_loop=start_loop)
        halo_map[i, :] = hid, k0, kf
        start_loop = kf

    # Dump the halo mapping.
    with h5py.File(fname, "r+") as f:
        dset = f["snapshot_final"].create_dataset("halo_map", data=halo_map)
        dset.attrs["header"] = """
        Halo to particle mapping. Columns are HID, start index, end index.
        """
        f.close()

    del part_hids
    collect()

    # Add the halo finder catalogue
    with h5py.File(fname, "r+") as f:
        group = f.create_group("halofinder_catalogue")
        group.attrs["header"] = f"Original {halo_finder} halo catalogue."
        cat = partreader.read_catalogue(nsnap, nsim, halo_finder)

        hid2pos = {hid: i for i, hid in enumerate(unique_halo_ids)}

        for key in cat.dtype.names:
            x = numpy.full(unique_halo_ids.size, numpy.nan,
                           dtype=cat[key].dtype)
            for i in range(len(cat)):
                j = hid2pos[cat["index"][i]]
                x[j] = cat[key][i]
            group.create_dataset(key, data=x)
        f.close()

    # Lastly create the halo catalogue
    with h5py.File(fname, "r+") as f:
        group = f.create_group("halo_catalogue")
        group.attrs["header"] = f"{halo_finder} halo catalogue."
        group.create_dataset("index", data=unique_halo_ids)
        f.close()



if __name__ == "__main__":
    parser = ArgumentParser(description="Convert RAMSES snapshot to a compressed HDF5 file.")  # noqa
    parser.add_argument("--snapshot_path", type=str, required=True,
                        help="Path to RAMSES snapshot folder.")
    parser.add_argument("--output_path", type=str, required=False,
                        default=None,
                        help="Path to output HDF5 file. By default stored in the same folder as the snapshot.")  # noqa
    parser.add_argument("--halomaker_path", type=str, required=False,
                        default=None,
                        help="Path to HaloMaker particle membership file.")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = join(args.snapshot_path, "compressed_snapshot.hdf5")

    for path in [args.snapshot_path, args.halomaker_path]:
        if path is not None and not exists(path):
            raise RuntimeError(f"Stopping! `{path}` does not exist.")

    convert_to_hdf5(args.snapshot_path, args.output_path, args.halomaker_path)
