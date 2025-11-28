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
Classes for reading in snapshots and unifying the snapshot interface. Here
should be implemented things such as flipping x- and z-axes, to make sure that
observed RA-dec can be mapped into the simulation box.
"""
from abc import ABC, abstractmethod
from os.path import join, dirname, basename

import numpy as np
from h5py import File
from glob import glob
from gc import collect
from tqdm import tqdm

from ..params import paths_glamdring, simname2boxsize, simname2Omega_m
from .paths import Paths
from .util import find_boxed
from ..utils import fprint

###############################################################################
#                          Base snapshot class                                #
###############################################################################


class BaseSnapshot(ABC):
    """
    Base class for reading snapshots.
    """
    def __init__(self, nsim, nsnap, paths, keep_snapshot_open=False,
                 flip_xz=False):
        self._keep_snapshot_open = None

        if not isinstance(nsim, (int, np.integer)):
            raise TypeError("`nsim` must be an integer")
        self._nsim = int(nsim)

        if not isinstance(nsnap, (int, np.integer)):
            raise TypeError("`nsnap` must be an integer")
        self._nsnap = int(nsnap)

        if not isinstance(keep_snapshot_open, bool):
            raise TypeError("`keep_snapshot_open` must be a boolean.")
        self._keep_snapshot_open = keep_snapshot_open
        self._snapshot_file = None

        if not isinstance(flip_xz, bool):
            raise TypeError("`flip_xz` must be a boolean.")
        self._flip_xz = flip_xz

        self._paths = paths
        self._hid2offset = None

    @property
    def nsim(self):
        """Simulation index."""
        return self._nsim

    @property
    def nsnap(self):
        """Snapshot index."""
        return self._nsnap

    @property
    def simname(self):
        """Simulation name."""
        if self._simname is None:
            raise ValueError("Simulation name not set.")
        return self._simname

    @property
    def boxsize(self):
        """Simulation boxsize in `cMpc/h`."""
        return simname2boxsize(self.simname)

    @property
    def paths(self):
        """Paths manager."""
        if self._paths is None:
            self._paths = Paths(**paths_glamdring)
        return self._paths

    @property
    def keep_snapshot_open(self):
        """
        Whether to keep the snapshot file open when reading halo particles.
        This is useful for repeated access to the snapshot.
        """
        return self._keep_snapshot_open

    @property
    def flip_xz(self):
        """
        Whether to flip the x- and z-axes to undo the MUSIC bug so that the
        coordinates are consistent with observations.
        """
        return self._flip_xz

    @property
    @abstractmethod
    def coordinates(self):
        """Particle coordinates."""
        pass

    @property
    @abstractmethod
    def velocities(self):
        """Particle velocities."""
        pass

    @property
    @abstractmethod
    def masses(self):
        """Particle masses."""
        pass

    @property
    @abstractmethod
    def particle_ids(self):
        """Particle IDs."""
        pass

    @abstractmethod
    def halo_coordinates(self, halo_id, is_group):
        """
        Halo particle coordinates.

        Parameters
        ----------
        halo_id : int
            Halo ID.
        is_group : bool
            If `True`, return the group coordinates. Otherwise, return the
            subhalo coordinates.

        Returns
        -------
        coords : 2-dimensional array
        """
        pass

    @abstractmethod
    def halo_velocities(self, halo_id, is_group):
        """
        Return the halo particle velocities.

        Parameters
        ----------
        halo_id : int
            Halo ID.
        is_group : bool
            If `True`, return the group velocities. Otherwise, return the
            subhalo velocities.

        Returns
        -------
        vel : 2-dimensional array
        """
        pass

    @abstractmethod
    def halo_masses(self, halo_id, is_group):
        """
        Return the halo particle masses.

        Parameters
        ----------
        halo_id : int
            Halo ID.
        is_group : bool
            If `True`, return the group masses. Otherwise, return the
            subhalo masses.

        Returns
        -------
        mass : 1-dimensional array
        """
        pass

    @property
    def hid2offset(self):
        if self._hid2offset is None:
            self._make_hid2offset()

        return self._hid2offset

    @abstractmethod
    def _make_hid2offset(self):
        pass

    def open_snapshot(self):
        """
        Open the snapshot file, particularly used in the context of loading in
        particles of individual haloes.
        """
        if not self.keep_snapshot_open:
            # Check if the snapshot path is set
            if not hasattr(self, "_snapshot_path"):
                raise RuntimeError("Snapshot path not set.")

            return File(self._snapshot_path, "r")

        # Here if we want to keep the snapshot open
        if self._snapshot_file is None:
            self._snapshot_file = File(self._snapshot_path, "r")

        return self._snapshot_file

    def close_snapshot(self):
        """
        Close the snapshot file opened with `open_snapshot`.
        """
        if not self.keep_snapshot_open:
            return

        if self._snapshot_file is not None:
            self._snapshot_file.close()
            self._snapshot_file = None

    def select_box(self, center, boxwidth):
        """
        Find particle coordinates of particles within a box of size `boxwidth`
        centered on `center`.

        Parameters
        ----------
        center : 1-dimensional array
            Center of the box.
        boxwidth : float
            Width of the box.

        Returns
        -------
        pos : 2-dimensional array
        """
        pos = self.coordinates()
        mask = find_boxed(pos, center, boxwidth, self.boxsize)

        return pos[mask]


###############################################################################
#                          CSiBORG1 snapshot class                            #
###############################################################################


class CSiBORG1Snapshot(BaseSnapshot):
    """
    CSiBORG1 snapshot class with the FoF halo finder particle assignment.
    CSiBORG1 was run with RAMSES. Note that the haloes are defined at z = 0 and
    index from 1.

    Parameters
    ----------
    nsim : int
        Simulation index.
    nsnap : int
        Snapshot index.
    paths : Paths, optional
        Paths object.
    keep_snapshot_open : bool, optional
        Whether to keep the snapshot file open when reading halo particles.
        This is useful for repeated access to the snapshot.
    flip_xz : bool, optional
        Whether to flip the x- and z-axes to undo the MUSIC bug so that the
        coordinates are consistent with observations.
    """
    def __init__(self, nsim, nsnap, paths=None, keep_snapshot_open=False,
                 flip_xz=False):
        super().__init__(nsim, nsnap, paths, keep_snapshot_open, flip_xz)
        self._snapshot_path = self.paths.snapshot(
            self.nsnap, self.nsim, "csiborg1")
        self._simname = "csiborg1"

    def _get_particles(self, kind):
        with File(self._snapshot_path, "r") as f:
            x = f[kind][...]

        if self.flip_xz and kind in ["Coordinates", "Velocities"]:
            x[:, [0, 2]] = x[:, [2, 0]]

        return x

    def coordinates(self):
        return self._get_particles("Coordinates")

    def velocities(self):
        return self._get_particles("Velocities")

    def masses(self):
        return self._get_particles("Masses")

    def particle_ids(self):
        with File(self._snapshot_path, "r") as f:
            ids = f["ParticleIDs"][...]

        return ids

    def _get_halo_particles(self, halo_id, kind, is_group):
        if not is_group:
            raise ValueError("There is no subhalo catalogue for CSiBORG1.")

        f = self.open_snapshot()
        i, j = self.hid2offset.get(halo_id, (None, None))
        if i is None:
            raise ValueError(f"Halo `{halo_id}` not found.")

        x = f[kind][i:j + 1]

        if not self.keep_snapshot_open:
            self.close_snapshot()

        if self.flip_xz and kind in ["Coordinates", "Velocities"]:
            x[:, [0, 2]] = x[:, [2, 0]]

        return x

    def halo_coordinates(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Coordinates", is_group)

    def halo_velocities(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Velocities", is_group)

    def halo_masses(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Masses", is_group)

    def _make_hid2offset(self):
        nsnap = max(self.paths.get_snapshots(self.nsim, "csiborg1"))
        catalogue_path = self.paths.snapshot_catalogue(
            nsnap, self.nsim, "csiborg1")

        with File(catalogue_path, "r") as f:
            offset = f["GroupOffset"][:]

        self._hid2offset = {i: (j, k) for i, j, k in offset}


###############################################################################
#                          CSiBORG2 snapshot class                            #
###############################################################################

class CSiBORG2Snapshot(BaseSnapshot):
    """
    CSiBORG2 snapshot class with the FoF halo finder particle assignment and
    SUBFIND subhalo finder. The simulations were run with Gadget4.

    Parameters
    ----------
    nsim : int
        Simulation index.
    nsnap : int
        Snapshot index.
    kind : str
        CSiBORG2 run kind. One of `main`, `random`, or `varysmall`.
    paths : Paths, optional
        Paths object.
    keep_snapshot_open : bool, optional
        Whether to keep the snapshot file open when reading halo particles.
        This is useful for repeated access to the snapshot.
    flip_xz : bool, optional
        Whether to flip the x- and z-axes to undo the MUSIC bug so that the
        coordinates are consistent with observations.
    """
    def __init__(self, nsim, nsnap, kind, paths=None,
                 keep_snapshot_open=False, flip_xz=False):
        super().__init__(nsim, nsnap, paths, keep_snapshot_open, flip_xz)
        self.kind = kind

        fpath = self.paths.snapshot(self.nsnap, self.nsim,
                                    f"csiborg2_{self.kind}")
        if nsnap == 99:
            fpath = fpath.replace(".hdf5", "_full.hdf5")
        elif nsnap == 0:
            fpath = fpath.replace(".hdf5", "_sorted.hdf5")
        else:
            fpath = fpath.replace(".hdf5", "_cut.hdf5")

        self._snapshot_path = fpath
        self._simname = f"csiborg2_{self.kind}"

    @property
    def kind(self):
        """
        CSiBORG2 run kind.

        Returns
        -------
        str
        """
        return self._kind

    @kind.setter
    def kind(self, value):
        if value not in ["main", "random", "varysmall"]:
            raise ValueError("`kind` must be one of `main`, `random`, or `varysmall`.")  # noqa

        self._kind = value

    def _get_particles(self, kind, high_resolution_only=False):
        with File(self._snapshot_path, "r") as f:
            if kind == "Masses":
                npart = f["Header"].attrs["NumPart_Total"][1]
                x = np.ones(npart, dtype=np.float32)
                x *= f["Header"].attrs["MassTable"][1]
            else:
                x = f[f"PartType1/{kind}"][...]

            if not high_resolution_only:
                if x.ndim == 1:
                    x = np.hstack([x, f[f"PartType5/{kind}"][...]])
                else:
                    x = np.vstack([x, f[f"PartType5/{kind}"][...]])

        if self.flip_xz and kind in ["Coordinates", "Velocities"]:
            x[:, [0, 2]] = x[:, [2, 0]]

        return x

    def coordinates(self, high_resolution_only=False):
        return self._get_particles("Coordinates", high_resolution_only)

    def velocities(self, high_resolution_only=False):
        return self._get_particles("Velocities", high_resolution_only)

    def masses(self, high_resolution_only=False):
        return self._get_particles("Masses", high_resolution_only) * 1e10

    def particle_ids(self, high_resolution_only=False):
        return self._get_particles("ParticleIDs", high_resolution_only)

    def _get_halo_particles(self, halo_id, kind, is_group):
        if not is_group:
            raise RuntimeError("While the CSiBORG2 subhalo catalogue exists, it is not currently implemented.")  # noqa

        f = self.open_snapshot()
        i1, j1 = self.hid2offset["type1"].get(halo_id, (None, None))
        i5, j5 = self.hid2offset["type5"].get(halo_id, (None, None))

        # Check if this is a valid halo
        if i1 is None and i5 is None:
            raise ValueError(f"Halo `{halo_id}` not found.")
        if j1 - i1 == 0 and j5 - i5 == 0:
            raise ValueError(f"Halo `{halo_id}` has no particles.")

        if i1 is not None and j1 - i1 > 0:
            if kind == "Masses":
                x1 = np.ones(j1 - i1, dtype=np.float32)
                x1 *= f["Header"].attrs["MassTable"][1]
            else:
                x1 = f[f"PartType1/{kind}"][i1:j1]

                # Flipping of x- and z-axes
                if self.flip_xz:
                    x1[:, [0, 2]] = x1[:, [2, 0]]

        if i5 is not None and j5 - i5 > 0:
            x5 = f[f"PartType5/{kind}"][i5:j5]

            # Flipping of x- and z-axes
            if self.flip_xz and kind in ["Coordinates", "Velocities"]:
                x5[:, [0, 2]] = x5[:, [2, 0]]

        # Close the snapshot file if we don't want to keep it open
        if not self.keep_snapshot_open:
            self.close_snapshot()

        # Are we stacking high-resolution and low-resolution particles?
        if i5 is None or j5 - i5 == 0:
            return x1

        if i1 is None or j1 - i1 == 0:
            return x5

        if x1.ndim > 1:
            x1 = np.vstack([x1, x5])
        else:
            x1 = np.hstack([x1, x5])

        return x1

    def halo_coordinates(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Coordinates", is_group)

    def halo_velocities(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Velocities", is_group)

    def halo_masses(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Masses", is_group) * 1e10

    def _make_hid2offset(self):
        catalogue_path = self.paths.snapshot_catalogue(
            99, self.nsim, f"csiborg2_{self.kind}")
        with File(catalogue_path, "r") as f:

            offset = f["Group/GroupOffsetType"][:, 1]
            lenghts = f["Group/GroupLenType"][:, 1]
            hid2offset_type1 = {i: (offset[i], offset[i] + lenghts[i])
                                for i in range(len(offset))}

            offset = f["Group/GroupOffsetType"][:, 5]
            lenghts = f["Group/GroupLenType"][:, 5]
            hid2offset_type5 = {i: (offset[i], offset[i] + lenghts[i])
                                for i in range(len(offset))}

        self._hid2offset = {"type1": hid2offset_type1,
                            "type5": hid2offset_type5,
                            }


###############################################################################
#                         CSiBORG2x snapshot class                            #
###############################################################################


class CSiBORG2XSnapshot(BaseSnapshot):
    """
    CSiBORG2X snapshot from the SWIFT N-body simulations provided by Stuart
    based on the Manticore ICs. The snapshots are at `z = 0` and correspond to
    `manticore_2MPP_N128_DES_V1`.

    Parameters
    ----------
    nsim : int
        Simulation index.
    version : int
        Manticore version index.
    paths : Paths, optional
        Paths object.
    keep_snapshot_open : bool, optional
        Whether to keep the snapshot file open when reading halo particles.
        This is useful for repeated access to the snapshot.
    """
    def __init__(self, nsim, version=1, paths=None, keep_snapshot_open=False):
        nsnap = 1
        flip_xz = False
        super().__init__(nsim, nsnap, paths, keep_snapshot_open, flip_xz)

        if version == 1:
            simname = "manticore_2MPP_N128_DES_V1"
        elif version == 2:
            simname = "manticore_2MPP_MULTIBIN_N128_DES_V1"
        else:
            raise ValueError("Invalid Manticore version index.")

        fpath = self.paths.snapshot(self.nsnap, self.nsim, simname)

        self._snapshot_path = fpath
        self._simname = simname

    def _get_particles(self, kind):
        with File(self._snapshot_path, "r") as f:
            h = f["Cosmology"].attrs["h"][0]

            if kind == "Masses":
                npart = f["Header"].attrs["NumPart_Total"][1]
                mpart = f["Header"].attrs["InitialMassTable"][1] * 1e10 * h
                x = np.ones(npart, dtype=np.float32) * mpart
            else:
                x = f[f"DMParticles/{kind}"][...]

            # Convert coordinates to Mpc / h
            if kind == "Coordinates":
                x *= h
        return x

    def coordinates(self):
        return self._get_particles("Coordinates")

    def velocities(self):
        return self._get_particles("Velocities")

    def masses(self):
        return self._get_particles("Masses")

    def particle_ids(self):
        raise NotImplementedError("Recovering particle IDs is not implemented "
                                  "for CSiBORG2X.")

    def _get_halo_particles(self, halo_id, kind, is_group):
        raise NotImplementedError("Recovering halo particles is not "
                                  "implemented for CSiBORG2X.")

    def halo_coordinates(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Coordinates", is_group)

    def halo_velocities(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Velocities", is_group)

    def halo_masses(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Masses", is_group) * 1e10

    def _make_hid2offset(self):
        raise NotImplementedError("Recovering halo offsets is not implemented"
                                  "for CSiBORG2X.")


###############################################################################
#                         CSiBORG3 snapshot class                            #
###############################################################################


class CSiBORG3Snapshot(BaseSnapshot):
    """
    CSiBORG3 snapshot from Gadget4 resimulations of the Manticore ICs, version
    2MPP_MULTIBIN_N128_DES_V2.

    Parameters
    ----------
    nsim : int
        Simulation index.
    nsnap : int
        Snapshot index.
    paths : Paths, optional
        Paths object.
    keep_snapshot_open : bool, optional
        Whether to keep the snapshot file open when reading halo particles.
        This is useful for repeated access to the snapshot.
    fpath_override : str, optional
        Override the snapshot file path with a custom path, if provided.
    """
    def __init__(self, nsim, nsnap, paths=None, keep_snapshot_open=False,
                 fpath_override=None, catalogue_fpath_override=None):
        flip_xz = True
        super().__init__(nsim, nsnap, paths, keep_snapshot_open, flip_xz)
        simname = "csiborg3"

        if fpath_override is None:
            fpath = self.paths.snapshot(self.nsnap, self.nsim, simname)
        else:
            fpath = fpath_override

        self._snapshot_path = fpath
        self._simname = simname

        if fpath_override is not None and catalogue_fpath_override is None:
            # Construct the base path for the catalogue files
            # e.g. from ".../snapdir_004/snapshot_004.23.hdf5"
            # to ".../groups_004/fof_subhalo_tab_004.hdf5"
            snapshot_dir = dirname(self._snapshot_path)
            snapshot_basename = basename(self._snapshot_path)
            groups_dir = snapshot_dir.replace("snapdir", "groups")
            snapshot_file_root = snapshot_basename.split('.')[0]
            catalogue_basename = snapshot_file_root.replace(
                "snapshot", "fof_subhalo_tab") + ".hdf5"
            self._catalogue_path = join(groups_dir, catalogue_basename)
        else:
            self._catalogue_path = catalogue_fpath_override

    def _get_particles(self, kind, high_resolution_only=False):
        # Handle both "snapshot_001.hdf5" and "snapshot_001.0.hdf5" formats
        # Split from the right to check if path has a block number
        parts = self._snapshot_path.rsplit('.', 2)
        if len(parts) == 3 and parts[-2].isdigit() and parts[-1] == 'hdf5':
            # Path has a block number like "snapshot_001.0.hdf5"
            # Reconstruct pattern as "snapshot_001.*.hdf5"
            pattern = f"{parts[0]}.*.hdf5"
        else:
            # Path is like "snapshot_001.hdf5", just add wildcard
            pattern = self._snapshot_path.replace(".hdf5", ".*.hdf5")

        files = glob(pattern)
        if len(files) == 0:
            raise FileNotFoundError(
                f"No files found for snapshot {self.nsnap} with pattern {pattern}.")
        files = sorted(files, key=lambda x: int(x.split(".")[-2]))

        fprint(f"opening {len(files)} blocks for snapshot `{self.nsnap}`.")
        for n, file in enumerate(tqdm(files, desc="Reading block")):
            with File(file, "r") as f:
                if "PartType1" in f:
                    if kind == "Masses":
                        npart = f["Header"].attrs["NumPart_ThisFile"][1]
                        x_high = np.ones(npart, dtype=np.float32)
                        x_high *= f["Header"].attrs["MassTable"][1]
                    else:
                        x_high = f[f"PartType1/{kind}"][...]
                else:
                    x_high = None

                if not high_resolution_only and "PartType5" in f:
                    if kind == "Masses" and "Masses" not in f["PartType5"]:
                        npart = f["Header"].attrs["NumPart_ThisFile"][5]
                        x_low = np.ones(npart, dtype=np.float32)
                        x_low *= f["Header"].attrs["MassTable"][5]
                    else:
                        x_low = f[f"PartType5/{kind}"][...]
                else:
                    x_low = None

                if x_high is not None and x_low is None:
                    x = x_high
                elif x_high is None and x_low is not None:
                    x = x_low
                else:
                    if x_high.ndim == 1:
                        x = np.hstack([x_high, x_low])
                    else:
                        x = np.vstack([x_high, x_low])

                if n == 0:
                    xs = np.copy(x)
                else:
                    if x.ndim == 1:
                        xs = np.hstack([xs, x])
                    else:
                        xs = np.vstack([xs, x])

                del x_high, x_low
                collect()

        if self.flip_xz and kind in ["Coordinates", "Velocities"]:
            xs[:, [0, 2]] = xs[:, [2, 0]]

        return xs

    def coordinates(self, high_resolution_only=False):
        return self._get_particles("Coordinates", high_resolution_only)

    def velocities(self, high_resolution_only=False):
        return self._get_particles("Velocities", high_resolution_only)

    def masses(self, high_resolution_only=False):
        return self._get_particles("Masses", high_resolution_only) * 1e10

    def particle_ids(self, high_resolution_only=False):
        return self._get_particles("ParticleIDs", high_resolution_only)

    def _get_halo_particles(self, halo_id, kind, is_group):
        if not is_group:
            raise RuntimeError("While the CSiBORG3 subhalo catalogue exists, "
                               "it is not currently implemented.")

        # Get global offsets for this halo
        i1, j1 = self.hid2offset["type1"].get(halo_id, (None, None))
        i5, j5 = self.hid2offset["type5"].get(halo_id, (None, None))

        # Check if this is a valid halo
        if i1 is None and i5 is None:
            raise ValueError(f"Halo `{halo_id}` not found.")
        if (i1 is None or j1 - i1 == 0) and (i5 is None or j5 - i5 == 0):
            raise ValueError(f"Halo `{halo_id}` has no particles.")

        # Get snapshot files
        # Handle both "snapshot_001.hdf5" and "snapshot_001.0.hdf5" formats
        parts = self._snapshot_path.rsplit('.', 2)
        if len(parts) == 3 and parts[-2].isdigit() and parts[-1] == 'hdf5':
            # Path has a block number like "snapshot_001.0.hdf5"
            pattern = f"{parts[0]}.*.hdf5"
        else:
            # Path is like "snapshot_001.hdf5", just add wildcard
            pattern = self._snapshot_path.replace(".hdf5", ".*.hdf5")

        files = glob(pattern)
        if len(files) == 0:
            raise FileNotFoundError(
                f"No files found for snapshot {self.nsnap} with pattern {pattern}.")
        files = sorted(files, key=lambda x: int(x.split(".")[-2]))

        x1_parts = []
        x5_parts = []

        # Track cumulative particle counts across blocks
        cumsum_type1 = 0
        cumsum_type5 = 0

        for file in files:
            with File(file, "r") as f:
                npart_type1 = f["Header"].attrs["NumPart_ThisFile"][1]
                npart_type5 = f["Header"].attrs["NumPart_ThisFile"][5]

                # Check if Type1 particles for this halo are in this block
                if i1 is not None and j1 - i1 > 0:
                    block_start = cumsum_type1
                    block_end = cumsum_type1 + npart_type1

                    # Check for overlap with halo particle range
                    if block_start < j1 and block_end > i1:
                        local_start = max(0, i1 - block_start)
                        local_end = min(npart_type1, j1 - block_start)

                        if kind == "Masses":
                            x_temp = np.ones(local_end - local_start,
                                             dtype=np.float32)
                            x_temp *= f["Header"].attrs["MassTable"][1]
                        else:
                            x_temp = f[f"PartType1/{kind}"][
                                local_start:local_end]

                            if self.flip_xz and kind in ["Coordinates",
                                                         "Velocities"]:
                                x_temp[:, [0, 2]] = x_temp[:, [2, 0]]

                        x1_parts.append(x_temp)

                # Check if Type5 particles for this halo are in this block
                if i5 is not None and j5 - i5 > 0:
                    block_start = cumsum_type5
                    block_end = cumsum_type5 + npart_type5

                    # Check for overlap with halo particle range
                    if block_start < j5 and block_end > i5:
                        local_start = max(0, i5 - block_start)
                        local_end = min(npart_type5, j5 - block_start)

                        if kind == "Masses":
                            x_temp = np.ones(local_end - local_start,
                                             dtype=np.float32)
                            x_temp *= f["Header"].attrs["MassTable"][5]
                        else:
                            x_temp = f[f"PartType5/{kind}"][
                                local_start:local_end]

                            if self.flip_xz and kind in ["Coordinates",
                                                         "Velocities"]:
                                x_temp[:, [0, 2]] = x_temp[:, [2, 0]]

                        x5_parts.append(x_temp)

                cumsum_type1 += npart_type1
                cumsum_type5 += npart_type5

        # Combine particles from all blocks
        if len(x1_parts) > 0:
            if x1_parts[0].ndim == 1:
                x1 = np.hstack(x1_parts)
            else:
                x1 = np.vstack(x1_parts)
        else:
            x1 = None

        if len(x5_parts) > 0:
            if x5_parts[0].ndim == 1:
                x5 = np.hstack(x5_parts)
            else:
                x5 = np.vstack(x5_parts)
        else:
            x5 = None

        # Stack high-resolution and low-resolution particles
        if x1 is None and x5 is None:
            raise ValueError(f"No particles found for halo {halo_id}")
        elif x5 is None:
            return x1
        elif x1 is None:
            return x5
        else:
            if x1.ndim > 1:
                return np.vstack([x1, x5])
            else:
                return np.hstack([x1, x5])

    def halo_coordinates(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Coordinates", is_group)

    def halo_velocities(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Velocities", is_group)

    def halo_masses(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Masses", is_group) * 1e10

    def halo_particle_ids(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "ParticleIDs", is_group)

    def particle_halo_ids(self, high_resolution_only=False):
        """
        Create an array mapping particle index to halo ID.

        For each particle, returns the halo index it belongs to. Particles
        not in any halo are assigned -1.

        Parameters
        ----------
        high_resolution_only : bool, optional
            Whether to include only high-resolution particles (Type 1).

        Returns
        -------
        halo_ids : 1-dimensional array of integers
            Array where halo_ids[i] gives the halo ID for particle i.
            Value is -1 if particle is not in any halo.
        """
        # Get the total number of particles from the first file
        files = glob(self._snapshot_path.replace(".hdf5", ".*.hdf5"))
        if len(files) == 0:
            # Handle single file or path with block number
            parts = self._snapshot_path.rsplit('.', 2)
            if len(parts) == 3 and parts[-2].isdigit() and parts[-1] == 'hdf5':
                pattern = f"{parts[0]}.*.hdf5"
            else:
                pattern = self._snapshot_path.replace(".hdf5", ".*.hdf5")
            files = glob(pattern)

        # Count total particles
        npart_type1 = 0
        npart_type5 = 0
        for file in files:
            with File(file, "r") as f:
                npart_type1 += f["Header"].attrs["NumPart_ThisFile"][1]
                if not high_resolution_only:
                    npart_type5 += f["Header"].attrs["NumPart_ThisFile"][5]

        # Create array initialized to -1 (no halo)
        total_npart = npart_type1 + npart_type5
        halo_ids = np.full(total_npart, -1, dtype=np.int32)

        # Fill in halo IDs using offsets
        # Type 1 particles are at indices [0:npart_type1]
        for halo_id, (start, end) in self.hid2offset["type1"].items():
            if end > start:
                halo_ids[start:end] = halo_id

        # Type 5 particles are at indices [npart_type1:npart_type1+npart_type5]
        if not high_resolution_only:
            for halo_id, (start, end) in self.hid2offset["type5"].items():
                if end > start:
                    halo_ids[npart_type1 + start:npart_type1 + end] = halo_id

        return halo_ids

    def _make_hid2offset(self):
        # Import here to avoid circular imports
        from .catalogue import CSiBORG3Catalogue

        catalogue = CSiBORG3Catalogue(
            nsim=self.nsim,
            nsnap=self.nsnap,
            paths=self.paths,
            fpath_override=self._catalogue_path,
            verbose=False
        )

        offset_type = catalogue._read_fof_catalogue("GroupOffsetType")
        len_type = catalogue._read_fof_catalogue("GroupLenType")

        offset = offset_type[:, 1]
        lengths = len_type[:, 1]
        hid2offset_type1 = {i: (offset[i], offset[i] + lengths[i])
                            for i in range(len(offset))}

        offset = offset_type[:, 5]
        lengths = len_type[:, 5]
        hid2offset_type5 = {i: (offset[i], offset[i] + lengths[i])
                            for i in range(len(offset))}

        self._hid2offset = {"type1": hid2offset_type1,
                            "type5": hid2offset_type5,
                            }


###############################################################################
#                           Quijote snapshot class                            #
###############################################################################


class QuijoteSnapshot(CSiBORG1Snapshot):
    """
    Quijote snapshot class with the FoF halo finder particle assignment.
    Because of similarities with how the snapshot is processed with CSiBORG1,
    it uses the same base class.

    Parameters
    ----------
    nsim : int
        Simulation index.
    nsnap : int
        Snapshot index.
    paths : Paths, optional
        Paths object.
    keep_snapshot_open : bool, optional
        Whether to keep the snapshot file open when reading halo particles.
        This is useful for repeated access to the snapshot.
    """
    def __init__(self, nsim, nsnap, paths=None, keep_snapshot_open=False):
        super().__init__(nsim, nsnap, paths, keep_snapshot_open, flip_xz=False)
        self._snapshot_path = self.paths.snapshot(self.nsnap, self.nsim,
                                                  "quijote")
        self._simname = "quijote"

    def _make_hid2offset(self):
        catalogue_path = self.paths.snapshot_catalogue(
            self.nsnap, self.nsim, "quijote")

        with File(catalogue_path, "r") as f:
            offset = f["GroupOffset"][:]

        self._hid2offset = {int(i): (int(j), int(k)) for i, j, k in offset}


###############################################################################
#                          Base field class                                   #
###############################################################################


class BaseField(ABC):
    """
    Base class for reading fields such as density or velocity fields.
    """
    def __init__(self, nsim, paths, flip_xz=False):
        if isinstance(nsim, np.integer):
            nsim = int(nsim)
        if not isinstance(nsim, int):
            raise TypeError(f"`nsim` must be an integer. Received `{type(nsim)}`.")  # noqa
        self._nsim = nsim

        if not isinstance(flip_xz, bool):
            raise TypeError("`flip_xz` must be a boolean.")
        self._flip_xz = flip_xz

        self._paths = paths

    @property
    def nsim(self):
        """Simulation index."""
        return self._nsim

    @property
    def paths(self):
        """Paths manager."""
        if self._paths is None:
            self._paths = Paths(**paths_glamdring)
        return self._paths

    @property
    def flip_xz(self):
        """
        Whether to flip the x- and z-axes to undo the MUSIC bug so that the
        coordinates are consistent with observations.
        """
        return self._flip_xz

    @abstractmethod
    def density_field(self, MAS, grid):
        """Precomputed density field."""
        pass

    @abstractmethod
    def velocity_field(self, MAS, grid):
        """Precomputed velocity field."""
        pass

    @abstractmethod
    def radial_velocity_field(self, MAS, grid):
        """Precomputed radial velocity field."""
        pass


###############################################################################
#                          CSiBORG1 field class                               #
###############################################################################


class CSiBORG1Field(BaseField):
    """
    CSiBORG1 `z = 0` field class.

    Parameters
    ----------
    nsim : int
        Simulation index.
    paths : Paths, optional
        Paths object. By default, the paths are set to the `glamdring` paths.
    flip_xz : bool, optional
        Whether to flip the x- and z-axes to undo the MUSIC bug so that the
        coordinates are consistent with observations.
    """
    def __init__(self, nsim, paths=None, flip_xz=True):
        super().__init__(nsim, paths, flip_xz)
        self._simname = "csiborg1"

    def density_field(self, MAS, grid):
        fpath = self.paths.field(
            "density", MAS, grid, self.nsim, self._simname)

        if MAS == "SPH":
            with File(fpath, "r") as f:
                field = f["density"][:]
                field /= (677.7 * 1e3 / grid)**3  # Convert to h^2 Msun / kpc^3
        else:
            field = np.load(fpath)

        if self.flip_xz:
            field = field.T

        return field

    def velocity_field(self, MAS, grid):
        fpath = self.paths.field(
            "velocity", MAS, grid, self.nsim, self._simname)

        if MAS == "SPH":
            with File(fpath, "r") as f:
                density = f["density"][:]
                v0 = f["p0"][:] / density
                v1 = f["p1"][:] / density
                v2 = f["p2"][:] / density
            field = np.array([v0, v1, v2])
        else:
            field = np.load(fpath)

        if self.flip_xz:
            field[0, ...] = field[0, ...].T
            field[1, ...] = field[1, ...].T
            field[2, ...] = field[2, ...].T
            field[[0, 2], ...] = field[[2, 0], ...]

        return field

    def radial_velocity_field(self, MAS, grid):
        if not self.flip_xz and self._simname == "csiborg1":
            raise ValueError("The radial velocity field is only implemented "
                             "for the flipped x- and z-axes.")

        fpath = self.paths.field("radvel", MAS, grid, self.nsim, self._simname)
        return np.load(fpath)


###############################################################################
#                          CSiBORG2 field class                               #
###############################################################################


class CSiBORG2Field(BaseField):
    """
    CSiBORG2 `z = 0` field class.

    Parameters
    ----------
    nsim : int
        Simulation index.
    kind : str
        CSiBORG2 run kind. One of `main`, `random`, or `varysmall`.
    paths : Paths, optional
        Paths object. By default, the paths are set to the `glamdring` paths.
    flip_xz : bool, optional
        Whether to flip the x- and z-axes to undo the MUSIC bug so that the
        coordinates are consistent with observations.
    """
    def __init__(self, nsim, kind, paths=None, flip_xz=True):
        super().__init__(nsim, paths, flip_xz)
        self.kind = kind

    @property
    def kind(self):
        """CSiBORG2 run kind."""
        return self._kind

    @kind.setter
    def kind(self, value):
        if value not in ["main", "random", "varysmall"]:
            raise ValueError("`kind` must be one of `main`, `random`, or `varysmall`.")  # noqa
        self._kind = value

    def density_field(self, MAS, grid):
        fpath = self.paths.field("density", MAS, grid, self.nsim,
                                 f"csiborg2_{self.kind}")

        if MAS == "SPH":
            with File(fpath, "r") as f:
                field = f["density"][:]
            field *= 1e10                     # Convert to Msun / h
            field /= (676.6 * 1e3 / grid)**3  # Convert to h^2 Msun / kpc^3
        else:
            field = np.load(fpath)

        if self.flip_xz:
            field = field.T

        return field

    def velocity_field(self, MAS, grid):
        fpath = self.paths.field("velocity", MAS, grid, self.nsim,
                                 f"csiborg2_{self.kind}")

        if MAS == "SPH":
            with File(fpath, "r") as f:
                density = f["density"][:]
                v0 = f["p0"][:] / density
                v1 = f["p1"][:] / density
                v2 = f["p2"][:] / density
            field = np.array([v0, v1, v2])
        else:
            field = np.load(fpath)

        if self.flip_xz:
            field[0, ...] = field[0, ...].T
            field[1, ...] = field[1, ...].T
            field[2, ...] = field[2, ...].T
            field[[0, 2], ...] = field[[2, 0], ...]

        return field

    def radial_velocity_field(self, MAS, grid):
        if not self.flip_xz:
            raise ValueError("The radial velocity field is only implemented "
                             "for the flipped x- and z-axes.")

        fpath = self.paths.field("radvel", MAS, grid, self.nsim,
                                 f"csiborg2_{self.kind}")
        return np.load(fpath)


class CSiBORG2XField(BaseField):
    """
    CSiBORG2X `z = 0` field class based on the Manticore ICs.

    Parameters
    ----------
    nsim : int
        Simulation index.
    version : str
        Manticore version index.
    paths : Paths, optional
        Paths object. By default, the paths are set to the `glamdring` paths.
    """
    def __init__(self, nsim, version, paths=None):
        self.version = version
        if version == 0:
            self.nametag = "csiborg2X"
        elif version == 1:
            self.nametag = "manticore_2MPP_N128_DES_V1"
        elif version == 2:
            self.nametag = "manticore_2MPP_MULTIBIN_N128_DES_V1"
        elif version == 3:
            self.nametag = "manticore_2MPP_MULTIBIN_N128_DES_V2"
        elif version == 4:
            self.nametag = "manticore_2MPP_MULTIBIN_N256_DES_V2"
        else:
            raise ValueError("Invalid Manticore version.")

        super().__init__(nsim, paths, False)

    def overdensity_field(self, **kwargs):
        if self.version == 0:
            fpath = self.paths.field(
                "overdensity", None, None, self.nsim, self.nametag)
            with File(fpath, "r") as f:
                field = f["delta_cic"][...].astype(np.float32)
        else:
            raise ValueError("Invalid Manticore version to read the "
                             "overdensity field.")

        return field

    def density_field(self, **kwargs):
        if self.version == 0:
            field = self.overdensity_field()
            omega0 = simname2Omega_m(self.nametag)
            rho_mean = omega0 * 277.53662724583074  # Msun / kpc^3
            field += 1
            field *= rho_mean
        elif self.version in [1, 2, 3, 4]:
            MAS = kwargs["MAS"]
            grid = kwargs["grid"]
            fpath = self.paths.field(
                "density", MAS, grid, self.nsim, self.nametag)

            if MAS == "SPH":
                with File(fpath, "r") as f:
                    field = f["density"][:]

                field /= (681.1 * 1e3 / grid)**3  # Convert to h^2 Msun / kpc^3
            else:
                field = np.load(fpath)
        else:
            raise ValueError("Invalid Manticore version to read the "
                             "density field.")

        return field

    def velocity_field(self, **kwargs):
        if self.version == 0:
            fpath = self.paths.field(
                "velocity", None, None, self.nsim, "csiborg2X")
            with File(fpath, "r") as f:
                v0 = f["v_0"][...]
                v1 = f["v_1"][...]
                v2 = f["v_2"][...]
                field = np.array([v0, v1, v2])
        elif self.version in [1, 2, 3, 4]:
            MAS = kwargs["MAS"]
            grid = kwargs["grid"]
            fpath = self.paths.field(
                "velocity", MAS, grid, self.nsim, self.nametag)

            if MAS:
                with File(fpath, "r") as f:
                    density = f["density"][:]
                    v0 = f["p0"][:] / density
                    v1 = f["p1"][:] / density
                    v2 = f["p2"][:] / density
                field = np.array([v0, v1, v2])
            else:
                field = np.load(fpath)

        return field

    def radial_velocity_field(self, **kwargs):
        raise RuntimeError("The radial velocity field is not available.")


###############################################################################
#                           BORG1 field class                                 #
###############################################################################


class BORG1Field(BaseField):
    """
    BORG2 `z = 0` field class.

    Parameters
    ----------
    nsim : int
        Simulation index.
    paths : Paths, optional
        Paths object. By default, the paths are set to the `glamdring` paths.
    """
    def __init__(self, nsim, paths=None):
        super().__init__(nsim, paths, False)

    def overdensity_field(self):
        fpath = self.paths.field(None, None, None, self.nsim, "borg1")
        with File(fpath, "r") as f:
            field = f["scalars/BORG_final_density"][:].astype(np.float32)

        return field

    def density_field(self):
        field = self.overdensity_field()
        omega0 = simname2Omega_m("borg1")
        rho_mean = omega0 * 277.53662724583074  # Msun / kpc^3
        field += 1
        field *= rho_mean
        return field

    def velocity_field(self, MAS, grid):
        raise RuntimeError("The velocity field is not available.")

    def radial_velocity_field(self, MAS, grid):
        raise RuntimeError("The radial velocity field is not available.")


###############################################################################
#                           BORG2 field class                                 #
###############################################################################


class BORG2Field(BaseField):
    """
    BORG2 `z = 0` field class.

    Parameters
    ----------
    nsim : int
        Simulation index.
    paths : Paths, optional
        Paths object. By default, the paths are set to the `glamdring` paths.
    """
    def __init__(self, nsim, paths=None):
        super().__init__(nsim, paths, False)

    def overdensity_field(self):
        fpath = self.paths.field(None, None, None, self.nsim, "borg2")
        with File(fpath, "r") as f:
            field = f["scalars/BORG_final_density"][:].astype(np.float32)

        return field

    def density_field(self):
        field = self.overdensity_field()
        omega0 = simname2Omega_m("borg2")
        rho_mean = omega0 * 277.53662724583074  # h^2 Msun / kpc^3
        field += 1
        field *= rho_mean
        return field

    def velocity_field(self, MAS, grid):
        raise RuntimeError("The velocity field is not available.")

    def radial_velocity_field(self, MAS, grid):
        raise RuntimeError("The radial velocity field is not available.")


###############################################################################
#                             Additional fields                               #
###############################################################################

class TNG300_1Field(BaseField):
    """
    TNG300-1 dark matter-only `z = 0` field class.

    Parameters
    ----------
    paths : Paths, optional
        Paths object. By default, the paths are set to the `glamdring` paths.
    """
    def __init__(self, paths=None):
        super().__init__(0, paths, False)

    def overdensity_field(self, MAS, grid):
        density = self.density_field(MAS, grid)
        omega_dm = 0.3089 - 0.0486
        rho_mean = omega_dm * 277.53662724583074  # h^2 Msun / kpc^3

        density /= rho_mean
        density -= 1

        return density

    def density_field(self, MAS, grid):
        fpath = join(self.paths.tng300_1(), "postprocessing", "density_field",
                     f"rho_dm_099_{grid}_{MAS}.npy")
        return np.load(fpath)

    def velocity_field(self, MAS, grid):
        raise RuntimeError("The velocity field is not available.")

    def radial_velocity_field(self, MAS, grid):
        raise RuntimeError("The radial velocity field is not available.")


class QuijoteField(CSiBORG1Field):
    """
    Quijote `z = 0` field class.

    Parameters
    ----------
    nsim : int
        Simulation index.
    paths : Paths
        Paths object.
    """
    def __init__(self, nsim, paths):
        super().__init__(nsim, paths, flip_xz=False)
        self._simname = "quijote"
