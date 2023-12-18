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
Classes for reading in snapshots and unifying the snapshot interface.
"""
from abc import ABC, abstractmethod, abstractproperty
import numpy

from h5py import File

###############################################################################
#                          Base snapshot class                                #
###############################################################################


class BaseSnapshot(ABC):
    """
    Base class for reading snapshots.
    """
    def __init__(self, nsim, nsnap, paths):
        if not isinstance(nsim, int):
            raise TypeError("`nsim` must be an integer")
        self._nsim = nsim

        if not isinstance(nsnap, int):
            raise TypeError("`nsnap` must be an integer")
        self._nsnap = nsnap

        self._paths = paths
        self._hid2offset = None

    @property
    def nsim(self):
        """
        Simulation index.

        Returns
        -------
        int
        """
        return self._nsim

    @property
    def nsnap(self):
        """
        Snapshot index.

        Returns
        -------
        int
        """
        return self._nsnap

    @property
    def paths(self):
        """
        Paths manager.

        Returns
        -------
        Paths
        """
        return self._paths

    @abstractproperty
    def coordinates(self):
        """
        Return the particle coordinates.

        Returns
        -------
        coords : 2-dimensional array
        """
        pass

    @abstractproperty
    def velocities(self):
        """
        Return the particle velocities.

        Returns
        -------
        vel : 2-dimensional array
        """
        pass

    @abstractproperty
    def masses(self):
        """
        Return the particle masses.

        Returns
        -------
        mass : 1-dimensional array
        """
        pass

    @abstractproperty
    def particle_ids(self):
        """
        Return the particle IDs.

        Returns
        -------
        ids : 1-dimensional array
        """
        pass

    @abstractmethod
    def halo_coordinates(self, halo_id, is_group):
        """
        Return the halo particle coordinates.

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
        """
        Private class function to make the halo ID to offset dictionary.
        """
        pass


###############################################################################
#                          CSiBORG1 snapshot class                            #
###############################################################################


class CSIBORG1Snapshot(BaseSnapshot):
    """
    CSiBORG1 snapshot class with the FoF halo finder particle assignment.
    CSiBORG1 was run with RAMSES.

    Parameters
    ----------
    nsim : int
        Simulation index.
    nsnap : int
        Snapshot index.
    paths : Paths
        Paths object.
    """
    def __init__(self, nsim, nsnap, paths):
        super().__init__(nsim, nsnap, paths)
        self._snapshot_path = self.paths.snapshot(
            self.nsnap, self.nsim, "csiborg1")

    def _get_particles(self, kind):
        with File(self._snapshot_path, "r") as f:
            x = f[kind][...]

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

        with File(self._snapshot_path, "r") as f:
            i, j = self.hid2offset.get(halo_id, (None, None))

            if i is None:
                raise ValueError(f"Halo `{halo_id}` not found.")

            x = f[kind][i:j + 1]

        return x

    def halo_coordinates(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Coordinates", is_group)

    def halo_velocities(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Velocities", is_group)

    def halo_masses(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Masses", is_group)

    def _make_hid2offset(self):
        catalogue_path = self.paths.snapshot_catalogue(
            self.nsnap, self.nsim, "csiborg1")

        with File(catalogue_path, "r") as f:
            offset = f["GroupOffset"][:]

        self._hid2offset = {i: (j, k) for i, j, k in offset}


###############################################################################
#                          CSiBORG2 snapshot class                            #
###############################################################################

class CSIBORG2Snapshot(BaseSnapshot):
    """
    CSiBORG2 snapshot class with the FoF halo finder particle assignment and
    SUBFIND subhalo finder. The simulations were run with Gadget4.

    Parameters
    ----------
    nsim : int
        Simulation index.
    nsnap : int
        Snapshot index.
    paths : Paths
        Paths object.
    kind : str
        CSiBORG2 run kind. One of `main`, `random`, or `varysmall`.
    """
    def __init__(self, nsim, nsnap, paths, kind):
        super().__init__(nsim, nsnap, paths)
        self.kind = kind

        self._snapshot_path = self.paths.snapshot(
            self.nsnap, self.nsim, f"csiborg2_{self.kind}")

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

    def _get_particles(self, kind):
        with File(self._snapshot_path, "r") as f:
            if kind == "Masses":
                npart = f["Header"].attrs["NumPart_Total"][1]
                x = numpy.ones(npart, dtype=numpy.float32)
                x *= f["Header"].attrs["MassTable"][1]
            else:
                x = f[f"PartType1/{kind}"][...]

            if x.ndim == 1:
                x = numpy.hstack([x, f[f"PartType5/{kind}"][...]])
            else:
                x = numpy.vstack([x, f[f"PartType5/{kind}"][...]])

        return x

    def coordinates(self):
        return self._get_particles("Coordinates")

    def velocities(self):
        return self._get_particles("Velocities")

    def masses(self):
        return self._get_particles("Masses") * 1e10

    def particle_ids(self):
        return self._get_particles("ParticleIDs")

    def _get_halo_particles(self, halo_id, kind, is_group):
        if not is_group:
            raise RuntimeError("While the CSiBORG2 subhalo catalogue exists, it is not currently implemented.")  # noqa

        with File(self._snapshot_path, "r") as f:
            i1, j1 = self.hid2offset["type1"].get(halo_id, (None, None))
            i5, j5 = self.hid2offset["type5"].get(halo_id, (None, None))

            # Check if this is a valid halo
            if i1 is None and i5 is None:
                raise ValueError(f"Halo `{halo_id}` not found.")
            if j1 - i1 == 0 and j5 - i5 == 0:
                raise ValueError(f"Halo `{halo_id}` has no particles.")

            if i1 is not None and j1 - i1 > 0:
                if kind == "Masses":
                    x1 = numpy.ones(j1 - i1, dtype=numpy.float32)
                    x1 *= f["Header"].attrs["MassTable"][1]
                else:
                    x1 = f[f"PartType1/{kind}"][i1:j1]

            if i5 is not None and j5 - i5 > 0:
                x5 = f[f"PartType5/{kind}"][i5:j5]

        if i5 is None or j5 - i5 == 0:
            return x1

        if i1 is None or j1 - i1 == 0:
            return x5

        if x1.ndim > 1:
            x1 = numpy.vstack([x1, x5])
        else:
            x1 = numpy.hstack([x1, x5])

        return x1

    def halo_coordinates(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Coordinates", is_group)

    def halo_velocities(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Velocities", is_group)

    def halo_masses(self, halo_id, is_group=True):
        return self._get_halo_particles(halo_id, "Masses", is_group) * 1e10

    def _make_hid2offset(self):
        catalogue_path = self.paths.snapshot_catalogue(
            self.nsnap, self.nsim, f"csiborg2_{self.kind}")
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
#                          CSiBORG2 snapshot class                            #
###############################################################################


class QuijoteSnapshot(CSIBORG1Snapshot):
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
    paths : Paths
        Paths object.
    """
    def __init__(self, nsim, nsnap, paths):
        super().__init__(nsim, nsnap, paths)
        self._snapshot_path = self.paths.snapshot(self.nsnap, self.nsim,
                                                  "quijote")

    def _make_hid2offset(self):
        catalogue_path = self.paths.snapshot_catalogue(
            self.nsnap, self.nsim, "quijote")

        with File(catalogue_path, "r") as f:
            offset = f["GroupOffset"][:]

        self._hid2offset = {int(i): (int(j), int(k)) for i, j, k in offset}
