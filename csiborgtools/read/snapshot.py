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


TODO: What to do here since some snapshots all have substructure?


"""
from abc import ABC, abstractmethod, abstractproperty

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

    def coordinates(self, halo_id=None):
        with File(self._snapshot_path, "r") as f:
            if halo_id is None:
                coords = f["Coordinates"][...]
            else:
                raise RuntimeError("Not implemented yet")

        return coords

    def velocities(self, halo_id=None):
        with File(self._snapshot_path, "r") as f:
            if halo_id is None:
                vel = f["Velocities"][...]
            else:
                raise RuntimeError("Not implemented yet")

        return vel

    def masses(self, halo_id=None):
        with File(self._snapshot_path, "r") as f:
            if halo_id is None:
                mass = f["Masses"][...]
            else:
                raise RuntimeError("Not implemented yet")

        return mass

    def particle_ids(self, halo_id=None):
        with File(self._snapshot_path, "r") as f:
            if halo_id is None:
                ids = f["ParticleIDs"][...]
            else:
                raise RuntimeError("Not implemented yet")

        return ids

    def _get_halo_particles(self, halo_id, kind):
        with File(self._snapshot_path, "r") as f:
            i, j = self.hid2offset.get(halo_id, (None, None))

            if i is None:
                raise ValueError(f"Halo `{halo_id}` not found.")

            x = f[kind][i:j + 1]

        return x

    def halo_coordinates(self, halo_id, is_group=True):
        if not is_group:
            raise ValueError("There is no subhalo catalogue for CSiBORG1.")

        return self._get_halo_particles(halo_id, "Coordinates")

    def halo_velocities(self, halo_id, is_group=True):
        if not is_group:
            raise ValueError("There is no subhalo catalogue for CSiBORG1.")

        return self._get_halo_particles(halo_id, "Velocities")

    def halo_masses(self, halo_id, is_group=True):
        if not is_group:
            raise ValueError("There is no subhalo catalogue for CSiBORG1.")

        return self._get_halo_particles(halo_id, "Masses")

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
    CSiBORG2 snapshot class with the FoF halo finder particle assignment.




    Read CSiBORG1 snapshots FOF SNAPSHOT
    """
    def __init__(self, nsim, nsnap, paths, kind):
        super().__init__(nsim, nsnap, paths)
        self._snapshot_path = self.paths.snapshot(self.nsnap, self.nsim,
                                                  "quijote")

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


#         # First do the high-resolution particles
#         pos = snapshot["PartType1"]["Coordinates"]
#         mass = numpy.ones(len(pos), dtype=pos.dtype)
#         mass *= snapshot.header.attrs["MassTable"][1]
#         field = gen(pos, mass, parser_args.grid, verbose=parser_args.verbose)
#
#         # And then the low-resolution particles
#         pos = snapshot["PartType5"]["Coordinates"]
#         mass = snapshot["PartType5"]["Masses"]
#         field += gen(pos, mass, parser_args.grid,
# verbose=parser_args.verbose)
#
#         # Convert to `Msun / h`
#         field *= 1e10


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

        self._hid2offset = {i: (j, k) for i, j, k in offset}
