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
from abc import ABC, abstractmethod

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

    @property
    def nsim(self):
        return self._nsim

    @property
    def nsnap(self):
        return self._nsnap

    @property
    def paths(self):
        return self._paths

    @abstractmethod
    def coordinates(self, halo_id=None):
        """
        Return the particle coordinates. If halo_id is `None`, return all
        particles in the snapshot, otherwise return only particles in the halo
        specified by `halo_id`.

        Parameters
        ----------
        halo_id : int
            Halo ID.

        Returns
        -------
        coords : 2-dimensional array
        """
        pass

    @abstractmethod
    def velocities(self, halo_id=None):
        """
        Return the particle velocities. If halo_id is `None`, return all
        particles in the snapshot, otherwise return only particles in the halo
        specified by `halo_id`.

        Parameters
        ----------
        halo_id : int
            Halo ID.

        Returns
        -------
        vel : 2-dimensional array
        """
        pass

    @abstractmethod
    def masses(self, halo_id=None):
        """
        Return the particle masses. If halo_id is `None`, return all particles
        in the snapshot. Otherwise return only particles in the halo specified
        by `halo_id`.

        Parameters
        ----------
        halo_id : int
            Halo ID.

        Returns
        -------
        mass : 1-dimensional array
        """
        pass

    @abstractmethod
    def particle_ids(self, halo_id=None):
        """
        Return the particle IDs. If halo_id is `None`, return all particles in
        in the snapshot. Otherwise return only particles in the halo specified
        by `halo_id`.

        Parameters
        ----------
        halo_id : int
            Halo ID.

        Returns
        -------
        ids : 1-dimensional array
        """
        pass


###############################################################################
#                          CSiBORG1 snapshot class                            #
###############################################################################


class CSIBORG1Snapshot(BaseSnapshot):
    """
    Read CSiBORG1 snapshots FOF SNAPSHOT
    """

    def __init__(self, nsim, nsnap, paths):
        super().__init__(nsim, nsnap, paths)
        self._snapshot_path = self.paths.snapshot(self.nsnap, self.nsim,
                                                  "csiborg1")

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


###############################################################################
#                          CSiBORG2 snapshot class                            #
###############################################################################


