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
Support for reading the CSiBORG merger trees.
"""

import numpy
from abc import ABC
from .paths import Paths
from tqdm import tqdm
from os.path import isfile, getsize
from treelib import Node, Tree
from h5py import File
from gc import collect


def clump_identifier(clump, nsnap):
    """
    Generate a unique identifier for a clump at a given snapshot.

    Parameters
    ----------
    clump : int
        Clump ID.
    nsnap : int
        Snapshot index.

    Returns
    -------
    str
    """
    return f"{str(clump).rjust(9, 'x')}__{str(nsnap).rjust(4, 'x')}"


def extract_identifier(identifier):
    clump, nsnap = identifier.split('__')
    return int(clump.lstrip('x')), int(nsnap.lstrip('x'))


class BaseMergerReader(ABC):
    _paths = None
    _nsim = None

    @property
    def paths(self):
        """Paths manager."""
        if self._paths is None:
            raise ValueError("`paths` is not set.")
        return self._paths

    @paths.setter
    def paths(self, paths):
        assert isinstance(paths, Paths)
        self._paths = paths

    @property
    def nsim(self):
        """Simulation index."""
        if self._nsim is None:
            raise ValueError("`nsim` is not set.")
        return self._nsim

    @nsim.setter
    def nsim(self, nsim):
        assert isinstance(nsim, (int, numpy.integer))
        self._nsim = nsim


class MergerReader(BaseMergerReader):
    # Again make some cache

    _cache = {}

    def __init__(self, nsim, paths):
        self.nsim = nsim
        self.paths = paths

    def cache_length(self):
        """Length of the cache."""
        return len(self._cache)

    def cache_clear(self):
        """Clear the cache."""
        self._cache = {}
        collect()

    def get_info(self, current_clump, current_snap):
        """
        Make a list of information about a clump at a given snapshot. Elements
        are mass and position.

        Parameters
        ----------
        current_clump : int
            Clump ID.
        current_snap : int
            Snapshot index.

        Returns
        -------
        list
        """
        if current_clump < 0:
            raise ValueError("Clump ID must be positive.")
        k = self[f"{current_snap}__clump_to_array"][current_clump][0]

        return [self[f"{current_snap}__desc_mass"][k],
                *self[f"{current_snap}__desc_pos"][k]
                ]

    def find_main_progenitor(self, clump, nsnap):
        """
        Find the main progenitor of a clump at a given snapshot. Cases are:
            - `clump > 0`, `progenitor > 0`: main progenitor is in the adjacent
            snapshot,
            - `clump > 0`, `progenitor < 0`: main progenitor is not in the
            adjacent snapshot.
            - `clump < 0`, `progenitor = 0`: no progenitor, newly formed clump.

        Parameters
        ----------
        clump : int
            Clump ID.
        nsnap : int
            Snapshot index.

        Returns
        -------
        progenitor : int
            Main progenitor clump ID.
        progenitor_snap : int
            Main progenitor snapshot index.
        """
        if not clump > 0:
            raise ValueError("Clump ID must be positive.")

        cl2array = self[f"{nsnap}__clump_to_array"]
        if clump in cl2array:
            k = cl2array[clump]
        else:
            raise ValueError("Clump ID not found.")

        if len(k) > 1:
            raise ValueError("Found more than one main progenitor.")

        k = k[0]
        progenitor = self[f"{nsnap}__progenitor"][k]
        progenitor_snap = self[f"{nsnap}__progenitor_outputnr"][k]

        print("Current snapshot: ", nsnap)
        print("Found main progenitor: ", progenitor, progenitor_snap)

        # TODO add a real termination
        if nsnap < 945:
            return 0, -1

        return progenitor, progenitor_snap

    def find_minor_progenitors(self, clump, nsnap):
        """
        Find the minor progenitors of a clump at a given snapshot. This means
        that `clump < 0`, `progenitor > 0`, i.e. this clump also has another
        main progenitor.

        If there are no minor progenitors, return `None` for both lists.

        Parameters
        ----------
        clump : int
            Clump ID.
        nsnap : int
            Snapshot index.

        Returns
        -------
        progenitor : list
            List of minor progenitor clump IDs.
        progenitor_snap : list
            List of minor progenitor snapshot indices.
        """
        if not clump > 0:
            raise ValueError("Clump ID must be positive.")

        try:
            ks = self[f"{nsnap}__clump_to_array"][-clump]
        except KeyError:
            return None, None

        progenitor = [self[f"{nsnap}__progenitor"][k]
                      for k in ks]
        progenitor_snap = [self[f"{nsnap}__progenitor_outputnr"][k]
                           for k in ks]

        print(f"Found minor progenitors: {len(ks)} for {clump} ({nsnap})")
        print(progenitor, progenitor_snap)

        return progenitor, progenitor_snap

    def find_progenitors(self, clump, nsnap):
        """
        Find all progenitors of a clump at a given snapshot. The main
        progenitor is the first element of the list.

        Parameters
        ----------
        clump : int
            Clump ID.
        nsnap : int
            Snapshot index.

        Returns
        -------
        prog : list
            List of progenitor clump IDs.
        prog_nsnap : list
            List of progenitor snapshot indices.
        """
        main_prog, main_prog_nsnap = self.find_main_progenitor(clump, nsnap)
        min_prog, min_prog_nsnap = self.find_minor_progenitors(clump, nsnap)

        if min_prog is None:
            prog = [main_prog,]
            prog_nsnap = [main_prog_nsnap,]
        else:
            prog = [main_prog,] + min_prog
            prog_nsnap = [main_prog_nsnap,] + min_prog_nsnap

        return prog, prog_nsnap

    def make_tree(self, current_clump, current_nsnap,
                  above_clump=None, above_nsnap=None,
                  tree=None):
        # There seems to still be issues..
        # Terminate when the progenitor is 0
        if current_clump == 0:
            print("Maybe not finish it like this? Actually maybe do.")
            return tree

        # Create the root node or add a new node
        if tree is None:
            tree = Tree()
            tree.create_node(
                "root",
                identifier=clump_identifier(current_clump, current_nsnap),
                data=self.get_info(current_clump, current_nsnap),
                )
        else:
            print("Adding node: ", current_clump, current_nsnap)
            print("From parent: ", above_clump, above_nsnap)
            print()

            tree.create_node(
                identifier=clump_identifier(current_clump, current_nsnap),
                parent=clump_identifier(above_clump, above_nsnap),
                data=self.get_info(current_clump, current_nsnap),
                )

        prog, prog_nsnap = self.find_progenitors(current_clump, current_nsnap)

        for p, psnap in zip(prog, prog_nsnap):
            print("Making tree for: ", p, psnap)
            self.make_tree(p, psnap, current_clump, current_nsnap, tree)

        # print("Returning.. ", current_clump, current_nsnap)

        # return tree

    def walk_main_progenitor(self, clump, nsnap):
        """
        Walk the main progenitor branch of a clump.

        Parameters
        ----------
        clump : int
            Clump ID.
        nsnap : int
            Snapshot index.

        Returns
        -------
        out : 2-dimensional array of shape `(nsteps, 5)`
            Array with columns `(nsnap, clump, nsnap, mass, pos)`.
        """
        out = [[nsnap,] + self.get_info(clump, nsnap),]

        while True:
            clump, nsnap = self.find_main_progenitor(clump, nsnap)
            if clump == 0:
                break

            out += [[nsnap,] + self.get_info(clump, nsnap),]

        return numpy.vstack(out)

    def __getitem__(self, key):
        """
        Must already be processed as a HDF5 file.
        """
        try:
            return self._cache[key]
        except KeyError:
            fname = self.paths.processed_merger_tree(self.nsim)

            nsnap, kind = key.split("__")

            with File(fname, "r") as f:
                if kind == "clump_to_array":
                    cl = self[f"{nsnap}__clump"]

                    x = {}
                    for i, c in enumerate(cl):
                        if c in x:
                            x[c] += (i,)
                        else:
                            x[c] = (i,)
                else:
                    x = f[f"{str(nsnap)}/{kind}"][:]

            # Cache it
            self._cache[key] = x

            return x
