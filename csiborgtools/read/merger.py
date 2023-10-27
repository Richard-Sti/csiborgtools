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

class ClumpInfo:
    _clump = None
    _nsnap = None

    def __init__(self, clump, nsnap, mass=None, pos=None, vel=None):
        self.clump = clump
        self.nsnap = nsnap

    @property
    def clump(self):
        """Clump ID."""
        if self._clump is None:
            raise ValueError("`clump` is not set.")
        return self._clump

    @clump.setter
    def clump(self, clump):
        if not isinstance(clump, int) and not clump > 0:
            raise ValueError(f"Invalid clump ID `{clump}`.")
        self._clump = clump

    @property
    def nsnap(self):
        """Snapshot number."""
        if self._nsnap is None:
            raise ValueError("`nsnap` is not set.")
        return self._nsnap

    @nsnap.setter
    def nsnap(self, nsnap):
        if not isinstance(nsnap, int) and not nsnap > 0:
            raise ValueError(f"Invalid snapshot number `{nsnap}`.")
        self._nsnap = nsnap

    def __repr__(self):
        return f"{str(self.clump)}__{str(self.nsnap).zfill(4)}"


def clump_identifier(clump, nsnap):
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

    def find_main_progenitor(self, clump, nsnap):
        if not clump > 0:
            raise ValueError("Clump ID must be positive.")

        print("Current snapshot: ", nsnap)

        cl2array = self[f"{nsnap}__clump_to_array"]
        if clump in cl2array:
            k = cl2array[clump]
        else:
            raise ValueError("Clump ID not found.")

        if len(k) > 1:
            raise ValueError("Detected more than one main progenitor.")
        k = k[0]

        progenitor = self[f"{nsnap}__progenitor"][k]
        progenitor_snap = self[f"{nsnap}__progenitor_outputnr"][k]

        if nsnap < 940:
            return 0, None

        return progenitor, progenitor_snap

    def find_minor_progenitors(self, clump, nsnap):
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
        return progenitor, progenitor_snap

    def walk_main_progenitor(self, clump, nsnap):
        print("Current snapshot: ", nsnap)

        progenitor, progenitor_snap = self.find_main_progenitor(clump, nsnap)


        if progenitor == 0:
            return None, None

        # k = self[f"{nsnap}__clump_to_array"][clump][0]

        # print(clump, nsnap, m)

        return self.walk_main_progenitor(progenitor, progenitor_snap)

    def make_tree(self, current_clump, current_nsnap, above_clump=None, above_nsnap=None, tree=None):
        if tree is None:
            tree = Tree()
            tree.create_node(
                "root",
                identifier=clump_identifier(current_clump, current_nsnap)
                )
        else:
            tree.create_node(
                identifier=clump_identifier(current_clump, current_nsnap),
                parent=clump_identifier(above_clump, above_nsnap)
                )

        if current_clump == 0:
            print("A")
            return tree

        main_prog, main_prog_nsnap = self.find_main_progenitor(
            current_clump, current_nsnap)

        if main_prog == 0:
            print("B")
            return tree

        min_prog, min_prog_nsnap = self.find_minor_progenitors(
            current_clump, current_nsnap)

        if min_prog is None:
            prog = (main_prog,)
            prog_nsnap = (main_prog_nsnap,)
        else:
            prog = (main_prog,) + min_prog
            prog_nsnap = (main_prog_nsnap,) + min_prog_nsnap

        for p, psnap in zip(prog, prog_nsnap):
            self.make_tree(p, psnap, current_clump, current_nsnap, tree)

        print("Returning.. ", current_clump, current_nsnap)

        return tree

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

