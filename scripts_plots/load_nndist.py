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
Script to cache to disk the nearest neighbour distance distributions.
"""
from mpi4py import MPI
from cache_to_disk import delete_disk_caches_for_function
from taskmaster import work_delegation

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools

from plot_match import read_dist


def main(run):
    neighbour_kwargs = {"rmax_radial": 155 / 0.705,
                        "nbins_radial": 50,
                        "rmax_neighbour": 100.,
                        "nbins_neighbour": 150,
                        "paths_kind": csiborgtools.paths_glamdring}
    for simname in ["csiborg", "quijote"]:
        for kind in ["pdf", "cdf"]:
            read_dist(simname, run, kind, neighbour_kwargs)


if __name__ == "__main__":
    cached_funcs = ["read_dist"]
    for func in cached_funcs:
        print(f"Cleaning cache for function {func}.")
        delete_disk_caches_for_function(func)

    runs = ["mass001", "mass002", "mass003", "mass004", "mass005", "mass006",
            "mass007", "mass008", "mass009"]

    work_delegation(main, runs, MPI.COMM_WORLD)
