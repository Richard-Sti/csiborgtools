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

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy
import scienceplots  # noqa
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function

import utils

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


@cache_to_disk(7)
def read_cdf(simname, run, kwargs):
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)
    return reader.build_cdf(simname, run, verbose=True)


def plot_cdf(kwargs):
    print("Plotting the CDFs.", flush=True)
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)
    x = reader.bin_centres("neighbour")

    y_quijote = read_cdf("quijote", "mass003", kwargs)
    y_csiborg = read_cdf("csiborg", "mass003", kwargs)
    ncdf = y_quijote.shape[0]

    with plt.style.context(utils.mplstyle):
        plt.figure()
        for i in range(ncdf):
            if i == 0:
                label1 = "Quijote"
                label2 = "CSiBORG"
            else:
                label1 = None
                label2 = None
            plt.plot(x, y_quijote[i], c="C0", label=label1)
            plt.plot(x, y_csiborg[i], c="C1", label=label2)
        plt.xlim(0, 75)
        plt.ylim(0, 1)
        plt.xlabel(r"$r_{\rm neighbour}~[\mathrm{Mpc}]$")
        plt.ylabel(r"$\mathrm{CDF}(r_{\rm neighbour})$")
        plt.legend()

        plt.tight_layout()
        plt.savefig("../plots/nearest_neighbour_cdf.png", dpi=450,
                    bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()

    kwargs = {"rmax_radial": 155 / 0.705,
              "nbins_radial": 20,
              "rmax_neighbour": 100.,
              "nbins_neighbour": 150,
              "paths_kind": csiborgtools.paths_glamdring}

    cached_funcs = ["read_cdf"]
    if args.clean:
        for func in cached_funcs:
            print(f"Cleaning cache for function {func}.")
            delete_disk_caches_for_function(func)


    plot_cdf(kwargs)
