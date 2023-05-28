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

from os.path import join
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy

import scienceplots  # noqa
import utils
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function
from tqdm import tqdm

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


def open_cat(nsim):
    """
    Open a CSiBORG halo catalogue.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    bounds = {"totpartmass": (1e12, None), "dist": (0, 155/0.705)}
    return csiborgtools.read.HaloCatalogue(nsim, paths, bounds=bounds)


def plot_mass_vs_ncells(nsim, pdf=False):
    cat = open_cat(nsim)
    mpart = 4.38304044e+09

    with plt.style.context(utils.mplstyle):
        plt.figure()
        plt.scatter(cat["totpartmass"], cat["lagpatch_ncells"], s=0.25,
                    rasterized=True)
        plt.xscale("log")
        plt.yscale("log")
        for n in [1, 10, 100]:
            plt.axvline(n * 512 * mpart, c="black", ls="--", zorder=0, lw=0.8)
        plt.xlabel(r"$M_{\rm tot} / M_\odot$")
        plt.ylabel(r"$N_{\rm cells}$")

        for ext in ["png"] if pdf is False else ["png", "pdf"]:
            fout = join(utils.fout, f"init_mass_vs_ncells_{nsim}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


###############################################################################
#                        Command line interface                               #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()

    cached_funcs = []
    if args.clean:
        for func in cached_funcs:
            print(f"Cleaning cache for function {func}.")
            delete_disk_caches_for_function(func)

    # plot_mass_vs_occupancy(7444)
    # plot_mass_vs_normcells(7444 + 24 * 4, pdf=False)
    plot_mass_vs_ncells(7444, pdf=True)
