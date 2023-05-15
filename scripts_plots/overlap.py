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
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    bounds = {"totpartmass": (1e12, None)}
    return csiborgtools.read.HaloCatalogue(nsim, paths, bounds=bounds)


@cache_to_disk(7)
def get_overlap(nsim0):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(nsim0, paths, smoothed=True)
    cat0 = open_cat(nsim0)

    catxs = []
    for nsimx in tqdm(nsimxs):
        catxs.append(open_cat(nsimx))

    reader = csiborgtools.read.NPairsOverlap(cat0, catxs, paths)
    x = reader.cat0("totpartmass")
    summed_overlap = reader.summed_overlap(True)
    prob_nomatch = reader.prob_nomatch(True)
    return x, summed_overlap, prob_nomatch


def plot_summed_overlap(nsim0):
    x, summed_overlap, prob_nomatch = get_overlap(nsim0)

    mean_overlap = numpy.mean(summed_overlap, axis=1)
    std_overlap = numpy.std(summed_overlap, axis=1)

    mean_prob_nomatch = numpy.mean(prob_nomatch, axis=1)
    # std_prob_nomatch = numpy.std(prob_nomatch, axis=1)

    mask = mean_overlap > 0
    x = x[mask]
    mean_overlap = mean_overlap[mask]
    std_overlap = std_overlap[mask]
    mean_prob_nomatch = mean_prob_nomatch[mask]

    # Mean summed overlap
    with plt.style.context(utils.mplstyle):
        plt.figure()
        plt.hexbin(x, mean_overlap, mincnt=1, xscale="log", bins="log",
                   gridsize=50)
        plt.colorbar(label="Counts in bins")
        plt.xlabel(r"$M_{\rm tot} / M_\odot$")
        plt.ylabel(r"$\langle \mathcal{O}_{a}^{\mathcal{A} \mathcal{B}} \rangle_{\mathcal{B}}$")  # noqa

        plt.tight_layout()
        for ext in ["png", "pdf"]:
            fout = join(utils.fout, f"overlap_mean_{nsim0}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()

    # Std summed overlap
    with plt.style.context(utils.mplstyle):
        plt.figure()
        plt.hexbin(x, std_overlap, mincnt=1, xscale="log", bins="log",
                   gridsize=50)
        plt.colorbar(label="Counts in bins")
        plt.xlabel(r"$M_{\rm tot} / M_\odot$")
        plt.ylabel(r"$\delta \left( \mathcal{O}_{a}^{\mathcal{A} \mathcal{B}} \right)_{\mathcal{B}}$")  # noqa
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            fout = join(utils.fout, f"overlap_std_{nsim0}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()

    # 1 - mean summed overlap vs mean prob nomatch
    with plt.style.context(utils.mplstyle):
        plt.figure()
        plt.scatter(1 - mean_overlap, mean_prob_nomatch, c=numpy.log10(x), s=2,
                    rasterized=True)
        plt.colorbar(label=r"$\log_{10} M_{\rm halo} / M_\odot$")

        t = numpy.linspace(0.3, 1, 100)
        plt.plot(t, t, color="red", linestyle="--")

        plt.xlabel(r"$1 - \langle \mathcal{O}_a^{\mathcal{A} \mathcal{B}} \rangle_{\mathcal{B}}$")  # noqa
        plt.ylabel(r"$\langle \eta_a^{\mathcal{A} \mathcal{B}} \rangle_{\mathcal{B}}$")  # noqa
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            fout = join(utils.fout, f"overlap_vs_prob_nomatch_{nsim0}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()

    cached_funcs = ["get_overlap"]
    if args.clean:
        for func in cached_funcs:
            print(f"Cleaning cache for function {func}.")
            delete_disk_caches_for_function(func)

    plot_summed_overlap(7444)
