#!/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python

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
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function  # noqa
from tqdm import tqdm

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


def open_csiborg(nsim):
    """
    Open a CSiBORG halo catalogue.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    bounds = {"totpartmass": (None, None), "dist": (0, 155/0.705)}
    return csiborgtools.read.HaloCatalogue(nsim, paths, bounds=bounds)


def open_quijote(nsim, nobs=None):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    cat = csiborgtools.read.QuijoteHaloCatalogue(nsim, paths, nsnap=4)
    if nobs is not None:
        cat = cat.pick_fiducial_observer(nobs, rmax=155.5 / 0.705)
    return cat


def plot_mass_vs_ncells(nsim, pdf=False):
    cat = open_csiborg(nsim)
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


def get_median_errors(x):
    """
    Get the median and errors corresponding to the 16th and 84th percentiles of
    a 2-dimensional array of shape `(nsims, nbins)`.
    """
    pmin = 15.865525393145707
    pmax = 84.13447460685429
    y = numpy.median(x, axis=0)
    ymax = numpy.percentile(x, pmax, axis=0)
    ymin = numpy.percentile(x, pmin, axis=0)
    yerr = numpy.array([y - ymin, ymax - y])
    return y, yerr


def plot_hmf(pdf=False):
    print("Plotting the HMF...", flush=True)
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    csiborg_nsims = paths.get_ics("csiborg")
    print("Loading CSiBORG halo counts.", flush=True)
    for i, nsim in enumerate(tqdm(csiborg_nsims)):
        data = numpy.load(paths.halo_counts("csiborg", nsim))
        if i == 0:
            bins = data["bins"]
            rmax = data["rmax"]
            csiborg_hmf = numpy.full((len(csiborg_nsims), len(bins) - 1),
                                     numpy.nan, dtype=numpy.float32)
        csiborg_hmf[i, :] = data["counts"]
    csiborg_hmf /= numpy.diff(bins).reshape(1, -1)
    csiborg_hmf /= 4 / 3 * numpy.pi * rmax**3

    print("Loading Quijote halo counts.", flush=True)
    quijote_nsims = paths.get_ics("quijote")
    for i, nsim in enumerate(tqdm(quijote_nsims)):
        data = numpy.load(paths.halo_counts("quijote", nsim))
        if i == 0:
            bins = data["bins"]
            rmax = data["rmax"]
            nmax = data["counts"].shape[0]
            quijote_hmf = numpy.full(
                (len(quijote_nsims) * nmax, len(bins) - 1), numpy.nan,
                dtype=numpy.float32)
        quijote_hmf[i * nmax:(i + 1) * nmax, :] = data["counts"]
    quijote_hmf /= numpy.diff(bins).reshape(1, -1)
    quijote_hmf /= 4 / 3 * numpy.pi * rmax**3

    x = 10**(0.5 * (bins[1:] + bins[:-1]))
    # Edit lower limits
    csiborg_hmf[:, x < 1e12] = numpy.nan
    quijote_hmf[:, x < 8e12] = numpy.nan
    # Edit upper limits
    csiborg_hmf[:, x > 4e15] = numpy.nan
    quijote_hmf[:, x > 4e15] = numpy.nan
    with plt.style.context(utils.mplstyle):
        fig, ax = plt.subplots(nrows=2, sharex=True,
                               figsize=(3.5, 2.625 * 1.5),
                               gridspec_kw={"height_ratios": [1, 0.5]})
        fig.subplots_adjust(hspace=0, wspace=0)

        y_csiborg, yerr_csiborg = get_median_errors(csiborg_hmf)
        ax[0].plot(x, y_csiborg, label="CSiBORG")
        ax[0].fill_between(x, y_csiborg - yerr_csiborg[0, :],
                           y_csiborg + yerr_csiborg[1, :], alpha=0.5)

        y_quijote, yerr_quijote = get_median_errors(quijote_hmf)
        ax[0].plot(x, y_quijote, label="Quijote")
        ax[0].fill_between(x, y_quijote - yerr_quijote[0, :],
                           y_quijote + yerr_quijote[1, :], alpha=0.5)

        log_y_csiborg = numpy.log10(y_csiborg)
        std_log_y_csiborg = (yerr_csiborg[0, :] + yerr_csiborg[1, :])
        std_log_y_csiborg /= y_csiborg * numpy.log(10)

        log_y_quijote = numpy.log10(y_quijote)
        std_log_y_quijote = (yerr_quijote[0, :] + yerr_quijote[1, :])
        std_log_y_quijote /= y_quijote * numpy.log(10)

        y = log_y_csiborg - log_y_quijote
        err = numpy.sqrt(std_log_y_csiborg**2 + std_log_y_quijote**2)
        print(err)

        ax[1].plot(x, 10**y)
        ax[1].fill_between(x, 10**(y - err), 10**(y + err), alpha=0.5)
        ax[1].axhline(1, color="k", ls=plt.rcParams["lines.linestyle"],
                      lw=0.5 * plt.rcParams["lines.linewidth"], zorder=0)
        ax[0].set_ylabel(r"$\frac{\mathrm{d}^2 n}{\mathrm{d}\log M_{\rm h}~\mathrm{d}V}~\mathrm{dex}^{-1}~\mathrm{Mpc}^{-3}$")  # noqa
        ax[1].set_xlabel(r"$M_{\rm h}$ [$M_\odot$]")
        ax[1].set_ylabel(r"$\mathrm{CSiBORG} / \mathrm{Quijote}$")

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[0].legend()

        fig.tight_layout(h_pad=0, w_pad=0)
        for ext in ["png"] if pdf is False else ["png", "pdf"]:
            fout = join(utils.fout, f"hmf_comparison.{ext}")
            print(f"Saving to `{fout}`.")
            fig.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
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
    # plot_mass_vs_ncells(7444, pdf=True)
    plot_hmf(pdf=True)
