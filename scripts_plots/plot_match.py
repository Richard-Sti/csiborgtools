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

import matplotlib.pyplot as plt
import numpy
import scienceplots  # noqa
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function
from scipy.stats import kendalltau
from tqdm import tqdm, trange

import csiborgtools
import plt_utils

MASS_KINDS = {"csiborg": "fof_totpartmass",
              "quijote": "group_mass",
              }


def open_cat(nsim, simname):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    if simname == "csiborg":
        bounds = {"dist": (0, 155)}
        cat = csiborgtools.read.CSiBORGHaloCatalogue(
            nsim, paths, bounds=bounds)
    elif simname == "quijote":
        cat = csiborgtools.read.QuijoteHaloCatalogue(
            nsim, paths, nsnap=4, load_fitted=True, load_initial=True,
            with_lagpatch=False)
    else:
        raise ValueError(f"Unknown simulation name: {simname}.")

    return cat


def open_cats(nsims, simname):
    catxs = [None] * len(nsims)

    for i, nsim in enumerate(tqdm(nsims, desc="Opening catalogues")):
        catxs[i] = open_cat(nsim, simname)

    return catxs


@cache_to_disk(120)
def get_overlap_summary(nsim0, simname, min_logmass, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(simname, nsim0, paths,
                                              min_logmass, smoothed=smoothed)
    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.summary.NPairsOverlap(cat0, catxs, paths,
                                                min_logmass)
    mass0 = reader.cat0(MASS_KINDS[simname])
    mask = mass0 > 10**min_logmass

    return {"mass0": mass0[mask],
            "hid0": reader.cat0("index")[mask],
            "summed_overlap": reader.summed_overlap(smoothed)[mask],
            "max_overlap": reader.max_overlap(0, smoothed)[mask],
            "prob_nomatch": reader.prob_nomatch(smoothed)[mask],
            }


@cache_to_disk(120)
def get_expected_mass(nsim0, simname, min_overlap, min_logmass, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(simname, nsim0, paths,
                                              min_logmass, smoothed=True)[:3]

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.summary.NPairsOverlap(cat0, catxs, paths,
                                                min_logmass)
    mass0 = reader.cat0(MASS_KINDS[simname])
    mask = mass0 > 10**min_logmass
    mu, std = reader.counterpart_mass(
        from_smoothed=True, overlap_threshold=min_overlap, return_full=False)

    return {"mass0": mass0[mask],
            "mu": mu[mask],
            "std": std[mask],
            "prob_nomatch": reader.prob_nomatch(smoothed)[mask],
            }

# --------------------------------------------------------------------------- #
###############################################################################
#                   Total DM halo mass vs pair overlaps                       #
###############################################################################
# --------------------------------------------------------------------------- #


@cache_to_disk(120)
def get_mtot_vs_all_pairoverlap(nsim0, simname, mass_kind, min_logmass,
                                smoothed, nbins):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(simname, nsim0, paths,
                                              min_logmass, smoothed=smoothed)
    nsimxs = nsimxs

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.summary.NPairsOverlap(cat0, catxs, paths,
                                                min_logmass)

    x = [None] * len(catxs)
    y = [None] * len(catxs)
    for i in trange(len(catxs), desc="Stacking catalogues"):
        x[i] = numpy.log10(
            numpy.concatenate(reader[i].copy_per_match(mass_kind)))
        y[i] = numpy.concatenate(reader[i].overlap(smoothed))

    x = numpy.concatenate(x)
    y = numpy.concatenate(y)

    xbins = numpy.linspace(min(x), max(x), nbins)

    return x, y, xbins


def mtot_vs_all_pairoverlap(nsim0, simname, min_logmass, smoothed, nbins,
                            ext="png"):
    mass_kind = MASS_KINDS[simname]
    x, y, xbins = get_mtot_vs_all_pairoverlap(nsim0, simname, mass_kind,
                                              min_logmass, smoothed, nbins)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        hb = plt.hexbin(x, y, mincnt=1, gridsize=50, bins="log")

        y_median, yerr = plt_utils.compute_error_bars(x, y, xbins, sigma=2)
        plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                     color='red', ls='dashed', capsize=3,
                     label="CSiBORG" if simname == "csiborg" else None)

        if simname == "csiborg":
            x_quijote, y_quijote, xbins_quijote = get_mtot_vs_all_pairoverlap(
                0, "quijote", "group_mass", min_logmass, smoothed, nbins)
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                x_quijote, y_quijote, xbins_quijote, sigma=2)
            plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                         y_median_quijote, yerr=yerr_quijote,
                         color='blue', ls='dashed', capsize=3,
                         label="Quijote")
            plt.legend(ncol=2, fontsize="small")

        plt.colorbar(hb, label="Counts in bins")
        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        plt.ylabel("Pair overlap")
        plt.xlim(numpy.min(x))
        plt.ylim(0., 1.)

        plt.tight_layout()
        fout = join(plt_utils.fout,
                    f"mass_vs_pair_overlap_{simname}_{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#                  Total DM halo mass vs maximum pair overlaps                #
###############################################################################
# --------------------------------------------------------------------------- #


@cache_to_disk(120)
def get_mtot_vs_maxpairoverlap(nsim0, simname, mass_kind, min_logmass,
                               smoothed, nbins):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(simname, nsim0, paths,
                                              min_logmass, smoothed=smoothed)
    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    def get_max(y_):
        if len(y_) == 0:
            return 0
        return numpy.nanmax(y_)

    reader = csiborgtools.summary.NPairsOverlap(cat0, catxs, paths,
                                                min_logmass)

    x = [None] * len(catxs)
    y = [None] * len(catxs)
    for i in trange(len(catxs), desc="Stacking catalogues"):
        x[i] = numpy.log10(cat0[mass_kind])
        y[i] = numpy.array([get_max(y_) for y_ in reader[i].overlap(smoothed)])

        mask = x[i] > min_logmass
        x[i] = x[i][mask]
        y[i] = y[i][mask]

    x = numpy.concatenate(x)
    y = numpy.concatenate(y)

    xbins = numpy.linspace(min(x), max(x), nbins)

    return x, y, xbins


def mtot_vs_maxpairoverlap(nsim0, simname, mass_kind, min_logmass, smoothed,
                           nbins, ext="png"):
    x, y, xbins = get_mtot_vs_maxpairoverlap(nsim0, simname, mass_kind,
                                             min_logmass, smoothed, nbins)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        plt.hexbin(x, y, mincnt=1, gridsize=50, bins="log")

        y_median, yerr = plt_utils.compute_error_bars(x, y, xbins, sigma=2)
        plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                     color='red', ls='dashed', capsize=3,
                     label="CSiBORG" if simname == "csiborg" else None)

        if simname == "csiborg":
            x_quijote, y_quijote, xbins_quijote = get_mtot_vs_all_pairoverlap(
                0, "quijote", "group_mass", min_logmass, smoothed, nbins)
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                x_quijote, y_quijote, xbins_quijote, sigma=2)
            plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                         y_median_quijote, yerr=yerr_quijote,
                         color='blue', ls='dashed', capsize=3,
                         label="Quijote")
            plt.legend(ncol=2, fontsize="small")

        plt.colorbar(label="Counts in bins")
        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        plt.ylabel("Maximum pair overlap")
        plt.ylim(-0.02, 1.)
        plt.xlim(numpy.min(x) - 0.05)

        plt.tight_layout()
        fout = join(plt_utils.fout, f"mass_vs_max_pair_overlap{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#            Total DM halo mass vs maximum pair overlap consistency           #
###############################################################################
# --------------------------------------------------------------------------- #

@cache_to_disk(120)
def get_mtot_vs_maxpairoverlap_consistency(nsim0, simname, mass_kind,
                                           min_logmass, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(simname, nsim0, paths,
                                              min_logmass, smoothed=smoothed)
    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.summary.NPairsOverlap(cat0, catxs, paths,
                                                min_logmass)

    x = numpy.log10(cat0[mass_kind])
    mask = x > min_logmass
    x = x[mask]
    nhalos = len(x)

    y = numpy.full((len(catxs), nhalos), numpy.nan)
    for i in trange(len(catxs), desc="Stacking catalogues"):
        overlaps = reader[i].overlap(smoothed)
        for j in range(nhalos):
            # if len(overlaps[j]) > 0:
            y[i, j] = numpy.sum(overlaps[j])

    return x, y


def mtot_vs_maxpairoverlap_consistency(nsim0, simname, mass_kind, min_logmass,
                                       smoothed, ext="png"):
    left_edges = numpy.arange(min_logmass, 15, 0.1)
    # delete_disk_caches_for_function("get_mtot_vs_maxpairoverlap_consistency")
    x, y0 = get_mtot_vs_maxpairoverlap_consistency(nsim0, simname, mass_kind,
                                                   min_logmass, smoothed)
    nsims, nhalos = y0.shape
    x_2, y0_2 = get_mtot_vs_maxpairoverlap_consistency(
        0, "quijote", "group_mass", min_logmass, smoothed)
    nsims2, nhalos = y0_2.shape

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()

        yplot = numpy.full(len(left_edges), numpy.nan)
        yplot_2 = numpy.full(len(left_edges), numpy.nan)

        for ymin in [0.3]:

            y = numpy.sum(y0 > ymin, axis=0) / nsims
            y_2 = numpy.sum(y0_2 > ymin, axis=0) / nsims2

            for i, left_edge in enumerate(left_edges):
                mask = x > left_edge
                yplot[i] = numpy.mean(y[mask]) #/ nsims
                mask = x_2 > left_edge
                yplot_2[i] = numpy.mean(y_2[mask]) #/ nsims

        plt.plot(left_edges, yplot, label="CSiBORG")
        plt.plot(left_edges, yplot_2, label="Quijote")
        plt.legend()
        # y2 = numpy.concatenate(y0)
        # y2 = y2[y2 > 0]
        # m = y0 > 0
        # plt.hist(y0[m], bins=30, density=True, histtype="step")
        # m = y0_2 > 0
        # plt.hist(y0_2[m], bins=30, density=True, histtype="step")
        # plt.yscale("log")


        plt.tight_layout()
        fout = join(
            plt_utils.fout,
            f"mass_vs_max_pair_overlap_consistency_{simname}_{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#                  Total DM halo mass vs summed pair overlaps                 #
###############################################################################
# --------------------------------------------------------------------------- #


def mtot_vs_summedpairoverlap(nsim0, simname, min_logmass, smoothed, nbins,
                              ext="png"):
    x = get_overlap_summary(nsim0, simname, min_logmass, smoothed)

    mass0 = numpy.log10(x["mass0"])
    mean_overlap = numpy.nanmean(x["summed_overlap"], axis=1)
    std_overlap = numpy.nanstd(x["summed_overlap"], axis=1)
    mean_prob_nomatch = numpy.nanmean(x["prob_nomatch"], axis=1)

    xbins = numpy.linspace(numpy.nanmin(mass0), numpy.nanmax(mass0), nbins)

    with plt.style.context(plt_utils.mplstyle):
        fig, axs = plt.subplots(ncols=3, figsize=(3.5 * 2, 2.625))
        im1 = axs[0].hexbin(mass0, mean_overlap, mincnt=1, bins="log",
                            gridsize=30)

        y_median, yerr = plt_utils.compute_error_bars(
            mass0, mean_overlap, xbins, sigma=2)
        axs[0].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                        color='red', ls='dashed', capsize=3)

        im2 = axs[1].hexbin(mass0, std_overlap, mincnt=1, bins="log",
                            gridsize=30)

        y_median, yerr = plt_utils.compute_error_bars(
            mass0, std_overlap, xbins, sigma=2)
        axs[1].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                        color='red', ls='dashed', capsize=3)

        if simname == "csiborg":
            x_quijote = get_overlap_summary(0, "quijote", min_logmass,
                                            smoothed)
            mass0_quijote = numpy.log10(x_quijote["mass0"])
            mean_overlap_quijote = numpy.nanmean(x_quijote["summed_overlap"],
                                                 axis=1)
            std_overlap_quijote = numpy.nanstd(x_quijote["summed_overlap"],
                                               axis=1)
            xbins_quijote = numpy.linspace(numpy.nanmin(mass0),
                                           numpy.nanmax(mass0), nbins)

            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                mass0_quijote, mean_overlap_quijote, xbins_quijote, sigma=2)
            axs[0].errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                            y_median_quijote, yerr=yerr_quijote,
                            color='blue', ls='dashed', capsize=3)

            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                mass0_quijote, std_overlap_quijote, xbins_quijote, sigma=2)
            axs[1].errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                            y_median_quijote, yerr=yerr_quijote,
                            color='blue', ls='dashed', capsize=3)

        im3 = axs[2].scatter(1 - mean_overlap, mean_prob_nomatch, c=mass0,
                             s=2, rasterized=True)

        t = numpy.linspace(numpy.nanmin(1 - mean_overlap), 1, 100)
        axs[2].plot(t, t, color="red", linestyle="--")

        axs[0].set_ylim(0., 0.75)
        axs[0].set_xlim(numpy.min(mass0))
        axs[0].set_xlim(numpy.min(mass0))
        axs[0].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[0].set_ylabel("Mean summed overlap")
        axs[1].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[1].set_ylabel("Uncertainty of summed overlap")
        axs[2].set_xlabel(r"$1 - $ mean summed overlap")
        axs[2].set_ylabel("Mean prob. of no match")

        label = ["Bin counts", "Bin counts",
                 r"$\log M_{\rm tot} ~ [M_\odot / h]$"]
        ims = [im1, im2, im3]
        for i in range(3):
            axins = axs[i].inset_axes([0.0, 1.0, 1.0, 0.05])
            fig.colorbar(ims[i], cax=axins, orientation="horizontal",
                         label=label[i])
            axins.xaxis.tick_top()
            axins.xaxis.set_tick_params(labeltop=True)
            axins.xaxis.set_label_position("top")

        fig.tight_layout()
        fout = join(plt_utils.fout, f"overlap_stat_{simname}_{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#                  Total DM halo mass vs mean maximum overlap                 #
###############################################################################
# --------------------------------------------------------------------------- #


def mtot_vs_mean_max_overlap(nsim0, simname, min_logmass, smoothed, nbins):
    x = get_overlap_summary(nsim0, simname, min_logmass, smoothed)

    mass0 = numpy.log10(x["mass0"])
    max_overlap = x["max_overlap"]

    xbins = numpy.linspace(numpy.nanmin(mass0), numpy.nanmax(mass0), nbins)

    mean_max_overlap = numpy.nanmean(max_overlap, axis=1)
    std_max_overlap = numpy.nanstd(max_overlap, axis=1)

    with plt.style.context(plt_utils.mplstyle):
        fig, axs = plt.subplots(ncols=3, figsize=(3.5 * 2, 2.625))

        im1 = axs[0].hexbin(mass0, mean_max_overlap, gridsize=30, mincnt=1,
                            bins="log")
        y_median, yerr = plt_utils.compute_error_bars(
            mass0, mean_max_overlap, xbins, sigma=2)
        axs[0].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                        color='red', ls='dashed', capsize=3,
                        label="CSiBORG" if simname == "csiborg" else None)

        im2 = axs[1].hexbin(mass0, std_max_overlap, gridsize=30, mincnt=1,
                            bins="log")
        y_median, yerr = plt_utils.compute_error_bars(
            mass0, std_max_overlap, xbins, sigma=2)
        axs[1].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                        color='red', ls='dashed', capsize=3)

        if simname == "csiborg":
            x_quijote = get_overlap_summary(0, "quijote", min_logmass,
                                            smoothed)
            mass0_quijote = numpy.log10(x_quijote["mass0"])
            max_overlap_quijote = x_quijote["max_overlap"]

            xbins_quijote = numpy.linspace(numpy.nanmin(mass0_quijote),
                                           numpy.nanmax(mass0_quijote), nbins)
            mean_max_overlap_quijote = numpy.nanmean(max_overlap_quijote,
                                                     axis=1)
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                mass0_quijote, mean_max_overlap_quijote, xbins_quijote,
                sigma=2)
            axs[0].errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                            y_median_quijote, yerr=yerr_quijote,
                            color='blue', ls='dashed', capsize=3,
                            label="Quijote")

            std_max_overlap_quijote = numpy.nanstd(max_overlap_quijote,
                                                   axis=1)
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                mass0_quijote, std_max_overlap_quijote, xbins_quijote,
                sigma=2)
            axs[1].errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                            y_median_quijote, yerr=yerr_quijote,
                            color='blue', ls='dashed', capsize=3)

            axs[0].legend(fontsize="small")

        im3 = axs[2].hexbin(numpy.nanmean(max_overlap, axis=1),
                            numpy.nanstd(max_overlap, axis=1), gridsize=30,
                            C=mass0, reduce_C_function=numpy.nanmean)

        axs[0].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[0].set_ylim(0.0, 0.75)
        axs[0].set_ylabel(r"Mean max. pair overlap")
        axs[1].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[1].set_ylabel(r"Uncertainty of max. pair overlap")
        axs[2].set_xlabel(r"Mean max. pair overlap")
        axs[2].set_ylabel(r"Uncertainty of max. pair overlap")

        label = ["Bin counts", "Bin counts",
                 r"$\log M_{\rm tot} ~ [M_\odot / h]$"]
        ims = [im1, im2, im3]
        for i in range(3):
            axins = axs[i].inset_axes([0.0, 1.0, 1.0, 0.05])
            fig.colorbar(ims[i], cax=axins, orientation="horizontal",
                         label=label[i])
            axins.xaxis.tick_top()
            axins.xaxis.set_tick_params(labeltop=True)
            axins.xaxis.set_label_position("top")

        fig.tight_layout()
        fout = join(plt_utils.fout, f"max_pairoverlap_{simname}_{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#                  Total DM halo mass vs mean separation                      #
###############################################################################
# --------------------------------------------------------------------------- #

@cache_to_disk(120)
def get_mass_vs_separation(nsim0, nsimx, simname, min_logmass, boxsize,
                           smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    cat0 = open_cat(nsim0, simname)
    catx = open_cat(nsimx, simname)

    reader = csiborgtools.summary.PairOverlap(cat0, catx, paths, min_logmass)

    mass = numpy.log10(reader.cat0(MASS_KINDS[simname]))
    dist = reader.dist(in_initial=False, boxsize=boxsize, norm_kind="r200c")
    overlap = reader.overlap(smoothed)
    dist = csiborgtools.summary.weighted_stats(dist, overlap, min_weight=0)

    mask = numpy.isfinite(dist[:, 0])
    mass = mass[mask]
    dist = dist[mask, :]
    dist = numpy.log10(dist)

    return mass, dist


def mass_vs_separation(nsim0, nsimx, simname, min_logmass, nbins, smoothed,
                       boxsize, plot_std):
    mass, dist = get_mass_vs_separation(nsim0, nsimx, simname, min_logmass,
                                        boxsize, smoothed)
    xbins = numpy.linspace(numpy.nanmin(mass), numpy.nanmax(mass), nbins)
    y = dist[:, 0] if not plot_std else dist[:, 1]

    with plt.style.context(plt_utils.mplstyle):
        fig, ax = plt.subplots()

        cx = ax.hexbin(mass, y, mincnt=1, bins="log", gridsize=50)
        y_median, yerr = plt_utils.compute_error_bars(mass, y, xbins, sigma=2)
        ax.errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                    color='red', ls='dashed', capsize=3,
                    label="CSiBORG" if simname == "csiborg" else None)

        if simname == "csiborg":
            mass_quijote, dist_quijote = get_mass_vs_separation(
                0, 1, "quijote", min_logmass, boxsize, smoothed)
            xbins_quijote = numpy.linspace(numpy.nanmin(mass_quijote),
                                           numpy.nanmax(mass_quijote), nbins)
            if not plot_std:
                y_quijote = dist_quijote[:, 0]
            else:
                y_quijote = dist_quijote[:, 1]
            print(mass_quijote)
            print(y_quijote)
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                mass_quijote, y_quijote, xbins_quijote, sigma=2)
            ax.errorbar(0.5 * (xbins_quijote[1:] + xbins_quijote[:-1]),
                        y_median_quijote, yerr=yerr_quijote, color='blue',
                        ls='dashed',
                        capsize=3, label="Quijote")
            ax.legend(fontsize="small", loc="upper left")

        if not plot_std:
            ax.set_ylabel(r"$\log \langle \Delta R / R_{\rm 200c}\rangle$")
        else:
            ax.set_ylabel(
                r"$\delta \log \langle \Delta R / R_{\rm 200c}\rangle$")

        fig.colorbar(cx, label="Bin counts")
        ax.set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        ax.set_ylabel(r"$\log \langle \Delta R / R_{\rm 200c}\rangle$")

        fig.tight_layout()
        fout = join(plt_utils.fout,
                    f"mass_vs_sep_{simname}_{nsim0}_{nsimx}.{ext}")
        if plot_std:
            fout = fout.replace(f".{ext}", f"_std.{ext}")
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#               Total DM halo mass vs max overlap separation                  #
###############################################################################


@cache_to_disk(120)
def get_mass_vs_max_overlap_separation(nsim0, nsimx, simname, min_logmass,
                                       boxsize, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    cat0 = open_cat(nsim0, simname)
    catx = open_cat(nsimx, simname)

    reader = csiborgtools.summary.PairOverlap(cat0, catx, paths, min_logmass)

    mass = numpy.log10(reader.cat0(MASS_KINDS[simname]))
    dist = reader.dist(in_initial=False, boxsize=boxsize, norm_kind="r200c")
    overlap = reader.overlap(smoothed)

    maxoverlap_dist = numpy.full(len(cat0), numpy.nan, dtype=numpy.float32)

    for i in range(len(cat0)):
        if len(overlap[i]) == 0:
            continue
        imax = numpy.argmax(overlap[i])
        maxoverlap_dist[i] = dist[i][imax]

    mask = numpy.isfinite(maxoverlap_dist)
    mass = mass[mask]
    maxoverlap_dist = numpy.log10(maxoverlap_dist[mask])
    return mass, maxoverlap_dist


def mass_vs_maxoverlap_separation(nsim0, nsimx, simname, min_logmass, nbins,
                                  smoothed, boxsize, plot_std):
    mass, y = get_mass_vs_max_overlap_separation(
        nsim0, nsimx, simname, min_logmass, boxsize, smoothed)
    xbins = numpy.linspace(numpy.nanmin(mass), numpy.nanmax(mass), nbins)

    with plt.style.context(plt_utils.mplstyle):
        fig, ax = plt.subplots()

        cx = ax.hexbin(mass, y, mincnt=1, bins="log", gridsize=50)
        y_median, yerr = plt_utils.compute_error_bars(mass, y, xbins, sigma=2)
        ax.errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                    color='red', ls='dashed', capsize=3,
                    label="CSiBORG" if simname == "csiborg" else None)

        if simname == "csiborg":
            mass_quijote, y_quijote = get_mass_vs_max_overlap_separation(
                0, 1, "quijote", min_logmass, boxsize, smoothed)
            xbins_quijote = numpy.linspace(numpy.nanmin(mass_quijote),
                                           numpy.nanmax(mass_quijote), nbins)
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                mass_quijote, y_quijote, xbins_quijote, sigma=2)
            ax.errorbar(0.5 * (xbins_quijote[1:] + xbins_quijote[:-1]),
                        y_median_quijote, yerr=yerr_quijote, color='blue',
                        ls='dashed',
                        capsize=3, label="Quijote")
            ax.legend(fontsize="small", loc="upper left")

        fig.colorbar(cx, label="Bin counts")
        ax.set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        ax.set_ylabel(r"$\log (\Delta R_{\mathcal{O}_{\max}} / R_{\rm 200c})$")

        fig.tight_layout()
        fout = join(plt_utils.fout,
                    f"mass_vs_maxoverlap_sep_{simname}_{nsim0}_{nsimx}.{ext}")
        if plot_std:
            fout = fout.replace(f".{ext}", f"_std.{ext}")
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#                Total DM halo mass vs maximum overlap matched mass           #
###############################################################################
# --------------------------------------------------------------------------- #


@cache_to_disk(120)
def get_property_maxoverlap(nsim0, simname, min_logmass, key, min_overlap,
                            smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(simname,  nsim0, paths,
                                              min_logmass, smoothed=smoothed)

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.summary.NPairsOverlap(cat0, catxs, paths,
                                                min_logmass)
    mass0 = reader.cat0(MASS_KINDS[simname])
    mask = mass0 > 10**min_logmass

    max_overlap = reader.max_overlap(min_overlap, smoothed)[mask]
    prop_maxoverlap = reader.max_overlap_key(key, min_overlap, smoothed)[mask]

    return {"mass0": mass0[mask],
            "prop0": reader.cat0(key)[mask],
            "max_overlap": max_overlap,
            "prop_maxoverlap": prop_maxoverlap,
            }


def mtot_vs_maxoverlap_mass(nsim0, simname, min_logmass, smoothed, nbins,
                            min_overlap=0, ext="png"):
    mass_kind = MASS_KINDS[simname]
    x = get_property_maxoverlap(nsim0, simname, min_logmass, mass_kind,
                                min_overlap, smoothed)

    mass0 = numpy.log10(x["mass0"])
    stat = numpy.log10(x["prop_maxoverlap"])
    weight = x["max_overlap"]

    mu = plt_utils.nan_weighted_average(stat, weight, axis=1)
    std = plt_utils.nan_weighted_std(stat, weight, axis=1)

    xbins = numpy.linspace(*numpy.percentile(mass0, [0, 100]), nbins)

    with plt.style.context(plt_utils.mplstyle):
        fig, axs = plt.subplots(ncols=2, figsize=(3.5 * 1.75, 2.625))

        m = numpy.isfinite(mass0) & numpy.isfinite(mu)

        im0 = axs[0].hexbin(mass0, mu, mincnt=1, bins="log", gridsize=50)
        y_median, yerr = plt_utils.compute_error_bars(mass0[m], mu[m], xbins,
                                                      sigma=2)
        axs[0].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                        color='red', ls='dashed', capsize=3)

        im1 = axs[1].hexbin(mass0, std, mincnt=1, bins="log", gridsize=50)
        y_median, yerr = plt_utils.compute_error_bars(mass0[m], std[m], xbins,
                                                      sigma=2)
        axs[1].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                        color='red', ls='dashed', capsize=3)

        print("True to expectation corr: ", kendalltau(mass0[m], mu[m]))

        t = numpy.linspace(*numpy.percentile(mass0, [0, 100]), 1000)
        axs[0].plot(t, t, color="blue", linestyle="--")
        axs[0].plot(t, t + 0.2, color="blue", linestyle="--", alpha=0.5)
        axs[0].plot(t, t - 0.2, color="blue", linestyle="--", alpha=0.5)

        axs[0].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[1].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[0].set_ylabel(
            r"Max. overlap mean of $\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[1].set_ylabel(
            r"Max. overlap std. of $\log M_{\rm tot} ~ [M_\odot / h]$")

        ims = [im0, im1]
        for i in range(2):
            axins = axs[i].inset_axes([0.0, 1.0, 1.0, 0.05])
            fig.colorbar(ims[i], cax=axins, orientation="horizontal",
                         label="Bin counts")
            axins.xaxis.tick_top()
            axins.xaxis.set_tick_params(labeltop=True)
            axins.xaxis.set_label_position("top")

        fig.tight_layout()
        fout = join(plt_utils.fout,
                    f"max_totpartmass_{simname}_{nsim0}_{min_logmass}_{min_overlap}.{ext}")  # noqa
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.show()


# --------------------------------------------------------------------------- #
###############################################################################
#                Total DM halo mass vs expected matched mass                  #
###############################################################################
# --------------------------------------------------------------------------- #


def mtot_vs_expected_mass(nsim0, simname, min_logmass, smoothed, nbins,
                          min_overlap=0, max_prob_nomatch=1, ext="png"):
    x = get_expected_mass(nsim0, simname, min_overlap, min_logmass, smoothed)

    mass = x["mass0"]
    mu = x["mu"]
    std = x["std"]
    prob_nomatch = x["prob_nomatch"]

    mass = numpy.log10(mass)
    prob_nomatch = numpy.nanmedian(prob_nomatch, axis=1)

    mask = numpy.isfinite(mass) & numpy.isfinite(mu)
    mask &= (prob_nomatch <= max_prob_nomatch)

    xbins = numpy.linspace(*numpy.percentile(mass[mask], [0, 100]), nbins)

    with plt.style.context(plt_utils.mplstyle):
        fig, axs = plt.subplots(ncols=3, figsize=(3.5 * 2, 2.625))

        im0 = axs[0].hexbin(mass[mask], mu[mask], mincnt=1, bins="log",
                            gridsize=50,)
        y_median, yerr = plt_utils.compute_error_bars(mass[mask], mu[mask],
                                                      xbins, sigma=2)
        axs[0].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                        color='red', ls='dashed', capsize=3)

        im1 = axs[1].hexbin(mass[mask], std[mask], mincnt=1, bins="log",
                            gridsize=50)
        y_median, yerr = plt_utils.compute_error_bars(mass[mask], std[mask],
                                                      xbins, sigma=2)
        axs[1].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                        color='red', ls='dashed', capsize=3)

        im2 = axs[2].hexbin(1 - prob_nomatch[mask], mass[mask] - mu[mask],
                            gridsize=50, C=mass[mask],
                            reduce_C_function=numpy.nanmedian)

        axs[2].axhline(0, color="red", linestyle="--", alpha=0.5)
        axs[0].set_xlabel(r"Reference $\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[0].set_ylabel(r"Expected $\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[1].set_xlabel(r"Reference $\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[1].set_ylabel(r"Std. of $\sigma_{\log M_{\rm tot}}$")
        axs[2].set_xlabel(r"1 - median prob. of no match")
        axs[2].set_ylabel(r"$\log M_{\rm tot} - \log M_{\rm tot, exp}$")

        t = numpy.linspace(*numpy.percentile(mass[mask], [0, 100]), 1000)
        axs[0].plot(t, t, color="blue", linestyle="--")
        axs[0].plot(t, t + 0.2, color="blue", linestyle="--", alpha=0.5)
        axs[0].plot(t, t - 0.2, color="blue", linestyle="--", alpha=0.5)

        ims = [im0, im1, im2]
        labels = ["Bin counts", "Bin counts",
                  r"$\log M_{\rm tot} ~ [M_\odot / h]$"]
        for i in range(3):
            axins = axs[i].inset_axes([0.0, 1.0, 1.0, 0.05])
            fig.colorbar(ims[i], cax=axins, orientation="horizontal",
                         label=labels[i])
            axins.xaxis.tick_top()
            axins.xaxis.set_tick_params(labeltop=True)
            axins.xaxis.set_label_position("top")

        fig.tight_layout()
        fout = join(
            plt_utils.fout,
            f"mass_vs_expmass_{nsim0}_{simname}_{max_prob_nomatch}.{ext}"
            )
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#               Total DM halo mass vs maximum overlap halo property           #
###############################################################################
# --------------------------------------------------------------------------- #


def mtot_vs_maxoverlap_property(nsim0, simname, min_logmass, key, min_overlap,
                                smoothed):
    mass_kind = MASS_KINDS[simname]
    assert key != mass_kind

    x = get_property_maxoverlap(nsim0, simname, min_logmass, key, min_overlap,
                                smoothed)
    mass0 = x["mass0"]
    prop0 = x["prop0"]
    stat = x["prop_maxoverlap"]

    xlabels = {"lambda200c": r"\log \lambda_{\rm 200c}"}
    key_label = xlabels.get(key, key)

    mass0 = numpy.log10(mass0)
    prop0 = numpy.log10(prop0)

    mu = numpy.nanmean(stat, axis=1)
    std = numpy.nanstd(numpy.log10(stat), axis=1)
    mu = numpy.log10(mu)

    with plt.style.context(plt_utils.mplstyle):
        fig, axs = plt.subplots(ncols=3, figsize=(3.5 * 2, 2.625))

        im0 = axs[0].hexbin(mass0, mu - prop0, mincnt=1, bins="log",
                            gridsize=30)
        im1 = axs[1].hexbin(mass0, std, mincnt=1, bins="log", gridsize=30)
        im2 = axs[2].hexbin(prop0, mu, mincnt=1, bins="log", gridsize=30)
        m = numpy.isfinite(prop0) & numpy.isfinite(mu)
        print("True to expectation corr: ", kendalltau(prop0[m], mu[m]))

        axs[0].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[0].set_ylabel(r"True - max. overlap mean of ${}$"
                          .format(key_label))
        axs[1].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[1].set_ylabel(r"Max. overlap std. of ${}$".format(key_label))
        axs[2].set_xlabel(r"${}$".format(key_label))
        axs[2].set_ylabel(r"Max. overlap mean of ${}$".format(key_label))

        t = numpy.linspace(*numpy.percentile(prop0[m], [0, 100]), 1000)
        axs[2].plot(t, t, color="blue", linestyle="--")
        axs[2].plot(t, t + 0.2, color="blue", linestyle="--", alpha=0.5)
        axs[2].plot(t, t - 0.2, color="blue", linestyle="--", alpha=0.5)

        ims = [im0, im1, im2]
        for i in range(3):
            axins = axs[i].inset_axes([0.0, 1.0, 1.0, 0.05])
            fig.colorbar(ims[i], cax=axins, orientation="horizontal",
                         label="Bin counts")
            axins.xaxis.tick_top()
            axins.xaxis.set_tick_params(labeltop=True)
            axins.xaxis.set_label_position("top")

        fig.tight_layout()
        fout = join(plt_utils.fout,
                    f"max_{key}_{simname}_{nsim0}_{min_overlap}.{ext}")
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#                      Max's matching vs overlap success                      #
###############################################################################
# --------------------------------------------------------------------------- #


@cache_to_disk(120)
def get_matching_max_vs_overlap(simname, nsim0, min_logmass, mult):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    nsimsx = [nsim for nsim in paths.get_ics(simname) if nsim != nsim0]
    for i in trange(len(nsimsx), desc="Loading data"):
        nsimx = nsimsx[i]
        fpath = paths.match_max(simname, nsim0, nsimx, min_logmass,
                                mult=mult)

        data = numpy.load(fpath, allow_pickle=True)

        if i == 0:
            mass0 = data["mass0"]
            max_overlap = numpy.full((mass0.size, len(nsimsx)), numpy.nan)
            match_overlap = numpy.full((mass0.size, len(nsimsx)), numpy.nan)
            success = numpy.zeros((mass0.size, len(nsimsx)), numpy.bool_)

        max_overlap[:, i] = data["max_overlap"]
        match_overlap[:, i] = data["match_overlap"]
        success[:, i] = data["success"]

    return {"mass0": mass0, "max_overlap": max_overlap,
            "match_overlap": match_overlap, "success": success}


def matching_max_vs_overlap(simname, nsim0, min_logmass):
    left_edges = numpy.arange(min_logmass, 15, 0.1)
    nsims = 100 if simname == "csiborg" else 9

    with plt.style.context("science"):
        fig, axs = plt.subplots(ncols=2, figsize=(3.5 * 2, 2.625))
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for n, mult in enumerate([2.5, 5., 7.5, 10.0]):
            x = get_matching_max_vs_overlap(simname,
                                            nsim0, min_logmass, mult=mult)
            mass0 = numpy.log10(x["mass0"])
            max_overlap = x["max_overlap"]
            match_overlap = x["match_overlap"]
            success = x["success"]

            nbins = len(left_edges)
            y = numpy.full((nbins, nsims), numpy.nan)
            y2 = numpy.full(nbins, numpy.nan)
            y2err = numpy.full(nbins, numpy.nan)
            for i in range(nbins):
                m = mass0 > left_edges[i]
                for j in range(nsims):
                    y[i, j] = numpy.sum(
                        max_overlap[m, j] == match_overlap[m, j])
                    y[i, j] /= numpy.sum(success[m, j])

                y2[i] = numpy.mean(numpy.sum(success[m, :], axis=1) / nsims)
                y2err[i] = numpy.std(numpy.sum(success[m, :], axis=1) / nsims)

            offset = numpy.random.normal(0, 0.015)

            ysummary = numpy.percentile(y, [16, 50, 84], axis=1)
            axs[0].errorbar(
                left_edges + offset, ysummary[1],
                yerr=[ysummary[1] - ysummary[0], ysummary[2] - ysummary[1]],
                capsize=4, c=cols[n], ls="dashed",
                label=r"$\leq {}~R_{{\rm 200c}}$".format(mult), errorevery=2)

            axs[1].errorbar(left_edges + offset, y2, yerr=y2err,
                            capsize=4, errorevery=2, c=cols[n], ls="dashed")

        axs[0].legend(ncols=2, fontsize="small")
        for i in range(2):
            axs[i].set_xlabel(r"$\log M_{\rm tot, min} ~ [M_\odot / h]$")

        axs[1].set_ylim(0)
        axs[0].set_ylabel(r"$f_{\rm agreement}$")
        axs[1].set_ylabel(r"$f_{\rm match}$")

        fig.tight_layout()
        fout = join(
            plt_utils.fout,
            f"matching_max_agreement_{simname}_{nsim0}_{min_logmass}.png")
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#                       KL final snapshot vs overlaps                         #
###############################################################################
# --------------------------------------------------------------------------- #


# def plot_kl_vs_overlap(runs, nsim, kwargs, runs_to_mass, plot_std=True,
#                        upper_threshold=False):
#     """
#     Plot KL divergence vs overlap for CSiBORG.
#
#     Parameters
#     ----------
#     runs : str
#         Run names.
#     nsim : int
#         Simulation index.
#     kwargs : dict
#         Nearest neighbour reader keyword arguments.
#     runs_to_mass : dict
#         Dictionary mapping run names to total halo mass range.
#     plot_std : bool, optional
#         Whether to plot the standard deviation of the overlap distribution.
#     upper_threshold : bool, optional
#         Whether to enforce an upper threshold on halo mass.
#
#     Returns
#     -------
#     None
#     """
#     paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
#     nn_reader = csiborgtools.summary.NearestNeighbourReader(**kwargs, paths=paths)
#
#     xs, ys1, ys2, cs = [], [], [], []
#     for run in runs:
#         nn_data = nn_reader.read_single("csiborg", run, nsim, nobs=None)
#         nn_hindxs = nn_data["ref_hindxs"]
#         mass, overlap_hindxs, __, summed_overlap, prob_nomatch = get_overlap_summary("csiborg", nsim)  # noqa
#
#         # We need to match the hindxs between the two.
#         hind2overlap_array = {hind: i for i, hind in enumerate(overlap_hindxs)}
#         mask = numpy.asanyarray([hind2overlap_array[hind]
#                                  for hind in nn_hindxs])
#         summed_overlap = summed_overlap[mask]
#         prob_nomatch = prob_nomatch[mask]
#         mass = mass[mask]
#
#         x = make_kl("csiborg", run, nsim, nobs=None, kwargs=kwargs)
#         y1 = 1 - numpy.mean(prob_nomatch, axis=1)
#         y2 = numpy.std(prob_nomatch, axis=1)
#         cmin, cmax = make_binlims(run, runs_to_mass, upper_threshold)
#         mask = (mass >= cmin) & (mass < cmax if upper_threshold else True)
#         xs.append(x[mask])
#         ys1.append(y1[mask])
#         ys2.append(y2[mask])
#         cs.append(numpy.log10(mass[mask]))
#
#     xs = numpy.concatenate(xs)
#     ys1 = numpy.concatenate(ys1)
#     ys2 = numpy.concatenate(ys2)
#     cs = numpy.concatenate(cs)
#
#     with plt.style.context(plt_utils.mplstyle):
#         plt.figure()
#         plt.hexbin(xs, ys1, C=cs, gridsize=50, mincnt=0,
#                    reduce_C_function=numpy.median)
#         mask = numpy.isfinite(xs) & numpy.isfinite(ys1)
#         corr = plt_utils.latex_float(*kendalltau(xs[mask], ys1[mask]))
#         plt.title(r"$\tau = {}, p = {}$".format(*corr), fontsize="small")
#
#         plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")
#         plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
#         plt.ylabel("1 - mean prob. of no match")
#
#         plt.tight_layout()
#         for ext in ["png"]:
#             nsim = str(nsim).zfill(5)
#             fout = join(plt_utils.fout,
#                         f"kl_vs_overlap_mean_{nsim}_{runs}.{ext}")
#             if upper_threshold:
#                 fout = fout.replace(f".{ext}", f"_upper_threshold.{ext}")
#             print(f"Saving to `{fout}`.")
#             plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
#         plt.close()
#
#     if not plot_std:
#         return
#
#     with plt.style.context(plt_utils.mplstyle):
#         plt.figure()
#         plt.hexbin(xs, ys2, C=cs, gridsize=50, mincnt=0,
#                    reduce_C_function=numpy.median)
#         plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")
#         plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
#         plt.ylabel(r"Ensemble std of summed overlap")
#         mask = numpy.isfinite(xs) & numpy.isfinite(ys2)
#         corr = plt_utils.latex_float(*kendalltau(xs[mask], ys2[mask]))
#         plt.title(r"$\tau = {}, p = {}$".format(*corr), fontsize="small")
#
#         plt.tight_layout()
#         for ext in ["png"]:
#             nsim = str(nsim).zfill(5)
#             fout = join(plt_utils.fout,
#                         f"kl_vs_overlap_std_{nsim}_{runs}.{ext}")
#             if upper_threshold:
#                 fout = fout.replace(f".{ext}", f"_upper_threshold.{ext}")
#             print(f"Saving to `{fout}`.")
#             plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
#         plt.close()


if __name__ == "__main__":
    min_logmass = 13.25
    smoothed = True
    nbins = 10
    ext = "png"
    plot_quijote = True
    min_maxoverlap = 0.

    funcs = [
        # "get_overlap_summary",
        # "get_expected_mass",
        # "get_mtot_vs_all_pairoverlap",
        # "get_mtot_vs_maxpairoverlap",
        # "get_mass_vs_separation",
        # "mass_vs_maxoverlap_separation",
        # "get_property_maxoverlap",
        ]
    for func in funcs:
        print(f"Cleaning up cache for `{func}`.")
        delete_disk_caches_for_function(func)

    if False:
        mtot_vs_all_pairoverlap(7444, "csiborg", min_logmass, smoothed,
                                nbins, ext=ext)
        if plot_quijote:
            mtot_vs_all_pairoverlap(0, "quijote",  min_logmass, smoothed,
                                    nbins, ext=ext)

    if False:
        mtot_vs_maxpairoverlap(7444, "csiborg", "fof_totpartmass", min_logmass,
                               smoothed, nbins, ext=ext)
        if plot_quijote:
            mtot_vs_maxpairoverlap(0, "quijote", "group_mass", min_logmass,
                                   smoothed, nbins, ext=ext)

    if False:
        mtot_vs_summedpairoverlap(7444, "csiborg", min_logmass, smoothed,
                                  nbins, ext)
        if plot_quijote:
            mtot_vs_summedpairoverlap(0, "quijote", min_logmass, smoothed,
                                      nbins, ext)
    if False:
        mtot_vs_maxoverlap_mass(7444, "csiborg", min_logmass, smoothed,
                                nbins, min_maxoverlap, ext)
        if plot_quijote:
            mtot_vs_maxoverlap_mass(0, "quijote", min_logmass, smoothed,
                                    nbins, min_maxoverlap, ext)

    if False:
        mtot_vs_expected_mass(7444, "csiborg", min_logmass, smoothed, nbins,
                              min_overlap=0, max_prob_nomatch=1, ext=ext)
        if plot_quijote:
            mtot_vs_expected_mass(0, "quijote", min_logmass, smoothed, nbins,
                                  min_overlap=0, max_prob_nomatch=1, ext=ext)

    if False:
        mass_vs_separation(7444, 7444 + 24, "csiborg", min_logmass, nbins,
                           smoothed, boxsize=677.7, plot_std=False)
        if plot_quijote:
            mass_vs_separation(0, 1, "quijote", min_logmass, nbins,
                               smoothed, boxsize=1000, plot_std=False)
    if False:
        mass_vs_maxoverlap_separation(7444, 7444 + 24, "csiborg", min_logmass,
                                      nbins, smoothed, boxsize=677.7,
                                      plot_std=False)
        if plot_quijote:
            mass_vs_maxoverlap_separation(
                0, 1, "quijote", min_logmass, nbins, smoothed, boxsize=1000,
                plot_std=False)

    if False:
        mtot_vs_mean_max_overlap(7444, "csiborg", min_logmass, smoothed, nbins)

        if plot_quijote:
            mtot_vs_mean_max_overlap(0, "quijote", min_logmass, smoothed,
                                     nbins)

    if False:
        key = "lambda200c"
        mtot_vs_maxoverlap_property(7444, "csiborg", min_logmass, key,
                                    min_maxoverlap, smoothed)
        if plot_quijote:
            mtot_vs_maxoverlap_property(0, "quijote", min_logmass, key,
                                        min_maxoverlap, smoothed)

    if False:
        matching_max_vs_overlap("csiborg", 7444, min_logmass)

        if plot_quijote:
            matching_max_vs_overlap("quijote", 0, min_logmass)

    if True:
        mtot_vs_maxpairoverlap_consistency(
            7444, "csiborg", "fof_totpartmass", min_logmass, smoothed,
            ext="png")
        # if plot_quijote:
        #     mtot_vs_maxpairoverlap_consistency(
        #         0, "quijote", "group_mass", min_logmass, smoothed,
        #         ext="png")


