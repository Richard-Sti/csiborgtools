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
from scipy.stats import kendalltau, binned_statistic, norm
from tqdm import tqdm, trange
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score


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
    nsimxs = csiborgtools.summary.get_cross_sims(
        simname, nsim0, paths, min_logmass, smoothed=smoothed)
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


# --------------------------------------------------------------------------- #
###############################################################################
#                   Total DM halo mass vs pair overlaps                       #
###############################################################################
# --------------------------------------------------------------------------- #


@cache_to_disk(120)
def get_mtot_vs_all_pairoverlap(nsim0, simname, mass_kind, min_logmass,
                                smoothed, nbins):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.summary.get_cross_sims(simname, nsim0, paths,
                                                 min_logmass,
                                                 smoothed=smoothed)
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
    sigma = 1

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        hb = plt.hexbin(x, y, mincnt=1, gridsize=50, bins="log")

        y_median, yerr = plt_utils.compute_error_bars(x, y, xbins, sigma=sigma)
        plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                     color='red', ls='dashed', capsize=3,
                     label="CSiBORG" if simname == "csiborg" else None)

        if simname == "csiborg":
            x_quijote, y_quijote, xbins_quijote = get_mtot_vs_all_pairoverlap(
                0, "quijote", "group_mass", min_logmass, smoothed, nbins)
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                x_quijote, y_quijote, xbins_quijote, sigma=sigma)
            plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                         y_median_quijote, yerr=yerr_quijote,
                         color='sandybrown', ls='dashed', capsize=3,
                         label="Quijote")
            plt.legend(loc="upper left", ncols=2, columnspacing=1.0)

        plt.colorbar(hb, label="Counts in bins", pad=0)
        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        plt.ylabel(r"$\mathcal{O}_{a b}$")
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
                               smoothed, nbins, concatenate=True):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.summary.get_cross_sims(
        simname, nsim0, paths, min_logmass, smoothed=smoothed)
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

    xbins = numpy.linspace(numpy.min(x), numpy.max(x), nbins)
    if concatenate:
        x = numpy.concatenate(x)
        y = numpy.concatenate(y)

    return x, y, xbins


def mtot_vs_maxpairoverlap(nsim0, simname, mass_kind, min_logmass, smoothed,
                           nbins, ext="png"):
    x, y, xbins = get_mtot_vs_maxpairoverlap(nsim0, simname, mass_kind,
                                             min_logmass, smoothed, nbins)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        plt.hexbin(x, y, mincnt=1, gridsize=50, bins="log")

        y_median, yerr = plt_utils.compute_error_bars(x, y, xbins, sigma=1)
        plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                     color='red', ls='dashed', capsize=3,
                     label="CSiBORG" if simname == "csiborg" else None)

        if simname == "csiborg":
            x_quijote, y_quijote, xbins_quijote = get_mtot_vs_all_pairoverlap(
                0, "quijote", "group_mass", min_logmass, smoothed, nbins)
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                x_quijote, y_quijote, xbins_quijote, sigma=1)
            plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                         y_median_quijote, yerr=yerr_quijote,
                         color='sandybrown', ls='dashed', capsize=3,
                         label="Quijote")

        plt.colorbar(label="Counts in bins", pad=0)
        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        plt.ylabel(r"$\max_{b \in \mathcal{B}} \mathcal{O}_{a b}$")
        plt.ylim(-0.02, 1.)
        plt.xlim(numpy.min(x) - 0.05)

        plt.tight_layout()
        fout = join(plt_utils.fout, f"mass_vs_max_pair_overlap{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def mtot_vs_maxpairoverlap_statistic(nsim0, simname, mass_kind, min_logmass,
                                     smoothed, nbins, ext="png"):
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

        # plt.colorbar(label="Counts in bins", pad=0)
        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        plt.ylabel(r"$\max_{b \in \mathcal{B}} \mathcal{O}_{a b}$")
        plt.ylim(-0.02, 1.)
        plt.xlim(numpy.min(x) - 0.05)

        plt.tight_layout()
        fout = join(plt_utils.fout, f"mass_vs_max_pair_overlap{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def mtot_vs_maxpairoverlap_fraction(min_logmass, smoothed, nbins, ext="png"):

    csiborg_nsims = [7444 + 24 * n for n in range(10)]
    quijote_nsims = [n for n in range(10)]

    @cache_to_disk(120)
    def get_xy_maxoverlap_fraction(n):
        x_csiborg, y_csiborg, __ = get_mtot_vs_maxpairoverlap(
            csiborg_nsims[n], "csiborg", MASS_KINDS["csiborg"], min_logmass,
            smoothed, nbins, concatenate=False)
        x_quijote, y_quijote, __ = get_mtot_vs_maxpairoverlap(
            quijote_nsims[n], "quijote", MASS_KINDS["quijote"], min_logmass,
            smoothed, nbins, concatenate=False)

        x_csiborg = x_csiborg[0]
        x_quijote = x_quijote[0]

        y_csiborg = numpy.asanyarray(y_csiborg)
        y_quijote = numpy.asanyarray(y_quijote)
        y_csiborg = numpy.median(y_csiborg, axis=0)
        y_quijote = numpy.median(y_quijote, axis=0)

        xbins = numpy.arange(min_logmass, 15.61, 0.2)
        x = 0.5 * (xbins[1:] + xbins[:-1])
        y = numpy.full((len(x), 3), numpy.nan)
        percentiles = norm.cdf(x=[1, 2, 3]) * 100

        for i in range(len(xbins)-1):
            mask_csiborg = (x_csiborg >= xbins[i]) & (x_csiborg < xbins[i+1])
            mask_quijote = (x_quijote >= xbins[i]) & (x_quijote < xbins[i+1])

            current_y_csiborg = y_csiborg[mask_csiborg]
            current_y_quijote = y_quijote[mask_quijote]
            current_tot_csiborg = len(current_y_csiborg)

            for j, q in enumerate(percentiles):
                threshold = numpy.percentile(current_y_quijote, q)
                y[i, j] = (current_y_csiborg > threshold).sum()
                y[i, j] /= current_tot_csiborg
        return x, y

    ys = [None] * 10
    for n in range(10):
        x, ys[n] = get_xy_maxoverlap_fraction(n)
    ys = numpy.asanyarray(ys)

    ymean = numpy.nanmean(ys, axis=0)
    ystd = numpy.nanstd(ys, axis=0)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        for i in range(3):
            plt.plot(x, ymean[:, i], label=r"${}\sigma$".format(i+1))
            plt.fill_between(x, ymean[:, i] - ystd[:, i],
                             ymean[:, i] + ystd[:, i], alpha=0.2)

        plt.legend()
        plt.ylim(0.0, 1.025)

        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        plt.ylabel(r"$f_{\rm significant}$")

        plt.tight_layout()
        fout = join(plt_utils.fout, f"mass_vs_max_pair_overlap_fraction.{ext}")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def summed_to_max_overlap(min_logmass, smoothed, nbins, ext="png"):
    x_csiborg = get_overlap_summary(7444, "csiborg", min_logmass, smoothed)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        x = numpy.mean(x_csiborg["summed_overlap"], axis=1)
        y = numpy.mean(x_csiborg["max_overlap"], axis=1)

        plt.hexbin(x, y, mincnt=0, gridsize=40,
                   C=numpy.log10(x_csiborg["mass0"]),
                   reduce_C_function=numpy.median)
        plt.colorbar(label=r"$\log M_{\rm tot} ~ [M_\odot / h]$", pad=0)

        plt.axline((0, 0), slope=1, color='red', linestyle='--',
                   label=r"$1-1$")

        plt.legend()

        plt.xlabel(r"$\langle \sum_{b \in \mathcal{B}} \mathcal{O}_{a b}\rangle_{\mathcal{B}}$") # noqa
        plt.ylabel(r"$\langle \max_{b \in \mathcal{B}} \mathcal{O}_{a b}\rangle_{\mathcal{B}}$") # noqa

        print(x.min(), x.max())
        print(y.min(), y.max())

        plt.tight_layout()
        ext = "pdf"
        fout = join(plt_utils.fout, f"summed_to_max_overlap.{ext}")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#                   Total DM halo mass vs pair overlaps                       #
###############################################################################
# --------------------------------------------------------------------------- #

@cache_to_disk(120)
def get_max_overlap_agreement(nsim0, simname, min_logmass, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.summary.get_cross_sims(
        simname, nsim0, paths, min_logmass, smoothed=smoothed)
    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    return csiborgtools.summary.max_overlap_agreements(cat0, catxs, 13.25,
                                                       155.5, paths)


def maximum_overlap_agreement(nsim0, simname, min_logmass, smoothed):

    agreements = get_max_overlap_agreement(nsim0, simname, min_logmass,
                                           smoothed)

    x, y, mass_bins = get_mtot_vs_maxpairoverlap(
        nsim0, simname, MASS_KINDS[simname], min_logmass, smoothed, 10,
        concatenate=False)
    x = x[0]
    y = numpy.asanyarray(y)
    mean_max_overlap = numpy.mean(y, axis=0)

    cat0 = open_cat(nsim0, simname)
    totpartmass = numpy.log10(cat0[MASS_KINDS[simname]])

    mask = totpartmass > min_logmass
    agreements = agreements[:, mask]
    totpartmass = totpartmass[mask]

    mask = numpy.any(numpy.isfinite(agreements), axis=0)
    y = numpy.sum(agreements == 1., axis=0)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure(figsize=(3.5, 2.625))

        plt.scatter(totpartmass[mask], y[mask], s=5, c=mean_max_overlap,
                    rasterized=True)
        plt.colorbar(label=r"$\langle \max_{b \in \mathcal{B}} \mathcal{O}_{a b}\rangle_{\mathcal{B}}$", pad=0) # noqa

        ymed, yerr = plt_utils.compute_error_bars(totpartmass[mask], y[mask],
                                                  mass_bins, sigma=1)
        plt.errorbar(0.5 * (mass_bins[1:] + mass_bins[:-1]), ymed, yerr=yerr,
                     capsize=3, c="red")

        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        plt.ylabel(r"$f_{\rm sym}$")
        plt.tight_layout()
        fout = join(plt_utils.fout,
                    f"maximum_overlap_agreement{simname}_{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()

    with plt.style.context(plt_utils.mplstyle):
        plt.figure(figsize=(3.5, 2.625))

        plt.scatter(mean_max_overlap[mask], y[mask], s=5, c=totpartmass,
                    rasterized=True)
        plt.colorbar(label=r"$\log M_{\rm tot} ~ [M_\odot / h]$", pad=0)
        bins = numpy.arange(0, 0.7, 0.05)
        ymed, yerr = plt_utils.compute_error_bars(
            mean_max_overlap[mask], y[mask], bins, sigma=1)

        plt.errorbar(0.5 * (bins[1:] + bins[:-1]), ymed, yerr=yerr, capsize=3,
                     c="red")
        plt.xlabel(r"$\langle \max_{b \in \mathcal{B}} \mathcal{O}_{a b}\rangle_{\mathcal{B}}$")  # noqa
        plt.ylabel(r"$f_{\rm sym}$")

        # plt.xscale("log")
        plt.tight_layout()
        fout = join(plt_utils.fout, f"maximum_overlap_agreement_mean_overlap{simname}_{nsim0}.{ext}")  # noqa
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

    xbins = numpy.linspace(numpy.nanmin(mass0), numpy.nanmax(mass0), nbins)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure(figsize=(3.5, 2.625))
        plt.hexbin(mass0, mean_overlap, mincnt=1, bins="log", gridsize=30)

        y_median, yerr = plt_utils.compute_error_bars(
            mass0, mean_overlap, xbins, sigma=1)
        plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                     color='red', ls='dashed', capsize=3, label="CSiBORG")

        if simname == "csiborg":
            x_quijote = get_overlap_summary(0, "quijote", min_logmass,
                                            smoothed)
            mass0_quijote = numpy.log10(x_quijote["mass0"])
            mean_overlap_quijote = numpy.nanmean(x_quijote["summed_overlap"],
                                                 axis=1)
            xbins_quijote = numpy.linspace(numpy.nanmin(mass0),
                                           numpy.nanmax(mass0), nbins)

            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                mass0_quijote, mean_overlap_quijote, xbins_quijote, sigma=1)
            plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                         y_median_quijote, yerr=yerr_quijote,
                         color='sandybrown', ls='dashed', capsize=3,
                         label="Quijote")
            plt.legend()

        plt.xlim(numpy.min(mass0))
        plt.xlim(numpy.min(mass0))
        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        plt.ylabel(r"$\langle \sum_{b \in \mathcal{B}} \mathcal{O}_{a b}\rangle_{\mathcal{B}}$") # noqa
        plt.colorbar(label="Counts in bins", pad=0)

        plt.tight_layout()
        fout = join(plt_utils.fout, f"prob_match_mean_{simname}_{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()

    with plt.style.context(plt_utils.mplstyle):
        plt.figure(figsize=(3.5, 2.625))
        plt.hexbin(mass0, std_overlap, mincnt=1, bins="log", gridsize=30)

        y_median, yerr = plt_utils.compute_error_bars(
            mass0, std_overlap, xbins, sigma=2)
        plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
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
                mass0_quijote, std_overlap_quijote, xbins_quijote, sigma=2)
            plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                         y_median_quijote, yerr=yerr_quijote,
                         color='sandybrown', ls='dashed', capsize=3)

        plt.colorbar(label=r"Counts in bins", pad=0)
        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        plt.ylabel(r"$\sigma\left(\sum_{b \in \mathcal{B}} \mathcal{O}_{a b}\right)_{\mathcal{B}}$")  # noqa

        plt.tight_layout()
        fout = join(plt_utils.fout, f"prob_match_std_{simname}_{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
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
    mu, __ = csiborgtools.summary.weighted_stats(dist, overlap, min_weight=0)

    mask = numpy.isfinite(mass) & numpy.isfinite(mu)

    return mass[mask], mu[mask]


def mass_vs_separation(nsim0, nsimx, simname, min_logmass, nbins, smoothed,
                       boxsize):
    mass, dist = get_mass_vs_separation(nsim0, nsimx, simname, min_logmass,
                                        boxsize, smoothed)
    dist = numpy.log10(dist)
    xbins = numpy.linspace(numpy.nanmin(mass), numpy.nanmax(mass), nbins)

    with plt.style.context(plt_utils.mplstyle):
        fig, ax = plt.subplots()

        cx = ax.hexbin(mass, dist, mincnt=0, bins="log", gridsize=50)
        y_median, yerr = plt_utils.compute_error_bars(mass, dist, xbins,
                                                      sigma=1)
        ax.errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                    color='red', ls='dashed', capsize=3,
                    label="CSiBORG" if simname == "csiborg" else None)

        if simname == "csiborg":
            mass_quijote, dist_quijote = get_mass_vs_separation(
                0, 1, "quijote", min_logmass, boxsize, smoothed)
            dist_quijote = numpy.log10(dist_quijote)
            xbins_quijote = numpy.linspace(numpy.nanmin(mass_quijote),
                                           numpy.nanmax(mass_quijote), nbins)
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                mass_quijote, dist_quijote, xbins_quijote, sigma=1)
            ax.errorbar(0.5 * (xbins_quijote[1:] + xbins_quijote[:-1]),
                        y_median_quijote, yerr=yerr_quijote,
                        color='sandybrown', ls='dashed', capsize=3,
                        label="Quijote")
            ax.legend()

        fig.colorbar(cx, label="Bin counts", pad=0)
        ax.set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        ax.set_ylabel(r"$\log \langle \Delta R / R_{\rm 200c}\rangle$")

        fig.tight_layout()
        fout = join(plt_utils.fout,
                    f"mass_vs_sep_{simname}_{nsim0}_{nsimx}.{ext}")
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#                Total DM halo mass vs expected matched mass                  #
###############################################################################
# --------------------------------------------------------------------------- #


@cache_to_disk(120)
def get_expected_mass(nsim0, simname, min_logmass, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.summary.get_cross_sims(simname, nsim0, paths,
                                                 min_logmass, smoothed=True)

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.summary.NPairsOverlap(cat0, catxs, paths,
                                                min_logmass)
    mass0 = reader.cat0(MASS_KINDS[simname])
    mask = mass0 > 10**min_logmass

    mean_expected, std_expected = reader.expected_property(
        MASS_KINDS[simname], smoothed, min_logmass)

    return {"mass0": mass0[mask],
            "mu": mean_expected[mask],
            "std": std_expected[mask],
            "prob_match": reader.summed_overlap(smoothed)[mask],
            }


def mtot_vs_expected_mass(nsim0, simname, min_logmass, smoothed, ext="png"):
    x = get_expected_mass(nsim0, simname, min_logmass, smoothed)

    mass = x["mass0"]
    mu = x["mu"]
    std = x["std"]
    prob_match = x["prob_match"]

    mass = numpy.log10(mass)
    prob_match = numpy.nanmean(prob_match, axis=1)
    mask = numpy.isfinite(mass) & numpy.isfinite(mu) & numpy.isfinite(std)

    with plt.style.context(plt_utils.mplstyle):
        fig, axs = plt.subplots(ncols=3, figsize=(3.5 * 2, 2.625))

        im0 = axs[0].hexbin(mass[mask], mu[mask], mincnt=1, bins="log",
                            gridsize=30,)
        im1 = axs[1].hexbin(mass[mask], std[mask], mincnt=1, bins="log",
                            gridsize=30)
        im2 = axs[2].hexbin(prob_match[mask], mu[mask] - mass[mask],
                            gridsize=30, C=mass[mask],
                            reduce_C_function=numpy.nanmedian)

        axs[2].axhline(0, color="red", linestyle="--", alpha=0.5)
        axs[0].set_xlabel(r"$\log M_{\rm tot, ref} ~ [M_\odot / h]$")
        axs[0].set_ylabel(r"$\log M_{\rm tot, exp} ~ [M_\odot / h]$")
        axs[1].set_xlabel(r"$\log M_{\rm tot, ref} ~ [M_\odot / h]$")
        axs[1].set_ylabel(r"$\sigma_{\log M_{\rm tot, exp}}$")
        axs[2].set_xlabel(r"$\langle \sum_{b \in \mathcal{B}} \mathcal{O}_{a b}\rangle_{\mathcal{B}}$") # noqa
        axs[2].set_ylabel(r"$\log (M_{\rm tot, exp} / M_{\rm tot, ref})$")

        z = numpy.nanmean(mass[mask])
        axs[0].axline((z, z), slope=1, color='red', linestyle='--',
                      label=r"$1-1$")
        axs[0].legend()

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
        fout = join(plt_utils.fout, f"mass_vs_expmass_{nsim0}_{simname}.{ext}")
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#               Total DM halo mass vs maximum overlap halo property           #
###############################################################################
# --------------------------------------------------------------------------- #

@cache_to_disk(120)
def get_expected_key(nsim0, simname, min_logmass, key, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.summary.get_cross_sims(simname, nsim0, paths,
                                                 min_logmass, smoothed=True)

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.summary.NPairsOverlap(cat0, catxs, paths,
                                                min_logmass)
    mass0 = reader.cat0(MASS_KINDS[simname])
    mask = mass0 > 10**min_logmass

    mean_expected, std_expected = reader.expected_property(key, smoothed,
                                                           min_logmass)

    log_mass0 = numpy.log10(mass0)
    control = numpy.full(len(log_mass0), numpy.nan)
    for i in trange(len(log_mass0), desc="Control"):
        if not mask[i]:
            continue

        control_ = [None] * len(catxs)
        for j in range(len(catxs)):
            log_massx = numpy.log10(reader[j].catx(MASS_KINDS[simname]))
            ks = numpy.argsort(numpy.abs(log_massx - log_mass0[i]))[:15]
            control_[j] = reader[j].catx(key)[ks]

        control[i] = numpy.nanmean(numpy.concatenate(control_))

    return {"mass0": mass0[mask],
            "prop0": reader.cat0(key)[mask],
            "mu": mean_expected[mask],
            "std": std_expected[mask],
            "control": control[mask],
            "prob_match": reader.summed_overlap(smoothed)[mask],
            }


def mtot_vs_expected_key(nsim0, simname, min_logmass, key, smoothed,
                         min_logmass_run=None):
    mass_kind = MASS_KINDS[simname]
    assert key != mass_kind

    x = get_expected_key(nsim0, simname, min_logmass, key, smoothed)
    mass0 = numpy.log10(x["mass0"])
    prop0 = numpy.log10(x["prop0"])
    mu = x["mu"]
    std = x["std"]
    prob_match = numpy.nanmean(x["prob_match"], axis=1)
    control = numpy.log10(x["control"])

    mask = numpy.isfinite(prop0) & numpy.isfinite(mu) & numpy.isfinite(std)
    if min_logmass_run is not None:
        mask &= mass0 > min_logmass_run
    mass0 = mass0[mask]
    prop0 = prop0[mask]
    mu = mu[mask]
    std = std[mask]
    control = control[mask]
    prob_match = prob_match[mask]

    def rmse(x, y, sample_weight=None):
        return numpy.sqrt(numpy.average((x - y)**2, weights=sample_weight))

    print("Unweigted R2         = ", r2_score(prop0, mu))
    print("Err Weighted R2      = ", r2_score(prop0, mu, sample_weight=1 / std**2))  # noqa
    print("Pmatch R2            = ", r2_score(prop0, mu, sample_weight=prob_match))  # noqa
    print("Control R2           = ", r2_score(prop0, control))

    print()

    print("Unweigted RMSE       = ", rmse(prop0, mu))
    print("Err Weighted RMSE    = ", rmse(prop0, mu, 1 / std**2))
    print("Pmatch RMSE          = ", rmse(prop0, mu, prob_match))
    print("Control RMSE         = ", rmse(prop0, control))

    with plt.style.context(plt_utils.mplstyle):
        fig, ax = plt.subplots(figsize=(3.5, 2.625))

        ax.errorbar(prop0, mu, yerr=std, capsize=3, ls="none")

        z = numpy.nanmean(prop0)
        ax.axline((z, z), slope=1, color='red', linestyle='--')

        if key == "lambda200c":
            ax.axhline(numpy.median(control), color="red", ls="--", zorder=0)

        if key == "lambda200c":
            ax.set_xlabel(r"$\log \lambda_{\rm 200c, ref}$")
            ax.set_ylabel(r"$\log \lambda_{\rm 200c, exp}$")
        elif key == "conc":
            ax.set_xlabel(r"$\log c_{\rm 200c, ref}$")
            ax.set_ylabel(r"$\log c_{\rm 200c, exp}$")

        fig.tight_layout()
        fout = join(plt_utils.fout, f"max_{key}_{simname}_{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#                  Total mass of a single halo expectation                    #
###############################################################################
# --------------------------------------------------------------------------- #


@cache_to_disk(120)
def get_expected_single(k, nsim0, simname, min_logmass, key, smoothed, in_log):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.summary.get_cross_sims(
        simname, nsim0, paths, min_logmass, smoothed=smoothed)

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.summary.NPairsOverlap(cat0, catxs, paths,
                                                min_logmass)

    x0 = reader.cat0(key)

    if k == "maxmass":
        k = numpy.nanargmax(reader.cat0(MASS_KINDS[simname]))

    if k == "max":
        k = numpy.nanargmax(x0)

    xcross, overlaps = reader.expected_property_single(k, key, smoothed,
                                                       in_log)

    control = [None] * len(catxs)
    log_mass0 = numpy.log10(reader.cat0(MASS_KINDS[simname])[k])
    for j in range(len(catxs)):
        log_massx = numpy.log10(reader[j].catx(MASS_KINDS[simname]))
        ks = numpy.argsort(numpy.abs(log_massx - log_mass0))[:15]
        control[j] = reader[j].catx(key)[ks]

    if in_log:
        x0 = numpy.log10(x0)

    return x0[k], xcross, overlaps, control


def mtot_vs_expected_single(k, nsim0, simname, min_logmass, key, smoothed,
                            in_log):
    x0, xcross, overlaps, control = get_expected_single(
        k, nsim0, simname, min_logmass, key, smoothed, in_log)
    xcross = numpy.concatenate(xcross)
    overlaps = numpy.concatenate(overlaps)
    control = numpy.concatenate(control)

    with plt.style.context("science"):
        plt.figure()
        plt.hist(xcross, bins=40, histtype="step", density=1,
                 label="Unweighted")
        plt.hist(xcross, weights=overlaps, bins=40, density=1, histtype="step",
                 label="Weighted")
        m = numpy.isfinite(xcross) & numpy.isfinite(overlaps)
        peak = csiborgtools.summary.find_peak(xcross[m], overlaps[m])
        plt.axvline(peak, color="forestgreen", ls="--")
        plt.axvline(x0, color="red", ls="--")

        if key != "totpartmass":
            plt.hist(control, bins=40, histtype="step", density=1,
                     label="Control")

        if key == "totpartmass" or key == "fof_totpartmass":
            plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        elif key == "lambda200c":
            plt.xlabel(r"$\log \lambda_{\rm 200c}$")
        elif key == "conc":
            plt.xlabel(r"$\log c$")

        plt.ylabel("Normalized counts")
        plt.legend()

        plt.tight_layout()
        fout = join(
            plt_utils.fout,
            f"expected_single_{k}_{key}_{nsim0}_{simname}_{min_logmass}.pdf")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
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


def matching_max_vs_overlap(min_logmass):
    left_edges = numpy.arange(min_logmass, 15, 0.1)

    with plt.style.context("science"):
        # fig, axs = plt.subplots(ncols=2, figsize=(2 * 3.5, 2.625))
        fig, axs = plt.subplots(ncols=1, figsize=(3.5, 2.625))
        axs = [axs]
        ax2 = axs[0].twinx()
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for n, mult in enumerate([2.5, 5., 7.5]):

            def make_y1_y2(simname):
                nsims = 100 if simname == "csiborg" else 9
                nsim0 = 7444 if simname == "csiborg" else 0
                x = get_matching_max_vs_overlap(simname,
                                                nsim0, min_logmass, mult=mult)

                mask = numpy.all(numpy.isfinite(x["max_overlap"]), axis=1)
                x["success"][~mask, :] = False

                mass0 = numpy.log10(x["mass0"])
                max_overlap = x["max_overlap"]
                match_overlap = x["match_overlap"]
                success = x["success"]

                nbins = len(left_edges)
                y = numpy.full((nbins, nsims), numpy.nan)
                y2 = numpy.full(nbins, numpy.nan)
                for i in range(nbins):
                    m = mass0 > left_edges[i]
                    for j in range(nsims):
                        y[i, j] = numpy.sum((max_overlap[m, j] == match_overlap[m, j]) & success[m, j])  # noqa
                        y[i, j] /= numpy.sum(success[m, j])

                    y2[i] = success[m, 0].mean()
                return y, y2

            offset = numpy.random.normal(0, 0.015)

            y1_csiborg, y2_csiborg = make_y1_y2("csiborg")

            ysummary = numpy.percentile(y1_csiborg, [16, 50, 84], axis=1)
            axs[0].plot(left_edges + offset, ysummary[1], c=cols[n],
                        label=r"${}~R_{{\rm 200c}}$".format(mult))
            ax2.plot(left_edges + offset, y2_csiborg, c=cols[n], ls="dotted",
                     zorder=0)

        axs[0].legend(ncols=1, loc="upper left")
        for i in range(1):
            axs[i].set_xlabel(r"$\log M_{\rm tot, min} ~ [M_\odot / h]$")

        axs[0].set_ylabel(r"$f_{\rm agreement}$")
        ax2.set_ylabel(r"$f_{\rm match}$")

        fig.tight_layout()
        fout = join(plt_utils.fout,
                    f"matching_max_agreement_{min_logmass}.pdf")
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    min_logmass = 13.25
    smoothed = True
    nbins = 10
    ext = "pdf"
    plot_quijote = False
    min_maxoverlap = 0.

    funcs = [
        # "get_overlap_summary",
        # "get_max_overlap_agreement",
        # "get_mtot_vs_all_pairoverlap",
        # "get_mtot_vs_maxpairoverlap",
        # "get_mass_vs_separation",
        # "get_expected_mass",
        # "get_expected_key",
        # "get_expected_single",
        ]
    for func in funcs:
        print(f"Cleaning up cache for `{func}`.")
        delete_disk_caches_for_function(func)

    if False:
        mtot_vs_all_pairoverlap(7444, "csiborg", min_logmass, smoothed,
                                nbins, ext=ext)
        mtot_vs_maxpairoverlap(7444, "csiborg", "fof_totpartmass", min_logmass,
                               smoothed, nbins, ext=ext)
        if plot_quijote:
            mtot_vs_all_pairoverlap(0, "quijote",  min_logmass, smoothed,
                                    nbins, ext=ext)
            mtot_vs_maxpairoverlap(0, "quijote", "group_mass", min_logmass,
                                   smoothed, nbins, ext=ext)

    if False:
        mtot_vs_maxpairoverlap_fraction(min_logmass, smoothed, nbins, ext=ext)

    if False:
        maximum_overlap_agreement(7444, "csiborg", min_logmass, smoothed)

    if False:
        summed_to_max_overlap(min_logmass, smoothed, nbins, ext=ext)

    if True:
        mtot_vs_summedpairoverlap(7444, "csiborg", min_logmass, smoothed,
                                  nbins, ext)
        if plot_quijote:
            mtot_vs_summedpairoverlap(0, "quijote", min_logmass, smoothed,
                                      nbins, ext)

    if False:
        mtot_vs_expected_mass(7444, "csiborg", min_logmass, smoothed, ext=ext)
        # if plot_quijote:
        #     mtot_vs_expected_mass(0, "quijote", min_logmass, smoothed, nbins,
        #                           max_prob_nomatch=1, ext=ext)

    if False:
        key = "conc"
        mtot_vs_expected_key(7444, "csiborg", min_logmass, key, smoothed, 14.)
        # if plot_quijote:
        #     mtot_vs_expected_key(0, "quijote", min_logmass, key, smoothed,
        #                          nbins)

    if False:
        mass_vs_separation(7444, 7444 + 24 * 60, "csiborg", min_logmass, nbins,
                           smoothed, boxsize=677.7)
        # if plot_quijote:
        #     mass_vs_separation(0, 1, "quijote", min_logmass, nbins,
        #                        smoothed, boxsize=1000, plot_std=False)

    if False:
        matching_max_vs_overlap(min_logmass)

    if False:
        # mtot_vs_expected_single("max", 7444, "csiborg", min_logmass,
        #                         "totpartmass", True, True)
        mtot_vs_expected_single("maxmass", 7444, "csiborg", min_logmass,
                                "lambda200c", True, True)
        # if plot_quijote:
        #     mtot_vs_expected_single("max", 0, "quijote", min_logmass,
        #                             "totpartmass", True, True)

