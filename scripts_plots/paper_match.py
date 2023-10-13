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
from scipy.stats import norm
from sklearn.metrics import r2_score
from tqdm import tqdm, trange
from astropy import units as u
from astropy.coordinates import SkyCoord

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
                     label="CSiBORG" if simname == "csiborg" else None,
                     fmt="o", ms=3)

        if simname == "csiborg":
            x_quijote, y_quijote, xbins_quijote = get_mtot_vs_all_pairoverlap(
                0, "quijote", "group_mass", min_logmass, smoothed, nbins)
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                x_quijote, y_quijote, xbins_quijote, sigma=sigma)
            plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                         y_median_quijote, yerr=yerr_quijote,
                         color='sandybrown', ls='dashed', capsize=3,
                         label="Quijote", fmt="o", ms=3)
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
                     label="CSiBORG" if simname == "csiborg" else None,
                     fmt="o", ms=3)

        if simname == "csiborg":
            x_quijote, y_quijote, xbins_quijote = get_mtot_vs_all_pairoverlap(
                0, "quijote", "group_mass", min_logmass, smoothed, nbins)
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                x_quijote, y_quijote, xbins_quijote, sigma=1)
            plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                         y_median_quijote, yerr=yerr_quijote,
                         color='sandybrown', ls='dashed', capsize=3,
                         label="Quijote", fmt="o", ms=3)

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
                     label="CSiBORG" if simname == "csiborg" else None,
                     fmt="o", ms=3)

        if simname == "csiborg":
            x_quijote, y_quijote, xbins_quijote = get_mtot_vs_all_pairoverlap(
                0, "quijote", "group_mass", min_logmass, smoothed, nbins)
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                x_quijote, y_quijote, xbins_quijote, sigma=2)
            plt.errorbar(0.5 * (xbins[1:] + xbins[:-1]) + 0.01,
                         y_median_quijote, yerr=yerr_quijote,
                         color='blue', ls='dashed', capsize=3,
                         label="Quijote", fmt="o", ms=3)
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

        plt.xlim(numpy.nanmin(x), numpy.nanmax(x))

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
                     capsize=3, c="red", ls="--", lw=0.5, fmt="o", ms=3)

        plt.xlim(numpy.nanmin(totpartmass[mask]))
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

        plt.errorbar(0.5 * (bins[1:] + bins[:-1]), ymed, yerr=yerr,
                     capsize=3, c="red", ls="--", lw=0.5, fmt="o", ms=3)
        plt.xlabel(r"$\langle \max_{b \in \mathcal{B}} \mathcal{O}_{a b}\rangle_{\mathcal{B}}$")  # noqa
        plt.ylabel(r"$f_{\rm sym}$")
        plt.xlim(numpy.nanmin(mean_max_overlap[mask]))

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
                     color='red', ls='dashed', capsize=3, label="CSiBORG",
                     fmt="o", ms=3)

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
                         label="Quijote", fmt="o", ms=3)
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
                     color='red', ls='dashed', capsize=3, fmt="o", ms=3)

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
                         color='sandybrown', ls='dashed', capsize=3,
                         fmt="o", ms=3)

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
def get_mass_vs_separation(nsim0, simname, min_logmass, boxsize,
                           smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.summary.get_cross_sims(
        simname, nsim0, paths, min_logmass, smoothed=smoothed)
    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.summary.NPairsOverlap(
        cat0, catxs, paths, min_logmass)

    mass = numpy.log10(reader.cat0(MASS_KINDS[simname]))
    mus = numpy.zeros((len(catxs), len(mass)))
    for i in trange(len(catxs), desc="Stacking catalogues"):
        dist = reader[i].dist(in_initial=False, boxsize=boxsize,
                              norm_kind=None)
        overlap = reader[i].overlap(smoothed)
        mus[i], __ = csiborgtools.summary.weighted_stats(dist, overlap)

    return mass, mus


def mass_vs_separation(nsim0, simname, min_logmass, nbins, smoothed,
                       boxsize):
    mass, dist = get_mass_vs_separation(
        nsim0, simname, min_logmass, boxsize, smoothed)

    dist = numpy.nanmean(dist, axis=0)
    mask = numpy.isfinite(mass) & numpy.isfinite(dist)

    mass = mass[mask]
    dist = dist[mask]

    xbins = numpy.linspace(numpy.nanmin(mass), numpy.nanmax(mass), nbins)

    with plt.style.context(plt_utils.mplstyle):
        fig, ax = plt.subplots()
        plt.rcParams["axes.grid"] = False

        cx = ax.hexbin(mass, dist, mincnt=0, bins="log", gridsize=50)
        y_median, yerr = plt_utils.compute_error_bars(mass, dist, xbins,
                                                      sigma=1)
        ax.errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                    color='red', ls='dashed', capsize=3,
                    label="CSiBORG" if simname == "csiborg" else None,
                    fmt="o", ms=3)
        ax.set_xlim(numpy.nanmin(mass))

        if simname == "csiborg":
            mass_quijote, dist_quijote = get_mass_vs_separation(
                0, "quijote", min_logmass, boxsize, smoothed)
            dist_quijote = numpy.nanmean(dist_quijote, axis=0)
            mask = numpy.isfinite(mass_quijote) & numpy.isfinite(dist_quijote)
            mass_quijote = mass_quijote[mask]
            dist_quijote = dist_quijote[mask]

            # dist_quijote = numpy.log10(dist_quijote)
            xbins_quijote = numpy.linspace(numpy.nanmin(mass_quijote),
                                           numpy.nanmax(mass_quijote), nbins)
            xbins_quijote = xbins_quijote[:-1]
            y_median_quijote, yerr_quijote = plt_utils.compute_error_bars(
                mass_quijote, dist_quijote, xbins_quijote, sigma=1)
            ax.errorbar(0.5 * (xbins_quijote[1:] + xbins_quijote[:-1]),
                        y_median_quijote, yerr=yerr_quijote,
                        color='sandybrown', ls='dashed', capsize=3,
                        label="Quijote", fmt="o", ms=3)
            ax.legend(ncols=2)

        fig.colorbar(cx, label="Bin counts", pad=0)
        ax.set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        ax.set_ylabel(r"$\langle \Delta R / R_{\rm 200c}\rangle_{\mathcal{B}}$")  # noqa
        ax.set_ylabel(r"$\langle \Delta R \rangle_{\mathcal{B}} ~ [\mathrm{Mpc} / h]$")  # noqa

        fig.tight_layout()
        fout = join(plt_utils.fout, f"mass_vs_sep_{simname}_{nsim0}.{ext}")
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

    in_log = False if key == "conc" else True
    mean_expected, std_expected = reader.expected_property(
        key, smoothed, min_logmass, in_log=in_log)

    log_mass0 = numpy.log10(mass0)
    control = numpy.full(len(log_mass0), numpy.nan)
    # for i in trange(len(log_mass0), desc="Control"):
    #     if not mask[i]:
    #         continue

    #     control_ = [None] * len(catxs)
    #     for j in range(len(catxs)):
    #         log_massx = numpy.log10(reader[j].catx(MASS_KINDS[simname]))
    #         ks = numpy.argsort(numpy.abs(log_massx - log_mass0[i]))[:15]
    #         control_[j] = reader[j].catx(key)[ks]

    #     control[i] = numpy.nanmean(numpy.concatenate(control_))

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
    prop0 = x["prop0"]
    mu = x["mu"]
    std = x["std"]
    prob_match = numpy.nanmean(x["prob_match"], axis=1)
    control = x["control"]

    print("prop0 ", prop0)
    print("mu    ", mu)
    print("std   ", std)
    print("control", control)

    mask = numpy.isfinite(prop0) & numpy.isfinite(mu) & numpy.isfinite(std)
    if min_logmass_run is not None:
        mask &= mass0 > 15
    mask &= prop0 > 0.4

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

    print()

    print("Unweigted RMSE       = ", rmse(prop0, mu))
    print("Err Weighted RMSE    = ", rmse(prop0, mu, 1 / std**2))
    print("Pmatch RMSE          = ", rmse(prop0, mu, prob_match))

    with plt.style.context(plt_utils.mplstyle):
        fig, ax = plt.subplots(figsize=(3.5, 2.625))

        ax.errorbar(prop0, mu, yerr=std, capsize=3, ls="none", marker="o")

        z = numpy.nanmean(prop0)
        ax.axline((z, z), slope=1, color='red', linestyle='--', label=r"$1-1$")

        ax.legend()

        # if key == "lambda200c":
        #     ax.axhline(numpy.median(control), color="red", ls="--", zorder=0)

        if key == "lambda200c":
            ax.set_xlabel(r"$\log \lambda_{\rm 200c, ref}$")
            ax.set_ylabel(r"$\log \lambda_{\rm 200c, exp}$")
        elif key == "conc":
            ax.set_xlabel(r"$c_{\rm 200c, ref}$")
            ax.set_ylabel(r"$c_{\rm 200c, exp}$")

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
def get_expected_single(k, nsim0, simname, min_logmass, key, smoothed,
                        in_log):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.summary.get_cross_sims(
        simname, nsim0, paths, min_logmass, smoothed=smoothed)

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.summary.NPairsOverlap(cat0, catxs, paths,
                                                min_logmass)

    x0 = reader.cat0(key)

    if "maxmass" in k:
        k0 = int(k.split("__")[1])
        k = numpy.argsort(reader.cat0(MASS_KINDS[simname]))[::-1][k0]
        print(f"Doing the {k0}th maximum hass halo with index {k}.")

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

    xcross = numpy.asanyarray(xcross)
    overlaps = numpy.asanyarray(overlaps)

    return x0[k], xcross, overlaps, control


def mtot_vs_expected_single(k, nsim0, simname, min_logmass, key, smoothed):
    x0, xcross, overlaps, control = get_expected_single(
        k, nsim0, simname, min_logmass, key, smoothed, False)

    control = numpy.concatenate(control)

    if key == "lambda200c" or key == "totpartmass":
        xcross = numpy.log10(xcross)
        control = numpy.log10(control)
        x0 = numpy.log10(x0)

    m = numpy.isfinite(xcross) & numpy.isfinite(overlaps)
    with plt.style.context("science"):
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plt.figure()
        plt.hist(xcross[m], weights=overlaps[m], bins=30, histtype="step",
                 density=1, label="Matched", color=cols[0])

        peak = csiborgtools.summary.find_peak(xcross[m], overlaps[m])
        plt.axvline(peak, color="mediumblue", ls="--")
        plt.axvline(x0, color="red", ls="--")

        if key != "totpartmass":
            m = numpy.isfinite(control)
            plt.hist(control[m], bins=30, histtype="step", density=1,
                     label="Control", color=cols[1])

            m = numpy.isfinite(control)
            xmin, xmax = numpy.percentile(control[m], [16, 84])
            plt.axvspan(xmin, xmax, color=cols[1], alpha=0.1)

        if key == "totpartmass" or key == "fof_totpartmass":
            plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        elif key == "lambda200c":
            plt.xlabel(r"$\log \lambda_{\rm 200c}$")
        elif key == "conc":
            plt.xlabel(r"$c$")

        plt.ylabel("Normalized counts")
        plt.legend()

        plt.tight_layout()
        fout = join(
            plt_utils.fout,
            f"expected_single_{k}_{key}_{nsim0}_{simname}_{min_logmass}.pdf")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def mtot_vs_expected_single_panel(k, nsim0, simname, min_logmass, smoothed):
    print("Plotting..")

    def local_data(key):
        x0, xcross, overlaps, control = get_expected_single(
            k, nsim0, simname, min_logmass, key, smoothed, False)

        control = numpy.concatenate(control)
        if key == "lambda200c" or key == "totpartmass":
            xcross = numpy.log10(xcross)
            control = numpy.log10(control)
            x0 = numpy.log10(x0)

        m = numpy.isfinite(xcross) & numpy.isfinite(overlaps)
        return x0, xcross[m], overlaps[m], control

    with plt.style.context("science"):
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, axs = plt.subplots(ncols=3, figsize=(2 * 3.5, 2.625))
        fig.subplots_adjust(wspace=0.0)
        nbins = 25

        # TOTAL MASS
        x0, x, weights, __ = local_data("totpartmass")
        axs[0].hist(x, weights=weights, bins=nbins, histtype="step", density=1,
                    label="Matched")
        axs[0].legend()
        peak = csiborgtools.summary.find_peak(x, weights)
        axs[0].axvline(peak, color="mediumblue", ls="--")
        axs[0].axvline(x0, color="red", ls="--")
        std = numpy.average((x - peak)**2, weights=weights)**0.5
        xmin, xmax = peak - std, peak + std
        axs[0].axvspan(xmin, xmax, color=cols[0], alpha=0.2)

        # SPIN
        x0, x, weights, control = local_data("lambda200c")
        axs[1].hist(x, weights=weights, bins=nbins, histtype="step", density=1,
                    color=cols[0])
        axs[1].hist(control, bins=nbins, histtype="step", density=1,
                    color=cols[1], label="Control")
        axs[1].legend()
        xmin, xmax = numpy.percentile(control, [16, 84])
        axs[1].axvspan(xmin, xmax, color=cols[1], alpha=0.1)
        peak = csiborgtools.summary.find_peak(x, weights)
        axs[1].axvline(peak, color="mediumblue", ls="--")
        axs[1].axvline(x0, color="red", ls="--")
        axs[1].set_yticklabels([])
        m = numpy.isfinite(control)
        xmin, mu, xmax = numpy.percentile(control[m], [16, 50, 84])
        axs[1].axvspan(xmin, xmax, color=cols[1], alpha=0.2)
        axs[1].axvline(mu, color=cols[1], ls="--")
        std = numpy.average((x - peak)**2, weights=weights)**0.5
        xmin, xmax = peak - std, peak + std
        axs[1].axvspan(xmin, xmax, color=cols[0], alpha=0.2)

        # CONCENTRATION
        x0, x, weights, control = local_data("conc")
        axs[2].hist(x, weights=weights, bins=nbins, histtype="step", density=1,
                    color=cols[0])
        axs[2].hist(control, bins=nbins, histtype="step", density=1,
                    color=cols[1])
        xmin, xmax = numpy.percentile(control, [16, 84])
        axs[2].axvspan(xmin, xmax, color=cols[1], alpha=0.1)
        peak = csiborgtools.summary.find_peak(x, weights)
        axs[2].axvline(peak, color="mediumblue", ls="--")
        axs[2].axvline(x0, color="red", ls="--")
        axs[2].set_yticklabels([])
        m = numpy.isfinite(control)
        xmin, mu, xmax = numpy.percentile(control[m], [16, 50, 84])
        axs[2].axvspan(xmin, xmax, color=cols[1], alpha=0.2)
        axs[2].axvline(mu, color=cols[1], ls="--")
        std = numpy.average((x - peak)**2, weights=weights)**0.5
        xmin, xmax = peak - std, peak + std
        axs[2].axvspan(xmin, xmax, color=cols[0], alpha=0.2)

        axs[0].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[1].set_xlabel(r"$\log \lambda_{\rm 200c}$")
        axs[2].set_xlabel(r"$c$")

        axs[0].set_ylabel("Normalized counts")

        fig.tight_layout(h_pad=0, w_pad=0)
        fout = join(
            plt_utils.fout,
            f"expected_single_{k}_panel_{nsim0}_{simname}_{min_logmass}.pdf")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#                        Concentration reconstruction                         #
###############################################################################
# --------------------------------------------------------------------------- #


@cache_to_disk(120)
def get_expected_many_topmass(kmax, nsim0, simname, min_logmass, key, smoothed,
                              in_log):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.summary.get_cross_sims(
        simname, nsim0, paths, min_logmass, smoothed=smoothed)

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.summary.NPairsOverlap(cat0, catxs, paths,
                                                min_logmass)

    log_mass0 = numpy.log10(reader.cat0(MASS_KINDS[simname]))
    x0 = reader.cat0(key)

    out_mass = numpy.full((kmax), numpy.nan)
    out_true = numpy.full((kmax), numpy.nan)
    out_matched = numpy.full((kmax, len(nsimxs), 2), numpy.nan)
    out_control = numpy.full((kmax, len(nsimxs)), numpy.nan)

    for k0 in trange(kmax, desc="Looping over top masses"):
        k = numpy.argsort(log_mass0)[::-1][k0]
        xcross, overlaps = reader.expected_property_single(k, key, smoothed,
                                                           in_log)
        out_matched[k0, :, 0] = xcross
        out_matched[k0, :, 1] = overlaps

        out_mass[k0] = log_mass0[k]
        out_true[k0] = x0[k]

        for j in range(len(catxs)):
            log_massx = numpy.log10(reader[j].catx(MASS_KINDS[simname]))
            ks = numpy.argsort(numpy.abs(log_massx - log_mass0[k]))[:15]
            out_control[k0, j] = numpy.nanmean(reader[j].catx(key)[ks])

    return out_mass, out_true, out_matched, out_control


def expected_in_tolerance(kmax, nsim0, simname, min_logmass, key, smoothed):
    out_mass, out_true, out_matched, out_control = get_expected_many_topmass(
        kmax, nsim0, simname, min_logmass, key, smoothed, False)

    if key == "lambda200c" or key == "totpartmass":
        out_true = numpy.log10(out_true)
        out_matched[..., 0] = numpy.log10(out_matched[..., 0])
        out_control = numpy.log10(out_control)

    tolerances = [0.1, 0.3, 0.5]

    in_tolerance = numpy.full((kmax, len(tolerances)), numpy.nan)
    for k in range(kmax):
        frac_diff = (out_matched[k, :, 0] - out_true[k]) / out_true[k]
        frac_diff = numpy.abs(frac_diff)
        for j, tol in enumerate(tolerances):
            in_tolerance[k, j] = numpy.sum(frac_diff < tol)

    left_edges = numpy.arange(numpy.nanmin(out_mass), numpy.nanmax(out_mass),
                              0.01)
    bw = 0.2
    x = left_edges + 0.5 * bw
    nsimx = out_matched.shape[1]

    with plt.style.context("science"):
        plt.figure()

        for j, tol in enumerate(tolerances):

            med = csiborgtools.binned_statistic(out_mass, in_tolerance[:, j],
                                                left_edges, bw, numpy.median)
            lower = csiborgtools.binned_statistic(
                out_mass, in_tolerance[:, j], left_edges, bw,
                lambda x: numpy.percentile(x, 16))
            upper = csiborgtools.binned_statistic(
                out_mass, in_tolerance[:, j], left_edges, bw,
                lambda x: numpy.percentile(x, 84))
            lower /= nsimx
            upper /= nsimx
            med /= nsimx

            plt.plot(x, med, label=r"${} \%$".format(int(tol * 100)),)
            plt.fill_between(x, lower, upper, alpha=0.2)

        plt.xlim(x.min(), x.max())
        plt.ylim(0, 1)
        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        plt.ylabel(r"$f_{\rm in~tolerance}$")
        plt.legend(ncols=3)

        plt.tight_layout()
        fout = join(
            plt_utils.fout,
            f"expected_tolerance_{key}_{nsim0}_{simname}_{min_logmass}.png")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# --------------------------------------------------------------------------- #
###############################################################################
#                        Concentration reconstruction                         #
###############################################################################
# --------------------------------------------------------------------------- #


@cache_to_disk(120)
def get_max_overlap_velocity(kmax, nsim0, simname, min_logmass, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.summary.get_cross_sims(
        simname, nsim0, paths, min_logmass, smoothed=smoothed)

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.summary.NPairsOverlap(cat0, catxs, paths,
                                                min_logmass)

    log_mass0 = numpy.log10(reader.cat0(MASS_KINDS[simname]))
    vx, vy, vz = reader.cat0("vx"), reader.cat0("vy"), reader.cat0("vz")

    out_mass = numpy.full((kmax), numpy.nan)
    out_true = numpy.full((kmax, 3), numpy.nan)
    out_matched = numpy.full((kmax, len(nsimxs), 4), numpy.nan)

    for k0 in trange(kmax, desc="Looping over top masses"):
        k = numpy.argsort(log_mass0)[::-1][k0]
        for i, key in enumerate(["vx", "vy", "vz"]):
            vel, overlaps = reader.expected_property_single(k, key, smoothed,
                                                            in_log=False)
            out_matched[k0, :, i] = vel
        out_matched[k0, :, -1] = overlaps

        out_mass[k0] = log_mass0[k]
        out_true[k0, 0] = vx[k]
        out_true[k0, 1] = vy[k]
        out_true[k0, 2] = vz[k]

    return out_mass, out_true, out_matched


def expected_in_velocity(kmax, nsim0, simname, min_logmass, smoothed):
    cat0 = open_cat(nsim0, simname)
    pos = cat0.position(cartesian=False)

    if kmax == -1:
        kmax = numpy.sum(numpy.log10(cat0[MASS_KINDS[simname]]) > min_logmass) - 1  # noqa
        print(f"Selected kmax = -1, so doing {kmax}.")

    out_mass, out_true, out_matched = get_max_overlap_velocity(
        kmax, nsim0, simname, min_logmass, smoothed)
    __, out_true_conc, out_matched_conc, out_control_conc = get_expected_many_topmass(kmax, nsim0, simname, min_logmass, "conc", smoothed, False)  # noqa
    __, out_true_mass, out_matched_mass, out_control_mass = get_expected_many_topmass(kmax, nsim0, simname, min_logmass, "totpartmass", smoothed, False)  # noqa
    clusters = {
        'Virgo': {'l': 283.8, 'b': 74.5, 'distance_mpc': 17},
        'Fornax': {'l': 236.8, 'b': -53.6, 'distance_mpc': 19},
        'Coma': {'l': 58.1, 'b': 87.9, 'distance_mpc': 98},
        'Perseus': {'l': 150.6, 'b': -13.3, 'distance_mpc': 73},
        'Hydra': {'l': 269.1, 'b': 26.5, 'distance_mpc': 49},
        'Centaurus': {'l': 302.4, 'b': 21.6, 'distance_mpc': 52}
        }

    nsimx = out_matched.shape[1]
    # Calculate the cosine similarity between the true and expected velocity
    # and the difference in velocity.
    yalign = numpy.full((kmax, nsimx), numpy.nan)
    ydiff = numpy.full((kmax, nsimx), numpy.nan)
    w = numpy.full((kmax, nsimx), numpy.nan)
    for k in range(kmax):
        yalign[k] = csiborgtools.utils.cosine_similarity(out_true[k],
                                                         out_matched[k, :, :3])
        ydiff[k] = numpy.sum(
            (out_true[k] - out_matched[k, :, :3])**2, axis=1)**0.5
        w[k] = out_matched[k, :, -1]

    for k in range(0):
        k0 = numpy.argsort(cat0["totpartmass"])[::-1][k]
        with plt.style.context("science"):
            fig, axs = plt.subplots(nrows=3, ncols=3,
                                    figsize=(2 * 3.5, 2 * 2.625))
            fig.subplots_adjust(wspace=0., hspace=0.0)

            # Maximum overlap histogram
            axs[0, 0].hist(w[k], bins="auto", histtype="step", density=1)
            axs[0, 0].set_xlabel(r"$\max_{b \in \mathcal{B}} \mathcal{O}_{a b}$")  # noqa
            axs[0, 0].set_ylabel(r"Normalized counts")
            axs[0, 0].set_xlim(0.01, 1)
            # Cosine similarity alignment
            axs[0, 1].hist(yalign[k], weights=w[k], bins=20, histtype="step",
                           density=1)
            axs[0, 1].set_xlabel(r"$\cos \theta$")
            axs[0, 1].set_xlim(-1, 1)
            axs[0, 1].set_ylabel(r"Normalized counts")
            # Absolute difference in velocity
            axs[0, 2].hist(ydiff[k], weights=w[k], bins=20, histtype="step",
                           density=1)
            axs[0, 2].set_xlabel(r"$|\Delta \textbf{v}|~[\mathrm{km} / \mathrm{s}]$")  # noqa
            axs[0, 2].set_xlim(0)
            axs[0, 2].set_ylabel(r"Normalized counts")
            # Max overlap vs mass
            axs[1, 0].scatter(w[k], numpy.log10(out_matched_mass[k, :, 0]),
                              s=3)
            axs[1, 0].axhline(out_mass[k], color="red", ls="--")
            axs[1, 0].axhline(
                numpy.average(numpy.log10(out_matched_mass[k, :, 0]),
                              weights=w[k]),
                color="blue", ls="--")
            axs[1, 0].set_xlabel(r"$\max_{b \in \mathcal{B}} \mathcal{O}_{a b}$")  # noqa
            axs[1, 0].set_ylabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
            # Max overlap vs concentration
            axs[1, 1].scatter(w[k], out_matched_conc[k, :, 0], s=3)
            axs[1, 1].axhline(out_true_conc[k], color="red", ls="--")
            axs[1, 1].set_xlabel(r"$\max_{b \in \mathcal{B}} \mathcal{O}_{a b}$")  # noqa
            axs[1, 1].set_ylabel(r"$c$")
            # Alignment vs difference in velocity
            axs[1, 2].scatter(yalign[k], ydiff[k], s=3)
            axs[1, 2].set_xlabel(r"$\cos \theta$")
            axs[1, 2].set_ylabel(r"$|\Delta \textbf{v}|~[\mathrm{km} / \mathrm{s}]$")  # noqa
            axs[1, 2].set_ylim(0.01)
            # Sky positions
            c = SkyCoord(ra=pos[k0, 1] * u.degree, dec=pos[k0, 2] * u.degree,
                         frame='icrs')
            axs[2, 0].scatter(c.galactic.l, c.galactic.b, s=3, zorder=1,
                              c="red")
            dist = str(round(pos[k0, 0], 0))
            axs[2, 0].text(c.galactic.l.to_value(), c.galactic.b.to_value(),
                           dist, fontsize=6, ha='left',
                           va='bottom', zorder=0)
            for cluster in clusters.keys():
                l, b = clusters[cluster]['l'], clusters[cluster]['b']
                axs[2, 0].scatter(l, b, c="blue", s=2, zorder=0)
                dist = round(clusters[cluster]["distance_mpc"], 0)
                key = f"{cluster} ({dist})"
                axs[2, 0].text(l, b, key, fontsize=6, ha='right',
                               va='bottom', zorder=0)
                axs[2, 0].axhspan(-10, 10, alpha=0.2)
            axs[2, 0].set_xlabel(r"$l$")
            axs[2, 0].set_ylabel(r"$b$")
            # Max overlap vs alignment
            axs[2, 1].scatter(w[k], yalign[k], s=3)
            axs[2, 1].set_xlabel(r"$\max_{b \in \mathcal{B}} \mathcal{O}_{a b}$")  # noqa
            axs[2, 1].set_ylabel(r"$\cos \theta$")
            # Max overlap vs velocity difference
            axs[2, 2].scatter(w[k], ydiff[k], s=3)
            axs[2, 2].set_xlabel(r"$\max_{b \in \mathcal{B}} \mathcal{O}_{a b}$")  # noqa
            axs[2, 2].set_ylabel(r"$|\Delta \textbf{v}|~[\mathrm{km} / \mathrm{s}]$")  # noqa

            fig.tight_layout()
            fout = join(
                plt_utils.fout,
                f"vel_{k}_{nsim0}_{simname}_{min_logmass}.png")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
            plt.close()

    with plt.style.context("science"):
        plt.figure()

        x = numpy.asanyarray([numpy.nanmean(w[k]) for k in range(kmax)])
        y = [csiborgtools.summary.find_peak(yalign[k], w[k])
             for k in range(kmax)]
        y = numpy.arccos(y) * 180 / numpy.pi
        c = numpy.log10(out_true_mass)

        # plt.errorbar(x, y, xerr=[xlower, xupper], fmt="o", ms=3, lw=0.5,)
        # plt.hexbin(x, y, gridsize=30, mincnt=1, bins="log")
        plt.scatter(x, y, s=7.5, c=c, rasterized=True)

        plt.ylim(top=90)

        m = numpy.isfinite(x) & numpy.isfinite(y)
        xbins = numpy.arange(x[m].min(), x[m].max(), 0.1)
        xcent = 0.5 * (xbins[1:] + xbins[:-1])
        ymed, yerr = plt_utils.compute_error_bars(x[m], y[m], xbins, sigma=1)
        plt.errorbar(xcent, ymed, yerr=yerr, fmt="o", ms=3, lw=0.5, c="red",
                     capsize=3, ls="--")

        plt.colorbar(label=r"$\log M_{\rm tot} ~ [M_\odot / h]$", pad=0)
        plt.xlabel(r"$\langle \max_{b \in \mathcal{B}} \mathcal{O}_{a b} \rangle_{\mathcal{B}}$")  # noqa
        plt.ylabel(r"$\theta ~ [\deg]$")

        plt.tight_layout()
        fout = join(
            plt_utils.fout,
            f"stat_vel_agreement_{kmax}_{nsim0}_{simname}_{min_logmass}.pdf")
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
        # "get_expected_many_topmass",
        # "get_max_overlap_velocity",
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

    if False:
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
        mtot_vs_expected_key(7444, "csiborg", min_logmass, key, smoothed, 15)
        # if plot_quijote:
        #     mtot_vs_expected_key(0, "quijote", min_logmass, key, smoothed,
        #                          nbins)

    if True:
        mass_vs_separation(7444, "csiborg", min_logmass, nbins, smoothed,
                           boxsize=677.7)
        # if plot_quijote:
        #     mass_vs_separation(0, 1, "quijote", min_logmass, nbins,
        #                        smoothed, boxsize=1000, plot_std=False)

    if False:
        matching_max_vs_overlap(min_logmass)

    if False:
        # mtot_vs_expected_single("max", 7444, "csiborg", min_logmass,
        #                         "totpartmass", True)
        # mtot_vs_expected_single("maxmass__0", 7444, "csiborg",
        #                         min_logmass, "lambda200c", True)
        for i in range(25):
            mtot_vs_expected_single(f"maxmass__{i}", 7444, "csiborg",
                                    min_logmass, "conc", True)
        # if plot_quijote:
        #     mtot_vs_expected_single("max", 0, "quijote", min_logmass,
        #                             "totpartmass", True, True)

    if False:
        mtot_vs_expected_single_panel("maxmass__0", 7444, "csiborg",
                                      min_logmass, True)

    if False:
        kmax = 100
        expected_in_tolerance(kmax, 7444, "csiborg", min_logmass, "conc", True)

    if False:
        kmax = 1000
        expected_in_velocity(kmax, 7444 + 24 * 30, "csiborg", min_logmass, smoothed)
