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
    sigma = 2

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
#                  Total DM halo mass vs summed pair overlaps                 #
###############################################################################
# --------------------------------------------------------------------------- #

def mtot_vs_summedpairoverlap(nsim0, simname, min_logmass, smoothed, nbins,
                              ext="png"):
    x = get_overlap_summary(nsim0, simname, min_logmass, smoothed)

    mass0 = numpy.log10(x["mass0"])
    mean_overlap = numpy.nanmean(x["summed_overlap"], axis=1)
    std_overlap = numpy.nanstd(x["summed_overlap"], axis=1)
    # mean_prob_nomatch = numpy.nanmean(x["prob_nomatch"], axis=1)

    xbins = numpy.linspace(numpy.nanmin(mass0), numpy.nanmax(mass0), nbins)

    with plt.style.context(plt_utils.mplstyle):
        fig, axs = plt.subplots(nrows=2, figsize=(3.5, 2.625 * 2))
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

        # im3 = axs[2].scatter(1 - mean_overlap, mean_prob_nomatch, c=mass0,
        #                      s=2, rasterized=True)

        # t = numpy.linspace(numpy.nanmin(1 - mean_overlap), 1, 100)
        # axs[2].plot(t, t, color="red", linestyle="--")

        axs[0].set_ylim(0., 0.75)
        axs[0].set_xlim(numpy.min(mass0))
        axs[0].set_xlim(numpy.min(mass0))
        axs[0].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[0].set_ylabel("Mean summed overlap")
        axs[1].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[1].set_ylabel("Uncertainty of summed overlap")
        # axs[2].set_xlabel(r"$1 - $ mean summed overlap")
        # axs[2].set_ylabel("Mean prob. of no match")

        label = ["Bin counts", "Bin counts"]
        ims = [im1, im2]
        for i in range(2):
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
    mu, std = csiborgtools.summary.weighted_stats(dist, overlap, min_weight=0)

    mask = numpy.isfinite(dist[:, 0])
    mass = mass[mask]
    mu = mu[mask]
    mu = numpy.log10(mu)

    return mass, mu


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

    mean_expected, upper_expected, lower_expected = reader.expected_property(
        MASS_KINDS[simname], smoothed, min_logmass)

    return {"mass0": mass0[mask],
            "mu": mean_expected[mask],
            "lower": lower_expected[mask],
            "upper": upper_expected[mask],
            "prob_nomatch": reader.prob_nomatch(smoothed)[mask],
            }


def mtot_vs_expected_mass(nsim0, simname, min_logmass, smoothed, nbins,
                          max_prob_nomatch=1, ext="png"):
    x = get_expected_mass(nsim0, simname, min_logmass, smoothed)

    mass = x["mass0"]
    print(mass.shape)
    mu = x["mu"]
    std = (x["upper"] - x["lower"]) / 2
    prob_nomatch = x["prob_nomatch"]

    mass = numpy.log10(mass)
    prob_nomatch = numpy.nanmedian(prob_nomatch, axis=1)

    mask = numpy.isfinite(mass) & numpy.isfinite(mu) & numpy.isfinite(std)
    mask &= (prob_nomatch <= max_prob_nomatch)

    xbins = numpy.linspace(*numpy.percentile(mass[mask], [0, 100]), nbins)

    with plt.style.context(plt_utils.mplstyle):
        fig, axs = plt.subplots(ncols=3, figsize=(3.5 * 2, 2.625))

        im0 = axs[0].hexbin(mass[mask], mu[mask], mincnt=1, bins="log",
                            gridsize=30,)
        # y_median, yerr = plt_utils.compute_error_bars(mass[mask], mu[mask],
        #                                               xbins, sigma=2)
        # axs[0].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
        #                 color='red', ls='dashed', capsize=3)

        im1 = axs[1].hexbin(mass[mask], std[mask], mincnt=1, bins="log",
                            gridsize=30)
        # y_median, yerr = plt_utils.compute_error_bars(mass[mask], std[mask],
        #                                               xbins, sigma=2)
        # axs[1].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
        #                 color='red', ls='dashed', capsize=3)

        im2 = axs[2].hexbin(1 - prob_nomatch[mask], mu[mask] - mass[mask],
                            gridsize=30, C=mass[mask],
                            reduce_C_function=numpy.nanmedian)

        axs[2].axhline(0, color="red", linestyle="--", alpha=0.5)
        axs[0].set_xlabel(r"$\log M_{\rm tot, ref} ~ [M_\odot / h]$")
        axs[0].set_ylabel(r"$\log M_{\rm tot, exp} ~ [M_\odot / h]$")
        axs[1].set_xlabel(r"$\log M_{\rm tot, ref} ~ [M_\odot / h]$")
        axs[1].set_ylabel(r"$\sigma_{\log M_{\rm tot, exp}}$")
        axs[2].set_xlabel(r"Median prob. of match")
        axs[2].set_ylabel(r"$\log (M_{\rm tot, exp} / M_{\rm tot, ref})$")

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

    mean_expected, upper_expected, lower_expected = reader.expected_property(
        key, smoothed, min_logmass)

    return {"mass0": mass0[mask],
            "prop0": reader.cat0(key)[mask],
            "mu": mean_expected[mask],
            "lower": lower_expected[mask],
            "upper": upper_expected[mask],
            "prob_nomatch": reader.prob_nomatch(smoothed)[mask],
            }


def mtot_vs_expected_key(nsim0, simname, min_logmass, key, smoothed, nbins):
    mass_kind = MASS_KINDS[simname]
    assert key != mass_kind

    x = get_expected_key(nsim0, simname, min_logmass, key, smoothed)
    mass0 = x["mass0"]
    prop0 = x["prop0"]
    mu = x["mu"]
    std = (x["upper"] - x["lower"]) / 2
    prob_nomatch = x["prob_nomatch"]

    xlabels = {"lambda200c": r"\log \lambda_{\rm 200c}"}
    key_label = xlabels.get(key, key)

    mass0 = numpy.log10(mass0)
    prop0 = numpy.log10(prop0)

    # mu = numpy.nanmean(stat, axis=1)
    # std = numpy.nanstd(numpy.log10(stat), axis=1)
    # mu = numpy.log10(mu)

    mask = numpy.isfinite(prop0) & numpy.isfinite(mu) & numpy.isfinite(std)
    xbins = numpy.linspace(*numpy.percentile(mass0[mask], [0, 100]), nbins)

    with plt.style.context(plt_utils.mplstyle):
        fig, axs = plt.subplots(ncols=3, figsize=(3.5 * 2, 2.625))

        im0 = axs[0].hexbin(mass0, mu - prop0, mincnt=1, bins="log",
                            gridsize=30)

        # y_median, yerr = plt_utils.compute_error_bars(
        #     mass0[mask], mu[mask] - prop0[mask], xbins, sigma=2)
        # axs[0].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
        #                 color='red', ls='dashed', capsize=3)

        im1 = axs[1].hexbin(mass0, std, mincnt=1, bins="log", gridsize=30)
        # y_median, yerr = plt_utils.compute_error_bars(
        #     mass0[mask], std[mask], xbins, sigma=2)
        # axs[1].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
        #                 color='red', ls='dashed', capsize=3)

        # if key == "lambda200c":
        #     axs[0].axhline(-0.46332417, color="violet", linestyle="--")
        #     axs[0].axhline(+0.58750349, color="violet", linestyle="--")
        #     # axs[1].axhline(0.26309842, color="violet", linestyle="--")

        # if key == "conc":
        #     x_ = numpy.array([13.37577211, 13.62729724, 13.87882238, 14.13034752, 14.38187265, 14.63339779, 14.88492293, 15.13644807, 15.3879732])  # noqa
        #     yerr_ = 1 * numpy.array([0.17946734, 0.20949572, 0.21856549, 0.2001923 , 0.19134766, 0.16186188, 0.16695528, 0.23876699, 0.17740743])   # noqa
        #     axs[1].plot(x_, yerr_, color="violet", linestyle="--")

        im2 = axs[2].hexbin(prop0, mu, mincnt=1, bins="log", gridsize=30)
        m = numpy.isfinite(prop0) & numpy.isfinite(mu)
        print("True to expectation corr: ", kendalltau(prop0[m], mu[m]))

        axs[0].set_xlabel(r"$\log M_{\rm tot, ref} ~ [M_\odot / h]$")
        if key == "lambda200c":
            axs[0].set_ylabel(r"$\log (\lambda_{\rm 200c, match} / \lambda_{\rm 200c, ref})$")  # noqa
        elif key == "conc":
            axs[0].set_ylabel(r"$\log (c_{\rm 200c, match} / c_{\rm 200c, ref})$")  # noqa
        else:
            axs[0].set_ylabel(r"True - max. overlap mean of ${}$"
                              .format(key_label))

        axs[1].set_xlabel(r"$\log M_{\rm tot, ref} ~ [M_\odot / h]$")
        if key == "lambda200c":
            axs[1].set_ylabel(r"$\sigma_{\lambda_{\rm 200c, match}}$")
        elif key == "conc":
            axs[1].set_ylabel(r"$\sigma_{c_{\rm 200c, match}}$")
        else:
            axs[1].set_ylabel(r"Max. overlap std. of ${}$".format(key_label))

        if key == "lambda200c":
            axs[2].set_xlabel(r"$\log \lambda_{\rm 200c, ref}$")
            axs[2].set_ylabel(r"$\log \lambda_{\rm 200c, match}$")
        elif key == "conc":
            axs[2].set_xlabel(r"$\log c_{\rm 200c, ref}$")
            axs[2].set_ylabel(r"$\log c_{\rm 200c, match}$")
        else:
            axs[2].set_xlabel(r"${}$".format(key_label))
            axs[2].set_ylabel(r"Max. overlap mean of ${}$".format(key_label))

        t = numpy.linspace(*numpy.percentile(prop0[m], [5, 95]), 1000)
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
        fout = join(plt_utils.fout, f"max_{key}_{simname}_{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# # --------------------------------------------------------------------------- #
# ###############################################################################
# #                      Max's matching vs overlap success                      #
# ###############################################################################
# # --------------------------------------------------------------------------- #
#
#
# @cache_to_disk(120)
# def get_matching_max_vs_overlap(simname, nsim0, min_logmass, mult):
#     paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
#
#     nsimsx = [nsim for nsim in paths.get_ics(simname) if nsim != nsim0]
#     for i in trange(len(nsimsx), desc="Loading data"):
#         nsimx = nsimsx[i]
#         fpath = paths.match_max(simname, nsim0, nsimx, min_logmass,
#                                 mult=mult)
#
#         data = numpy.load(fpath, allow_pickle=True)
#
#         if i == 0:
#             mass0 = data["mass0"]
#             max_overlap = numpy.full((mass0.size, len(nsimsx)), numpy.nan)
#             match_overlap = numpy.full((mass0.size, len(nsimsx)), numpy.nan)
#             success = numpy.zeros((mass0.size, len(nsimsx)), numpy.bool_)
#
#         max_overlap[:, i] = data["max_overlap"]
#         match_overlap[:, i] = data["match_overlap"]
#         success[:, i] = data["success"]
#
#     return {"mass0": mass0, "max_overlap": max_overlap,
#             "match_overlap": match_overlap, "success": success}
#
#
# def matching_max_vs_overlap(simname, nsim0, min_logmass):
#     left_edges = numpy.arange(min_logmass, 15, 0.1)
#     nsims = 100 if simname == "csiborg" else 9
#
#     with plt.style.context("science"):
#         fig, axs = plt.subplots(ncols=2, figsize=(3.5 * 2, 2.625))
#         cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#         for n, mult in enumerate([2.5, 5., 7.5, 10.0]):
#             x = get_matching_max_vs_overlap(simname,
#                                             nsim0, min_logmass, mult=mult)
#             mass0 = numpy.log10(x["mass0"])
#             max_overlap = x["max_overlap"]
#             match_overlap = x["match_overlap"]
#             success = x["success"]
#
#             nbins = len(left_edges)
#             y = numpy.full((nbins, nsims), numpy.nan)
#             y2 = numpy.full(nbins, numpy.nan)
#             y2err = numpy.full(nbins, numpy.nan)
#             for i in range(nbins):
#                 m = mass0 > left_edges[i]
#                 for j in range(nsims):
#                     y[i, j] = numpy.sum(
#                         max_overlap[m, j] == match_overlap[m, j])
#                     y[i, j] /= numpy.sum(success[m, j])
#
#                 y2[i] = numpy.mean(numpy.sum(success[m, :], axis=1) / nsims)
#                 y2err[i] = numpy.std(numpy.sum(success[m, :], axis=1) / nsims)
#
#             offset = numpy.random.normal(0, 0.015)
#
#             ysummary = numpy.percentile(y, [16, 50, 84], axis=1)
#             axs[0].errorbar(
#                 left_edges + offset, ysummary[1],
#                 yerr=[ysummary[1] - ysummary[0], ysummary[2] - ysummary[1]],
#                 capsize=4, c=cols[n], ls="dashed",
#                 label=r"$\leq {}~R_{{\rm 200c}}$".format(mult), errorevery=2)
#
#             axs[1].errorbar(left_edges + offset, y2, yerr=y2err,
#                             capsize=4, errorevery=2, c=cols[n], ls="dashed")
#
#         axs[0].legend(ncols=2, fontsize="small")
#         for i in range(2):
#             axs[i].set_xlabel(r"$\log M_{\rm tot, min} ~ [M_\odot / h]$")
#
#         axs[1].set_ylim(0)
#         axs[0].set_ylabel(r"$f_{\rm agreement}$")
#         axs[1].set_ylabel(r"$f_{\rm match}$")
#
#         fig.tight_layout()
#         fout = join(
#             plt_utils.fout,
#             f"matching_max_agreement_{simname}_{nsim0}_{min_logmass}.png")
#         print(f"Saving to `{fout}`.")
#         fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
#         plt.close()
#
#
# # --------------------------------------------------------------------------- #
# ###############################################################################
# #                       KL final snapshot vs overlaps                         #
# ###############################################################################
# # --------------------------------------------------------------------------- #
#
#
# # def plot_kl_vs_overlap(runs, nsim, kwargs, runs_to_mass, plot_std=True,
# #                        upper_threshold=False):
# #     """
# #     Plot KL divergence vs overlap for CSiBORG.
# #
# #     Parameters
# #     ----------
# #     runs : str
# #         Run names.
# #     nsim : int
# #         Simulation index.
# #     kwargs : dict
# #         Nearest neighbour reader keyword arguments.
# #     runs_to_mass : dict
# #         Dictionary mapping run names to total halo mass range.
# #     plot_std : bool, optional
# #         Whether to plot the standard deviation of the overlap distribution.
# #     upper_threshold : bool, optional
# #         Whether to enforce an upper threshold on halo mass.
# #
# #     Returns
# #     -------
# #     None
# #     """
# #     paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
# #     nn_reader = csiborgtools.summary.NearestNeighbourReader(**kwargs, paths=paths)
# #
# #     xs, ys1, ys2, cs = [], [], [], []
# #     for run in runs:
# #         nn_data = nn_reader.read_single("csiborg", run, nsim, nobs=None)
# #         nn_hindxs = nn_data["ref_hindxs"]
# #         mass, overlap_hindxs, __, summed_overlap, prob_nomatch = get_overlap_summary("csiborg", nsim)  # noqa
# #
# #         # We need to match the hindxs between the two.
# #         hind2overlap_array = {hind: i for i, hind in enumerate(overlap_hindxs)}
# #         mask = numpy.asanyarray([hind2overlap_array[hind]
# #                                  for hind in nn_hindxs])
# #         summed_overlap = summed_overlap[mask]
# #         prob_nomatch = prob_nomatch[mask]
# #         mass = mass[mask]
# #
# #         x = make_kl("csiborg", run, nsim, nobs=None, kwargs=kwargs)
# #         y1 = 1 - numpy.mean(prob_nomatch, axis=1)
# #         y2 = numpy.std(prob_nomatch, axis=1)
# #         cmin, cmax = make_binlims(run, runs_to_mass, upper_threshold)
# #         mask = (mass >= cmin) & (mass < cmax if upper_threshold else True)
# #         xs.append(x[mask])
# #         ys1.append(y1[mask])
# #         ys2.append(y2[mask])
# #         cs.append(numpy.log10(mass[mask]))
# #
# #     xs = numpy.concatenate(xs)
# #     ys1 = numpy.concatenate(ys1)
# #     ys2 = numpy.concatenate(ys2)
# #     cs = numpy.concatenate(cs)
# #
# #     with plt.style.context(plt_utils.mplstyle):
# #         plt.figure()
# #         plt.hexbin(xs, ys1, C=cs, gridsize=50, mincnt=0,
# #                    reduce_C_function=numpy.median)
# #         mask = numpy.isfinite(xs) & numpy.isfinite(ys1)
# #         corr = plt_utils.latex_float(*kendalltau(xs[mask], ys1[mask]))
# #         plt.title(r"$\tau = {}, p = {}$".format(*corr), fontsize="small")
# #
# #         plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")
# #         plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
# #         plt.ylabel("1 - mean prob. of no match")
# #
# #         plt.tight_layout()
# #         for ext in ["png"]:
# #             nsim = str(nsim).zfill(5)
# #             fout = join(plt_utils.fout,
# #                         f"kl_vs_overlap_mean_{nsim}_{runs}.{ext}")
# #             if upper_threshold:
# #                 fout = fout.replace(f".{ext}", f"_upper_threshold.{ext}")
# #             print(f"Saving to `{fout}`.")
# #             plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
# #         plt.close()
# #
# #     if not plot_std:
# #         return
# #
# #     with plt.style.context(plt_utils.mplstyle):
# #         plt.figure()
# #         plt.hexbin(xs, ys2, C=cs, gridsize=50, mincnt=0,
# #                    reduce_C_function=numpy.median)
# #         plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")
# #         plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
# #         plt.ylabel(r"Ensemble std of summed overlap")
# #         mask = numpy.isfinite(xs) & numpy.isfinite(ys2)
# #         corr = plt_utils.latex_float(*kendalltau(xs[mask], ys2[mask]))
# #         plt.title(r"$\tau = {}, p = {}$".format(*corr), fontsize="small")
# #
# #         plt.tight_layout()
# #         for ext in ["png"]:
# #             nsim = str(nsim).zfill(5)
# #             fout = join(plt_utils.fout,
# #                         f"kl_vs_overlap_std_{nsim}_{runs}.{ext}")
# #             if upper_threshold:
# #                 fout = fout.replace(f".{ext}", f"_upper_threshold.{ext}")
# #             print(f"Saving to `{fout}`.")
# #             plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
# #         plt.close()


if __name__ == "__main__":
    min_logmass = 13.25
    smoothed = True
    nbins = 10
    ext = "pdf"
    plot_quijote = True
    min_maxoverlap = 0.

    funcs = [
        # "get_overlap_summary",
        # "get_mtot_vs_all_pairoverlap",
        # "get_mtot_vs_maxpairoverlap",
        # "get_mass_vs_separation",
        # "get_expected_mass",
        # "get_expected_key",
        ]
    for func in funcs:
        print(f"Cleaning up cache for `{func}`.")
        delete_disk_caches_for_function(func)

    if True:
        mtot_vs_all_pairoverlap(7444, "csiborg", min_logmass, smoothed,
                                nbins, ext=ext)
        if plot_quijote:
            mtot_vs_all_pairoverlap(0, "quijote",  min_logmass, smoothed,
                                    nbins, ext=ext)

    if True:
        mtot_vs_maxpairoverlap(7444, "csiborg", "fof_totpartmass", min_logmass,
                               smoothed, nbins, ext=ext)
        if plot_quijote:
            mtot_vs_maxpairoverlap(0, "quijote", "group_mass", min_logmass,
                                   smoothed, nbins, ext=ext)

    if True:
        mtot_vs_summedpairoverlap(7444, "csiborg", min_logmass, smoothed,
                                  nbins, ext)
        if plot_quijote:
            mtot_vs_summedpairoverlap(0, "quijote", min_logmass, smoothed,
                                      nbins, ext)

    if True:
        mtot_vs_expected_mass(7444, "csiborg", min_logmass, smoothed, nbins,
                              max_prob_nomatch=1, ext=ext)
        if plot_quijote:
            mtot_vs_expected_mass(0, "quijote", min_logmass, smoothed, nbins,
                                  min_overlap=0, max_prob_nomatch=1, ext=ext)

    if True:
        key = "lambda200c"
        mtot_vs_expected_key(7444, "csiborg", min_logmass, key, smoothed,
                             nbins)
        if plot_quijote:
            mtot_vs_expected_key(0, "quijote", min_logmass, key, smoothed,
                                 nbins)

    if True:
        mass_vs_separation(7444, 7444 + 24, "csiborg", min_logmass, nbins,
                           smoothed, boxsize=677.7, plot_std=False)
        if plot_quijote:
            mass_vs_separation(0, 1, "quijote", min_logmass, nbins,
                               smoothed, boxsize=1000, plot_std=False)

    if False:
        matching_max_vs_overlap("csiborg", 7444, min_logmass)

        if plot_quijote:
            matching_max_vs_overlap("quijote", 0, min_logmass)

    if False:
        mtot_vs_maxpairoverlap_consistency(
            7444, "csiborg", "fof_totpartmass", min_logmass, smoothed,
            ext="png")
        # if plot_quijote:
        #     mtot_vs_maxpairoverlap_consistency(
        #         0, "quijote", "group_mass", min_logmass, smoothed,
        #         ext="png")


