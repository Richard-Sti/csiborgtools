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
from os.path import join

import matplotlib.pyplot as plt
import numpy
import scienceplots  # noqa
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

    reader = csiborgtools.read.NPairsOverlap(cat0, catxs, paths, min_logmass)
    mass0 = reader.cat0(MASS_KINDS[simname])
    mask = mass0 > 10**min_logmass

    return {"mass0": mass0[mask],
            "hid0": reader.cat0("index")[mask],
            "summed_overlap": reader.summed_overlap(smoothed)[mask],
            "max_overlap": reader.max_overlap(smoothed)[mask],
            "prob_nomatch": reader.prob_nomatch(smoothed)[mask],
            }


@cache_to_disk(120)
def get_property_maxoverlap(nsim0, simname, min_logmass, key, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(simname,  nsim0, paths,
                                              min_logmass, smoothed=smoothed)

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.read.NPairsOverlap(cat0, catxs, paths, min_logmass)
    mass0 = reader.cat0(MASS_KINDS[simname])
    mask = mass0 > 10**min_logmass

    return {"mass0": mass0[mask],
            "prop0": reader.cat0(key)[mask],
            "max_overlap": reader.max_overlap(smoothed)[mask],
            "prop_maxoverlap": reader.max_overlap_key(key, smoothed)[mask],
            }


@cache_to_disk(120)
def get_expected_mass(nsim0, simname, min_overlap, min_logmass, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(simname, nsim0, paths,
                                              min_logmass, smoothed=True)

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.read.NPairsOverlap(cat0, catxs, paths, min_logmass)
    mass0 = reader.cat0(MASS_KINDS[simname])
    mask = mass0 > 10**min_logmass
    mu, std = reader.counterpart_mass(True, overlap_threshold=min_overlap,
                                      in_log=False, return_full=False)

    return {"mass0": mass0[mask],
            "mu": mu[mask],
            "std": std[mask],
            "prob_nomatch": reader.prob_nomatch(smoothed)[mask],
            }

# ---------------------------------------------------------------------------- #
################################################################################
#                   Total DM halo mass vs pair overlaps                        #
################################################################################
# ---------------------------------------------------------------------------- #


@cache_to_disk(120)
def get_mtot_vs_all_pairoverlap(nsim0, simname, mass_kind, min_logmass,
                                smoothed, nbins):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(simname, nsim0, paths, min_logmass,
                                              smoothed=smoothed)
    nsimxs = nsimxs

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.read.NPairsOverlap(cat0, catxs, paths, min_logmass)

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


def mtot_vs_all_pairoverlap(nsim0, simname, mass_kind, min_logmass, smoothed,
                            nbins, ext="png"):
    x, y, xbins = get_mtot_vs_all_pairoverlap(nsim0, simname, mass_kind,
                                              min_logmass, smoothed, nbins)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        hb = plt.hexbin(x, y, mincnt=1, gridsize=50, bins="log")
        plt_utils.normalize_hexbin(hb)

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


        plt.colorbar(hb, label="Normalized counts in bins")
        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        plt.ylabel("Pair overlap")
        plt.xlim(numpy.min(x))
        plt.ylim(0., 1.)

        plt.tight_layout()
        fout = join(plt_utils.fout, f"mass_vs_pair_overlap_{simname}_{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------- #
################################################################################
#                  Total DM halo mass vs maximum pair overlaps                 #
################################################################################
# ---------------------------------------------------------------------------- #


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
        return numpy.max(y_)

    reader = csiborgtools.read.NPairsOverlap(cat0, catxs, paths, min_logmass)

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

    plt.close("all")
    with plt.style.context(plt_utils.mplstyle):
        plt.figure()

        hb = plt.hexbin(x, y, mincnt=1, gridsize=50, bins="log")
        plt_utils.normalize_hexbin(hb)

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


        plt.colorbar(label="Normalized counts in bins")
        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        plt.ylabel("Maximum pair overlap")
        plt.ylim(-0.02, 1.)
        plt.xlim(numpy.min(x) - 0.05)

        plt.tight_layout()
        fout = join(plt_utils.fout, f"mass_vs_max_pair_overlap{nsim0}.{ext}")
        print(f"Saving to `{fout}`.")
        plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------- #
################################################################################
#                  Total DM halo mass vs summed pair overlaps                  #
################################################################################
# ---------------------------------------------------------------------------- #


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
        im1 = axs[0].hexbin(mass0, mean_overlap, mincnt=1, bins="log", gridsize=30)
        plt_utils.normalize_hexbin(im1)

        y_median, yerr = plt_utils.compute_error_bars(
            mass0, mean_overlap, xbins, sigma=2)
        axs[0].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                        color='red', ls='dashed', capsize=3)


        im2 = axs[1].hexbin(mass0, std_overlap, mincnt=1, bins="log", gridsize=30)
        plt_utils.normalize_hexbin(im2)

        y_median, yerr = plt_utils.compute_error_bars(
            mass0, std_overlap, xbins, sigma=2)
        axs[1].errorbar(0.5 * (xbins[1:] + xbins[:-1]), y_median, yerr=yerr,
                        color='red', ls='dashed', capsize=3)

        if simname == "csiborg":
            x_quijote = get_overlap_summary(0, "quijote", min_logmass, smoothed)
            mass0_quijote = numpy.log10(x_quijote["mass0"])
            mean_overlap_quijote = numpy.nanmean(x_quijote["summed_overlap"], axis=1)
            std_overlap_quijote = numpy.nanstd(x_quijote["summed_overlap"], axis=1)
            xbins_quijote = numpy.linspace(numpy.nanmin(mass0), numpy.nanmax(mass0), nbins)

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

        label = ["Normalized bin counts", "Normalized bin counts",
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


def plot_kl_vs_overlap(runs, nsim, kwargs, runs_to_mass, plot_std=True,
                       upper_threshold=False):
    """
    Plot KL divergence vs overlap for CSiBORG.

    Parameters
    ----------
    runs : str
        Run names.
    nsim : int
        Simulation index.
    kwargs : dict
        Nearest neighbour reader keyword arguments.
    runs_to_mass : dict
        Dictionary mapping run names to total halo mass range.
    plot_std : bool, optional
        Whether to plot the standard deviation of the overlap distribution.
    upper_threshold : bool, optional
        Whether to enforce an upper threshold on halo mass.

    Returns
    -------
    None
    """
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    nn_reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)

    xs, ys1, ys2, cs = [], [], [], []
    for run in runs:
        nn_data = nn_reader.read_single("csiborg", run, nsim, nobs=None)
        nn_hindxs = nn_data["ref_hindxs"]
        mass, overlap_hindxs, __, summed_overlap, prob_nomatch = get_overlap_summary("csiborg", nsim)  # noqa

        # We need to match the hindxs between the two.
        hind2overlap_array = {hind: i for i, hind in enumerate(overlap_hindxs)}
        mask = numpy.asanyarray([hind2overlap_array[hind]
                                 for hind in nn_hindxs])
        summed_overlap = summed_overlap[mask]
        prob_nomatch = prob_nomatch[mask]
        mass = mass[mask]

        x = make_kl("csiborg", run, nsim, nobs=None, kwargs=kwargs)
        y1 = 1 - numpy.mean(prob_nomatch, axis=1)
        y2 = numpy.std(prob_nomatch, axis=1)
        cmin, cmax = make_binlims(run, runs_to_mass, upper_threshold)
        mask = (mass >= cmin) & (mass < cmax if upper_threshold else True)
        xs.append(x[mask])
        ys1.append(y1[mask])
        ys2.append(y2[mask])
        cs.append(numpy.log10(mass[mask]))

    xs = numpy.concatenate(xs)
    ys1 = numpy.concatenate(ys1)
    ys2 = numpy.concatenate(ys2)
    cs = numpy.concatenate(cs)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        plt.hexbin(xs, ys1, C=cs, gridsize=50, mincnt=0,
                   reduce_C_function=numpy.median)
        mask = numpy.isfinite(xs) & numpy.isfinite(ys1)
        corr = plt_utils.latex_float(*kendalltau(xs[mask], ys1[mask]))
        plt.title(r"$\tau = {}, p = {}$".format(*corr), fontsize="small")

        plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")
        plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
        plt.ylabel("1 - mean prob. of no match")

        plt.tight_layout()
        for ext in ["png"]:
            nsim = str(nsim).zfill(5)
            fout = join(plt_utils.fout,
                        f"kl_vs_overlap_mean_{nsim}_{runs}.{ext}")
            if upper_threshold:
                fout = fout.replace(f".{ext}", f"_upper_threshold.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()

    if not plot_std:
        return

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        plt.hexbin(xs, ys2, C=cs, gridsize=50, mincnt=0,
                   reduce_C_function=numpy.median)
        plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")
        plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
        plt.ylabel(r"Ensemble std of summed overlap")
        mask = numpy.isfinite(xs) & numpy.isfinite(ys2)
        corr = plt_utils.latex_float(*kendalltau(xs[mask], ys2[mask]))
        plt.title(r"$\tau = {}, p = {}$".format(*corr), fontsize="small")

        plt.tight_layout()
        for ext in ["png"]:
            nsim = str(nsim).zfill(5)
            fout = join(plt_utils.fout,
                        f"kl_vs_overlap_std_{nsim}_{runs}.{ext}")
            if upper_threshold:
                fout = fout.replace(f".{ext}", f"_upper_threshold.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    min_logmass = 13.25
    smoothed = True
    nbins = 10
    ext = "png"

    if False:
        mtot_vs_all_pairoverlap(7444, "csiborg", "fof_totpartmass",
                                min_logmass, smoothed, nbins, ext=ext)

        mtot_vs_all_pairoverlap(0, "quijote", "group_mass",
                                min_logmass, smoothed, nbins, ext=ext)

    if False:
        mtot_vs_maxpairoverlap(7444, "csiborg", "fof_totpartmass", min_logmass,
                               smoothed, nbins, ext=ext)
        mtot_vs_maxpairoverlap(0, "quijote", "group_mass", min_logmass,
                               smoothed, nbins, ext=ext)

    if True:
        mtot_vs_summedpairoverlap(7444, "csiborg", min_logmass, smoothed,
                                  nbins, ext)
        mtot_vs_summedpairoverlap(0, "quijote", min_logmass, smoothed, nbins,
                                  ext)