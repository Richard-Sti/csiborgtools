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
from tqdm import tqdm

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


@cache_to_disk(21)
def get_overlap_summary(nsim0, simname, min_logmass, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(simname, nsim0, paths,
                                              min_logmass, smoothed=smoothed)[:2]
    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.read.NPairsOverlap(cat0, catxs, paths, min_logmass)
    return {"mass0": reader.cat0(MASS_KINDS[simname]),
            "hid0": reader.cat0("index"),
            "summed_overlap": reader.summed_overlap(smoothed),
            "max_overlap": reader.max_overlap(smoothed),
            "prob_nomatch": reader.prob_nomatch(smoothed),
            }


@cache_to_disk(7)
def get_property_maxoverlap(nsim0, simname, min_logmass, key, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(simname,  nsim0, paths,
                                              min_logmass, smoothed=smoothed)

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.read.NPairsOverlap(cat0, catxs, paths, min_logmass)
    return {"mass0": reader.cat0(MASS_KINDS[simname]),
            "prop0": reader.cat0(key),
            "max_overlap": reader.max_overlap(smoothed),
            "prop_maxoverlap": reader.max_overlap_key(key, smoothed),
            }


@cache_to_disk(21)
def get_expected_mass(nsim0, simname, min_overlap, min_logmass, smoothed):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(simname, nsim0, paths,
                                              min_logmass, smoothed=True)

    cat0 = open_cat(nsim0, simname)
    catxs = open_cats(nsimxs, simname)

    reader = csiborgtools.read.NPairsOverlap(cat0, catxs, paths, min_logmass)
    mu, std = reader.counterpart_mass(True, overlap_threshold=min_overlap,
                                      in_log=False, return_full=False)

    return {"mass0": reader.cat0(MASS_KINDS[simname]),
            "mu": mu,
            "std": std,
            "prob_nomatch": reader.prob_nomatch(smoothed),
            }


def plot_maxoverlapstat(nsim0, key):
    """
    Plot the mass of the reference haloes against the value of the maximum
    overlap statistic.

    Parameters
    ----------
    nsim0 : int
        Reference simulation index.
    key : str
        Property to get.
    """
    assert key != "totpartmass"
    mass0, key_val, __, stat = get_property_maxoverlap("csiborg", nsim0, key)

    xlabels = {"lambda200c": r"\log \lambda_{\rm 200c}"}
    key_label = xlabels.get(key, key)

    mass0 = numpy.log10(mass0)
    key_val = numpy.log10(key_val)

    mu = numpy.mean(stat, axis=1)
    std = numpy.std(numpy.log10(stat), axis=1)
    mu = numpy.log10(mu)

    with plt.style.context(plt_utils.mplstyle):
        fig, axs = plt.subplots(ncols=3, figsize=(3.5 * 2, 2.625))

        im0 = axs[0].hexbin(mass0, mu, mincnt=1, bins="log", gridsize=30)
        im1 = axs[1].hexbin(mass0, std, mincnt=1, bins="log", gridsize=30)
        im2 = axs[2].hexbin(key_val, mu, mincnt=1, bins="log", gridsize=30)
        m = numpy.isfinite(key_val) & numpy.isfinite(mu)
        print("True to expectation corr: ", kendalltau(key_val[m], mu[m]))

        axs[0].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[0].set_ylabel(r"Max. overlap mean of ${}$".format(key_label))
        axs[1].set_xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        axs[1].set_ylabel(r"Max. overlap std. of ${}$".format(key_label))
        axs[2].set_xlabel(r"${}$".format(key_label))
        axs[2].set_ylabel(r"Max. overlap mean of ${}$".format(key_label))

        ims = [im0, im1, im2]
        for i in range(3):
            axins = inset_axes(axs[i], width="100%", height="5%",
                               loc='upper center', borderpad=-0.75)
            fig.colorbar(ims[i], cax=axins, orientation="horizontal",
                         label="Bin counts")
            axins.xaxis.tick_top()
            axins.xaxis.set_tick_params(labeltop=True)
            axins.xaxis.set_label_position("top")

        fig.tight_layout()
        for ext in ["png"]:
            fout = join(plt_utils.fout,
                        f"max_{key}_{nsim0}.{ext}")
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


###############################################################################
#                        Command line interface                               #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()
    neighbour_kwargs = csiborgtools.neighbour_kwargs

    runs_to_mass = {
        "mass001": (12.4, 12.8),
        "mass002": (12.6, 13.0),
        "mass003": (12.8, 13.2),
        "mass004": (13.0, 13.4),
        "mass005": (13.2, 13.6),
        "mass006": (13.4, 13.8),
        "mass007": (13.6, 14.0),
        "mass008": (13.8, 14.2),
        "mass009": (14.0, 14.4),  # There is no upper limit.
        }

    # cached_funcs = ["get_overlap_summary", "read_dist", "make_kl", "make_ks"]
    cached_funcs = ["get_property_maxoverlap"]
    if args.clean:
        for func in cached_funcs:
            print(f"Cleaning cache for function {func}.")
            delete_disk_caches_for_function(func)

    if True:
        mass_vs_pairoverlap(7444, "csiborg", 13.0)

    if False:
        plot_mass_vs_maxpairoverlap(7444 + 24, 8956 + 24 * 3)

    if False:
        plot_mass_vsmedmaxoverlap(7444)

    if False:
        plot_summed_overlap_vs_mass(7444)

    if False:
        plot_mass_vs_separation(7444 + 24, 8956 + 24 * 3, min_overlap=0.0)

    if False:
        plot_maxoverlap_mass(7444)

    if False:
        plot_maxoverlapstat(7444, "lambda200c")

    if False:
        plot_maxoverlapstat(7444, "totpartmass")

    if False:
        plot_mass_vs_expected_mass(7444, max_prob_nomatch=1.0)

    # Plot 1NN distance distributions.
    if False:
        for i in range(1, 10):
            run = f"mass00{i}"
            for pulled_cdf in [True, False]:
                plot_dist(run, "cdf", neighbour_kwargs, runs_to_mass,
                          pulled_cdf=pulled_cdf,)
            plot_dist(run, "pdf", neighbour_kwargs, runs_to_mass)

    # Plot 1NN CDF differences.
    if False:
        runs = [f"mass00{i}" for i in range(1, 10)]
        for pulled_cdf in [True, False]:
            plot_cdf_diff(runs, neighbour_kwargs, pulled_cdf=pulled_cdf,
                          runs_to_mass=runs_to_mass)
    if False:
        runs = [f"mass00{i}" for i in range(1, 9)]
        for kind in ["kl", "ks"]:
            plot_significance("csiborg", runs, 7444, nobs=None, kind=kind,
                              kwargs=neighbour_kwargs,
                              runs_to_mass=runs_to_mass)

    if False:
        # runs = [[f"mass00{i}"] for i in range(1, 10)]
        runs = [[f"mass00{i}"] for i in [4]]
        for runs_ in runs:
            # runs = ["mass007"]
            for kind in ["kl"]:
                plot_significance_vs_mass("csiborg", runs_, 7444, nobs=None,
                                          kind=kind, kwargs=neighbour_kwargs,
                                          runs_to_mass=runs_to_mass,
                                          upper_threshold=100)

    if False:
        # runs = [f"mass00{i}" for i in range(1, 10)]
        runs = ["mass004"]
        plot_kl_vs_ks("csiborg", runs, 7444, None, kwargs=neighbour_kwargs,
                      runs_to_mass=runs_to_mass, upper_threshold=100)

    if False:
        # runs = [f"mass00{i}" for i in range(1, 10)]
        runs = ["mass007"]
        plot_kl_vs_overlap(runs, 7444, neighbour_kwargs, runs_to_mass,
                           upper_threshold=100, plot_std=False)
