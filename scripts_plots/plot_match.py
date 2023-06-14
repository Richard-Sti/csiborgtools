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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import scienceplots  # noqa
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function
from tqdm import tqdm

import plt_utils
from inv_pit import InversePIT

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


###############################################################################
#                           IC overlap plotting                               #
###############################################################################

def open_cat(nsim):
    """
    Open a CSiBORG halo catalogue.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    bounds = {"totpartmass": (1e12, None)}
    return csiborgtools.read.HaloCatalogue(nsim, paths, bounds=bounds)


@cache_to_disk(7)
def get_overlap(nsim0):
    """
    Calculate the summed overlap and probability of no match for a single
    reference simulation.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsimxs = csiborgtools.read.get_cross_sims(nsim0, paths, smoothed=True)
    cat0 = open_cat(nsim0)

    catxs = []
    print("Opening catalogues...", flush=True)
    for nsimx in tqdm(nsimxs):
        catxs.append(open_cat(nsimx))

    reader = csiborgtools.read.NPairsOverlap(cat0, catxs, paths)
    mass = reader.cat0("totpartmass")
    hindxs = reader.cat0("index")
    summed_overlap = reader.summed_overlap(True)
    prob_nomatch = reader.prob_nomatch(True)
    return mass, hindxs, summed_overlap, prob_nomatch


def plot_summed_overlap(nsim0):
    """
    Plot the summed overlap and probability of no matching for a single
    reference simulation as a function of the reference halo mass.
    """
    x, __, summed_overlap, prob_nomatch = get_overlap(nsim0)

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
    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        plt.hexbin(x, mean_overlap, mincnt=1, xscale="log", bins="log",
                   gridsize=50)
        plt.colorbar(label="Counts in bins")
        plt.xlabel(r"$M_{\rm tot} / M_\odot$")
        plt.ylabel(r"$\langle \mathcal{O}_{a}^{\mathcal{A} \mathcal{B}} \rangle_{\mathcal{B}}$")  # noqa
        plt.ylim(0., 1.)

        plt.tight_layout()
        for ext in ["png", "pdf"]:
            fout = join(plt_utils.fout, f"overlap_mean_{nsim0}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()

    # Std summed overlap
    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        plt.hexbin(x, std_overlap, mincnt=1, xscale="log", bins="log",
                   gridsize=50)
        plt.colorbar(label="Counts in bins")
        plt.xlabel(r"$M_{\rm tot} / M_\odot$")
        plt.ylabel(r"$\delta \left( \mathcal{O}_{a}^{\mathcal{A} \mathcal{B}} \right)_{\mathcal{B}}$")  # noqa
        plt.ylim(0., 1.)
        plt.tight_layout()

        for ext in ["png", "pdf"]:
            fout = join(plt_utils.fout, f"overlap_std_{nsim0}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()

    # 1 - mean summed overlap vs mean prob nomatch
    with plt.style.context(plt_utils.mplstyle):
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
            fout = join(plt_utils.fout,
                        f"overlap_vs_prob_nomatch_{nsim0}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


###############################################################################
#                        Nearest neighbour plotting                           #
###############################################################################


@cache_to_disk(7)
def read_dist(simname, run, kind, kwargs):
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)

    fpath = paths.cross_nearest(simname, run, "tot_counts", nsim=0, nobs=0)
    counts = numpy.load(fpath)["tot_counts"]
    return reader.build_dist(counts, kind)


@cache_to_disk(7)
def make_kl(simname, run, nsim, nobs, kwargs):
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)

    pdf = read_dist("quijote", run, "pdf", kwargs)
    return reader.kl_divergence(simname, run, nsim, pdf, nobs=nobs)


@cache_to_disk(7)
def make_ks(simname, run, nsim, nobs, kwargs):
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)

    cdf = read_dist("quijote", run, "cdf", kwargs)
    return reader.ks_significance(simname, run, nsim, cdf, nobs=nobs)


def pull_cdf(x, fid_cdf, test_cdf):
    """
    Pull a CDF so that it matches the fiducial CDF at 0.5.

    Parameters
    ----------
    x : 1-dimensional array
        The x-axis of the CDF.
    fid_cdf : 1-dimensional array
        The fiducial CDF.
    test_cdf : 1-dimensional array
        The test CDF to be pulled.

    Returns
    -------
    xnew : 1-dimensional array
        The new x-axis of the test CDF.
    test_cdf : 1-dimensional array
        The new test CDF.
    """
    xnew = x * numpy.interp(0.5, fid_cdf, x) / numpy.interp(0.5, test_cdf, x)
    return xnew, test_cdf


def plot_dist(run, kind, kwargs, pulled_cdf=False, r200=None):
    """
    Plot the PDF/CDF of the nearest neighbour distance for Quijote and CSiBORG.
    """
    assert kind in ["pdf", "cdf"]
    print(f"Plotting the {kind}.", flush=True)
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)
    x = reader.bin_centres("neighbour")
    if r200 is not None:
        x /= r200

    y_quijote = read_dist("quijote", run, kind, kwargs)
    y_csiborg = read_dist("csiborg", run, kind, kwargs)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        for i in range(y_csiborg.shape[0]):
            if i == 0:
                label1 = "Quijote"
                label2 = "CSiBORG"
            else:
                label1 = None
                label2 = None

            if not pulled_cdf:
                plt.plot(x, y_quijote[i], c="C0", label=label1)
                plt.plot(x, y_csiborg[i], c="C1", label=label2)
            else:
                plt.plot(*pull_cdf(x, y_quijote[0], y_quijote[i]), c="C0",
                         label=label1)
                plt.plot(*pull_cdf(x, y_csiborg[0], y_csiborg[i]), c="C1",
                         label=label2)

        if r200 is None:
            plt.xlabel(r"$r_{1\mathrm{NN}}~[\mathrm{Mpc}]$")
        else:
            plt.xlabel(r"$r_{1\mathrm{NN}} / R_{200c}$")
        if kind == "pdf":
            plt.ylabel(r"$p(r_{1\mathrm{NN}})$")
        else:
            plt.ylabel(r"$\mathrm{CDF}(r_{1\mathrm{NN}})$")
            xmax = numpy.min(x[numpy.isclose(y_quijote[-1, :], 1.)])
            if xmax > 0:
                plt.xlim(0, xmax)
            plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        for ext in ["png"]:
            if pulled_cdf:
                fout = join(plt_utils.fout, f"1nn_{kind}_{run}_pulled.{ext}")
            else:
                fout = join(plt_utils.fout, f"1nn_{kind}_{run}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def get_cdf_diff(x, y_csiborg, y_quijote):
    y_quijote = read_dist("quijote", run, "cdf", kwargs)
    y_csiborg = read_dist("csiborg", run, "cdf", kwargs)
    ncdf = y_csiborg.shape[0]
    for j in range(ncdf):
        if pulled_cdf:
            x1, y1 = pull_cdf(x, y_csiborg[0], y_csiborg[j])
            y1 = numpy.interp(x, x1, y1, left=0., right=1.)
            x2, y2 = pull_cdf(x, y_quijote[0], y_quijote[j])
            y2 = numpy.interp(x, x2, y2, left=0., right=1.)
        else:
            y1 = y_csiborg[j]
            y2 = y_quijote[j]


def plot_cdf_diff(run, kwargs, pulled_cdf=False,):
    """
    Plot the CDF difference between Quijote and CSiBORG.
    """
    print("Plotting the CDF difference.", flush=True)
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)
    x = reader.bin_centres("neighbour")

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()

        runs = ["mass001"]

        for i, run in enumerate(runs):
            y_quijote = read_dist("quijote", run, "cdf", kwargs)
            y_csiborg = read_dist("csiborg", run, "cdf", kwargs)
            ncdf = y_csiborg.shape[0]
            for j in range(ncdf):
                if pulled_cdf:
                    x1, y1 = pull_cdf(x, y_csiborg[0], y_csiborg[j])
                    y1 = numpy.interp(x, x1, y1, left=0., right=1.)
                    x2, y2 = pull_cdf(x, y_quijote[0], y_quijote[j])
                    y2 = numpy.interp(x, x2, y2, left=0., right=1.)
                else:
                    y1 = y_csiborg[j]
                    y2 = y_quijote[j]
                plt.plot(x, y1 - y2, c="C0")
        plt.xlim(0, 30)
        plt.xlabel(r"$r_{1\mathrm{NN}}~[\mathrm{Mpc}]$")
        plt.ylabel(r"$\Delta \mathrm{CDF}(r_{1\mathrm{NN}})$")
        plt.ylim(0)
        # plt.legend()
        plt.tight_layout()
        for ext in ["png"]:
            if pulled_cdf:
                fout = join(plt_utils.fout, f"1nn_diff_{run}_pulled.{ext}")
            else:
                fout = join(plt_utils.fout, f"1nn_diff_{run}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def plot_significance_hist(simname, run, nsim, nobs, kind, kwargs):
    """Plot a histogram of the significance of the 1NN distance."""
    assert kind in ["kl", "ks"]
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    if kind == "kl":
        x = make_kl(simname, run, nsim, nobs, kwargs)
    else:
        x = make_ks(simname, run, nsim, nobs, kwargs)
        x = numpy.log10(x)
    x = x[numpy.isfinite(x)]

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        plt.hist(x, bins="auto")

        if kind == "ks":
            plt.xlabel(r"$\log p$-value of $r_{1\mathrm{NN}}$ distribution")
        else:
            plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
        plt.ylabel(r"Counts")
        plt.tight_layout()

        for ext in ["png"]:
            if simname == "quijote":
                nsim = paths.quijote_fiducial_nsim(nsim, nobs)
            fout = join(plt_utils.fout, f"significance_{kind}_{simname}_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def plot_significance_mass(simname, run, nsim, nobs, kind, kwargs):
    """
    Plot significance of the 1NN distance as a function of the total mass.
    """
    assert kind in ["kl", "ks"]
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)

    x = reader.read_single(simname, run, nsim, nobs)["mass"]
    if kind == "kl":
        y = make_kl(simname, run, nsim, nobs, kwargs)
    else:
        y = make_ks(simname, run, nsim, nobs, kwargs)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        plt.scatter(x, y)

        plt.xscale("log")
        plt.xlabel(r"$M_{\rm tot} / M_\odot$")
        if kind == "ks":
            plt.ylabel(r"$p$-value of $r_{1\mathrm{NN}}$ distribution")
            plt.yscale("log")
        else:
            plt.ylabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")

        plt.tight_layout()
        for ext in ["png"]:
            if simname == "quijote":
                nsim = paths.quijote_fiducial_nsim(nsim, nobs)
            fout = join(plt_utils.fout, f"significance_vs_mass_{kind}_{simname}_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def plot_kl_vs_ks(simname, run, nsim, nobs, kwargs):
    """
    Plot Kullback-Leibler divergence vs Kolmogorov-Smirnov statistic p-value.
    """
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)

    x = reader.read_single(simname, run, nsim, nobs)["mass"]
    y_kl = make_kl(simname, run, nsim, nobs, kwargs)
    y_ks = make_ks(simname, run, nsim, nobs, kwargs)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        plt.scatter(y_kl, y_ks, c=numpy.log10(x))
        plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")

        plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
        plt.ylabel(r"$p$-value of $r_{1\mathrm{NN}}$ distribution")
        plt.yscale("log")

        plt.tight_layout()
        for ext in ["png"]:
            if simname == "quijote":
                nsim = paths.quijote_fiducial_nsim(nsim, nobs)
            fout = join(plt_utils.fout, f"kl_vs_ks{simname}_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def plot_kl_vs_overlap(run, nsim, kwargs):
    """
    Plot KL divergence vs overlap.
    """
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    nn_reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)
    nn_data = nn_reader.read_single("csiborg", run, nsim, nobs=None)
    nn_hindxs = nn_data["ref_hindxs"]

    mass, overlap_hindxs, summed_overlap, prob_nomatch = get_overlap(nsim)

    # We need to match the hindxs between the two.
    hind2overlap_array = {hind: i for i, hind in enumerate(overlap_hindxs)}
    mask = numpy.asanyarray([hind2overlap_array[hind] for hind in nn_hindxs])

    summed_overlap = summed_overlap[mask]
    prob_nomatch = prob_nomatch[mask]
    mass = mass[mask]

    kl = make_kl("csiborg", run, nsim, nobs=None, kwargs=kwargs)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        mu = numpy.mean(prob_nomatch, axis=1)
        plt.scatter(kl, 1 - mu, c=numpy.log10(mass))
        plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")
        plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
        plt.ylabel(r"$1 - \langle \eta^{\mathcal{B}}_a \rangle_{\mathcal{B}}$")

        plt.tight_layout()
        for ext in ["png"]:
            fout = join(plt_utils.fout, f"kl_vs_overlap_mean_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        std = numpy.std(prob_nomatch, axis=1)
        plt.scatter(kl, std, c=numpy.log10(mass))
        plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")
        plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
        plt.ylabel(r"$\langle \left(\eta^{\mathcal{B}}_a - \langle \eta^{\mathcal{B}^\prime}_a \rangle_{\mathcal{B}^\prime}\right)^2\rangle_{\mathcal{B}}^{1/2}$")  # noqa

        plt.tight_layout()
        for ext in ["png"]:
            fout = join(plt_utils.fout, f"kl_vs_overlap_std_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
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

    cached_funcs = ["get_overlap", "read_dist", "make_kl", "make_ks"]
    if args.clean:
        for func in cached_funcs:
            print(f"Cleaning cache for function {func}.")
            delete_disk_caches_for_function(func)

    # for i in range(1, 10):
    #     run = f"mass00{i}"
    #     plot_dist(run, "cdf", neighbour_kwargs)
    # plot_dist("mass005", "cdf", neighbour_kwargs, pulled_cdf=True, )
    plot_cdf_diff("mass009", neighbour_kwargs, pulled_cdf=True)

    # plot_significance_hist("csiborg", run, 7444, nobs=None, kind="ks",
    #                        kwargs=neighbour_kwargs)
    # plot_significance_mass("csiborg", run, 7444, nobs=None, kind="ks",
    #                        kwargs=neighbour_kwargs)

    # quit()

    # paths = csiborgtools.read.Paths(**neighbour_kwargs["paths_kind"])
    # nn_reader = csiborgtools.read.NearestNeighbourReader(**neighbour_kwargs,
    #                                                      paths=paths)

    # sizes = numpy.full(2700, numpy.nan)
    # from tqdm import trange
    # k = 0
    # for nsim in trange(100):
    #     for nobs in range(27):
    #         d = nn_reader.read_single("quijote", run, nsim, nobs)
    #         sizes[k] = d["mass"].size

    #         k += 1
    # print(sizes)
    # print(numpy.mean(sizes), numpy.std(sizes))

    # plot_kl_vs_overlap("mass003", 7444, neighbour_kwargs)

    # plot_cdf_r200("mass003", neighbour_kwargs)
