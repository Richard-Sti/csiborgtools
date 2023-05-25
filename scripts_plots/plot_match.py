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
from astropy import units
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
    with plt.style.context(utils.mplstyle):
        plt.figure()
        plt.hexbin(x, mean_overlap, mincnt=1, xscale="log", bins="log",
                   gridsize=50)
        plt.colorbar(label="Counts in bins")
        plt.xlabel(r"$M_{\rm tot} / M_\odot$")
        plt.ylabel(r"$\langle \mathcal{O}_{a}^{\mathcal{A} \mathcal{B}} \rangle_{\mathcal{B}}$")  # noqa
        plt.ylim(0., 1.)

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
        plt.ylim(0., 1.)
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


###############################################################################
#                        Nearest neighbour plotting                           #
###############################################################################


@cache_to_disk(7)
def read_dist(simname, run, kind, kwargs):
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)
    return reader.build_dist(simname, run, kind, verbose=True)


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


def plot_dist(run, kind, kwargs):
    """
    Plot the PDF/CDF of the nearest neighbour distance for Quijote and CSiBORG.
    """
    assert kind in ["pdf", "cdf"]
    print(f"Plotting the {kind}.", flush=True)
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)
    x = reader.bin_centres("neighbour")

    y_quijote = read_dist("quijote", run, kind, kwargs)
    y_csiborg = read_dist("csiborg", run, kind, kwargs)
    ncdf = y_csiborg.shape[0]

    with plt.style.context(utils.mplstyle):
        plt.figure()
        for i in range(ncdf):
            if i == 0:
                label1 = "Quijote"
                label2 = "CSiBORG"
            else:
                label1 = None
                label2 = None
            plt.plot(x, y_quijote[i], c="C0", label=label1)
            plt.plot(x, y_csiborg[i], c="C1", label=label2)
        plt.xlim(0, 75)
        plt.xlabel(r"$r_{1\mathrm{NN}}~[\mathrm{Mpc}]$")
        if kind == "pdf":
            plt.ylabel(r"$p(r_{1\mathrm{NN}})$")
        else:
            plt.ylabel(r"$\mathrm{CDF}(r_{1\mathrm{NN}})$")
            plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        for ext in ["png"]:
            fout = join(utils.fout, f"1nn_{kind}_{run}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


def plot_cdf_r200(run, kwargs):
    print("Plotting the matching probability.", flush=True)
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])

    box = csiborgtools.read.QuijoteBox(nsnap=4)
    # Get R200 in Mpc of a halo of mass 1e14 Msun
    Msun = 1.98847e30
    M200 = 1e14 * Msun * units.kg
    rhoc = box.cosmo.critical_density0
    R200 = (M200 / (4 * numpy.pi / 3 * 200 * rhoc))**(1. / 3)
    R200 = R200.to(units.Mpc)
    R200 = R200.value

    reader = csiborgtools.read.NearestNeighbourReader(**kwargs, paths=paths)
    x = reader.bin_centres("neighbour")
    y = read_dist("quijote", run, "cdf", kwargs)
    ncdf = y.shape[0]

    with plt.style.context(utils.mplstyle):
        plt.figure()
        for i in range(ncdf):
            plt.plot(x, y[i], c="C0")
        plt.yscale("log")
        plt.xlabel(r"$r_{1\mathrm{NN}} / R_{200}$")
        plt.ylabel(r"$\mathrm{CDF}(r_{1\mathrm{NN}})$")
        plt.xscale("log")
        plt.grid(alpha=0.25)

        plt.tight_layout()
        for ext in ["png"]:
            fout = join(utils.fout, f"1nn_r200_quijote_{run}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
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

    with plt.style.context(utils.mplstyle):
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
            fout = join(utils.fout, f"significance_{kind}_{simname}_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
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

    with plt.style.context(utils.mplstyle):
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
            fout = join(utils.fout, f"significance_vs_mass_{kind}_{simname}_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
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

    with plt.style.context(utils.mplstyle):
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
            fout = join(utils.fout, f"kl_vs_ks{simname}_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
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

    with plt.style.context(utils.mplstyle):
        plt.figure()
        mu = numpy.mean(prob_nomatch, axis=1)
        plt.scatter(kl, 1 - mu, c=numpy.log10(mass))
        plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")
        plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
        plt.ylabel(r"$1 - \langle \eta^{\mathcal{B}}_a \rangle_{\mathcal{B}}$")

        plt.tight_layout()
        for ext in ["png"]:
            fout = join(utils.fout, f"kl_vs_overlap_mean_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()

    with plt.style.context(utils.mplstyle):
        plt.figure()
        std = numpy.std(prob_nomatch, axis=1)
        plt.scatter(kl, std, c=numpy.log10(mass))
        plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")
        plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
        plt.ylabel(r"$\langle \left(\eta^{\mathcal{B}}_a - \langle \eta^{\mathcal{B}^\prime}_a \rangle_{\mathcal{B}^\prime}\right)^2\rangle_{\mathcal{B}}^{1/2}$")  # noqa

        plt.tight_layout()
        for ext in ["png"]:
            fout = join(utils.fout, f"kl_vs_overlap_std_{run}_{str(nsim).zfill(5)}.{ext}")  # noqa
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

    cached_funcs = ["get_overlap", "read_dist", "make_kl", "make_ks"]
    if args.clean:
        for func in cached_funcs:
            print(f"Cleaning cache for function {func}.")
            delete_disk_caches_for_function(func)

    neighbour_kwargs = {"rmax_radial": 155 / 0.705,
                        "nbins_radial": 20,
                        "rmax_neighbour": 100.,
                        "nbins_neighbour": 150,
                        "paths_kind": csiborgtools.paths_glamdring}
    run = "mass003"

    # paths = csiborgtools.read.Paths(**neighbour_kwargs["paths_kind"])
    # nn_reader = csiborgtools.read.NearestNeighbourReader(**neighbour_kwargs,
    #                                                      paths=paths)

    # plot_kl_vs_overlap("mass003", 7444, neighbour_kwargs)

    plot_cdf_r200("mass003", neighbour_kwargs)

