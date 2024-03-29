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
from scipy.stats import kendalltau
from tqdm import tqdm

import plt_utils

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


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


def read_dist(simname, run, kind, kwargs):
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.summary.NearestNeighbourReader(**kwargs, paths=paths)

    fpath = paths.cross_nearest(simname, run, "tot_counts", nsim=0, nobs=0)
    counts = numpy.load(fpath)["tot_counts"]
    return reader.build_dist(counts, kind)


def pull_cdf(x, fid_cdf, test_cdf):
    xnew = x * numpy.interp(0.5, fid_cdf, x) / numpy.interp(0.5, test_cdf, x)
    return xnew, test_cdf


def plot_dist(run, kind, kwargs, runs_to_mass, pulled_cdf=False, r200=None):
    r"""
    Plot the PDF or CDF of the nearest neighbour distance for CSiBORG and
    Quijote.

    Parameters
    ----------
    run : str
        Run name.
    kind : str
        Kind of distribution. Must be either `pdf` or `cdf`.
    kwargs : dict
        Nearest neighbour reader keyword arguments.
    runs_to_mass : dict
        Dictionary mapping run names to halo mass range.
    pulled_cdf : bool, optional
        Whether to pull the CDFs of CSiBORG and Quijote so that they match
        (individually) at 0.5. Default is `False`.
    r200 : float, optional
        Halo radial size :math:`R_{200}`. If set, the x-axis will be scaled by
        it.

    Returns
    -------
    None
    """
    assert kind in ["pdf", "cdf"]
    print(f"Plotting the {kind} for {run}...", flush=True)
    reader = csiborgtools.summary.NearestNeighbourReader(
        **kwargs, paths=csiborgtools.read.Paths(**kwargs["paths_kind"]))
    raddist = reader.bin_centres("radial")
    r = reader.bin_centres("neighbour")
    r = r / r200 if r200 is not None else r

    y_csiborg = read_dist("csiborg", run, kind, kwargs)
    y_quijote = read_dist("quijote", run, kind, kwargs)

    with plt.style.context(plt_utils.mplstyle):
        norm = mpl.colors.Normalize(vmin=numpy.min(raddist),
                                    vmax=numpy.max(raddist))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
        cmap.set_array([])

        fig, ax = plt.subplots()
        if run != "mass009":
            ax.set_title(r"${} \leq \log M_{{\rm tot}} / (M_\odot  h) < {}$"
                         .format(*runs_to_mass[run]), fontsize="small")
        else:
            ax.set_title(r"$\log M_{{\rm tot}} / (M_\odot h) \geq {}$"
                         .format(runs_to_mass[run][0]), fontsize="small")
        # Plot data
        nrad = y_csiborg.shape[0]
        for i in range(nrad):
            if pulled_cdf:
                x1, y1 = pull_cdf(r, y_csiborg[0], y_csiborg[i])
                x2, y2 = pull_cdf(r, y_quijote[0], y_quijote[i])
            else:
                x1, y1 = r, y_csiborg[i]
                x2, y2 = r, y_quijote[i]

            ax.plot(x1, y1, c=cmap.to_rgba(raddist[i]),
                    label="CSiBORG" if i == 0 else None)
            ax.plot(x2, y2, c="gray", ls="--",
                    label="Quijote" if i == 0 else None)

        fig.colorbar(cmap, ax=ax, label=r"$R_{\rm dist}~[\mathrm{Mpc} / h]$")
        ax.grid(alpha=0.5, lw=0.4)
        # Plot labels
        if pulled_cdf:
            if r200 is None:
                ax.set_xlabel(r"$\tilde{r}_{1\mathrm{NN}}~[\mathrm{Mpc} / h]$")
                if kind == "pdf":
                    ax.set_ylabel(r"$p(\tilde{r}_{1\mathrm{NN}})$")
                else:
                    ax.set_ylabel(r"$\mathrm{CDF}(\tilde{r}_{1\mathrm{NN}})$")
            else:
                ax.set_xlabel(r"$\tilde{r}_{1\mathrm{NN}} / R_{200c}$")
                if kind == "pdf":
                    ax.set_ylabel(r"$p(\tilde{r}_{1\mathrm{NN}} / R_{200c})$")
                else:
                    ax.set_ylabel(r"$\mathrm{CDF}(\tilde{r}_{1\mathrm{NN}} / R_{200c})$")  # noqa
        else:
            if r200 is None:
                ax.set_xlabel(r"$r_{1\mathrm{NN}}~[\mathrm{Mpc} / h]$")
                if kind == "pdf":
                    ax.set_ylabel(r"$p(r_{1\mathrm{NN}})$")
                else:
                    ax.set_ylabel(r"$\mathrm{CDF}(r_{1\mathrm{NN}})$")
            else:
                ax.set_xlabel(r"$r_{1\mathrm{NN}} / R_{200c}$")
                if kind == "pdf":
                    ax.set_ylabel(r"$p(r_{1\mathrm{NN}} / R_{200c})$")
                else:
                    ax.set_ylabel(r"$\mathrm{CDF}(r_{1\mathrm{NN}} / R_{200c})$")  # noqa

        if kind == "cdf":
            xmax = numpy.min(r[numpy.isclose(y_quijote[-1, :], 1.)])
            if xmax > 0:
                ax.set_xlim(0, xmax)
            ax.set_ylim(0, 1)

        ax.legend(fontsize="small")
        fig.tight_layout()
        for ext in ["png"]:
            if pulled_cdf:
                fout = join(plt_utils.fout, f"1nn_{kind}_{run}_pulled.{ext}")
            else:
                fout = join(plt_utils.fout, f"1nn_{kind}_{run}.{ext}")
            print(f"Saving to `{fout}`.")
            fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def get_cdf_diff(x, y_csiborg, y_quijote, pulled_cdf):
    """
    Get difference between the two CDFs as a function of radial distance.

    Parameters
    ----------
    x : 1-dimensional array
        The x-axis of the CDFs.
    y_csiborg : 2-dimensional array
        The CDFs of CSiBORG.
    y_quijote : 2-dimensional array
        The CDFs of Quijote.
    pulled_cdf : bool
        Whether to pull the CDFs of CSiBORG and Quijote.

    Returns
    -------
    dy : 2-dimensional array
        The difference between the two CDFs.
    """
    dy = numpy.full_like(y_csiborg, numpy.nan)
    for i in range(y_csiborg.shape[0]):
        if pulled_cdf:
            x1, y1 = pull_cdf(x, y_csiborg[0], y_csiborg[i])
            y1 = numpy.interp(x, x1, y1, left=0., right=1.)
            x2, y2 = pull_cdf(x, y_quijote[0], y_quijote[i])
            y2 = numpy.interp(x, x2, y2, left=0., right=1.)
            dy[i] = y1 - y2
        else:
            dy[i] = y_csiborg[i] - y_quijote[i]
    return dy


def plot_cdf_diff(runs, kwargs, pulled_cdf, runs_to_mass):
    """
    Plot the CDF difference between Quijote and CSiBORG.

    Parameters
    ----------
    runs : list of str
        Run names.
    kwargs : dict
        Nearest neighbour reader keyword arguments.
    pulled_cdf : bool
        Whether to pull the CDFs of CSiBORG and Quijote.
    runs_to_mass : dict
        Dictionary mapping run names to halo mass range.

    Returns
    -------
    None
    """
    print("Plotting the CDF difference...", flush=True)
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.summary.NearestNeighbourReader(**kwargs, paths=paths)
    r = reader.bin_centres("neighbour")
    runs_to_mass = [numpy.mean(runs_to_mass[run]) for run in runs]

    with plt.style.context(plt_utils.mplstyle):
        norm = mpl.colors.Normalize(vmin=min(runs_to_mass),
                                    vmax=max(runs_to_mass))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
        cmap.set_array([])

        fig, ax = plt.subplots()
        for i, run in enumerate(runs):
            y_quijote = read_dist("quijote", run, "cdf", kwargs)
            y_csiborg = read_dist("csiborg", run, "cdf", kwargs)

            dy = get_cdf_diff(r, y_csiborg, y_quijote, pulled_cdf)
            ax.plot(r, numpy.median(dy, axis=0),
                    c=cmap.to_rgba(runs_to_mass[i]))
            ax.fill_between(r, *numpy.percentile(dy, [16, 84], axis=0),
                            alpha=0.5, color=cmap.to_rgba(runs_to_mass[i]))
        fig.colorbar(cmap, ax=ax, ticks=runs_to_mass,
                     label=r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        ax.set_xlim(0.0, 55)
        ax.set_ylim(0)

        ax.grid(alpha=1/3, lw=0.4)

        # Plot labels
        if pulled_cdf:
            ax.set_xlabel(r"$\tilde{r}_{1\mathrm{NN}}~[\mathrm{Mpc} / h]$")
        else:
            ax.set_xlabel(r"$r_{1\mathrm{NN}}~[\mathrm{Mpc} / h]$")
        ax.set_ylabel(r"$\Delta \mathrm{CDF}(r_{1\mathrm{NN}})$")

        # Plot labels
        if pulled_cdf:
            ax.set_xlabel(r"$\tilde{r}_{1\mathrm{NN}}~[\mathrm{Mpc} / h]$")
            ax.set_ylabel(r"$\Delta \mathrm{CDF}(\tilde{r}_{1\mathrm{NN}})$")
        else:
            ax.set_xlabel(r"$r_{1\mathrm{NN}}~[\mathrm{Mpc} / h]$")
            ax.set_ylabel(r"$\Delta \mathrm{CDF}(r_{1\mathrm{NN}})$")

        fig.tight_layout()
        for ext in ["png"]:
            if pulled_cdf:
                fout = join(plt_utils.fout, f"1nn_diff_pulled.{ext}")
            else:
                fout = join(plt_utils.fout, f"1nn_diff.{ext}")
            print(f"Saving to `{fout}`.")
            fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


@cache_to_disk(7)
def make_kl(simname, run, nsim, nobs, kwargs):
    """
    Calculate the KL divergence between the distribution of nearest neighbour
    distances of haloes in a reference simulation with respect to Quijote.

    Parameters
    ----------
    simname : str
        Simulation name. Must be either `csiborg` or `quijote`.
    run : str
        Run name.
    nsim : int
        Simulation index.
    nobs : int
        Fiducial Quijote observer index. For CSiBORG must be set to `None`.
    kwargs : dict
        Nearest neighbour reader keyword arguments.

    Returns
    -------
    kl : 1-dimensional array
        KL divergence of the distribution of nearest neighbour distances
        of each halo in the reference simulation.
    """
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.summary.NearestNeighbourReader(**kwargs, paths=paths)
    # This is the reference PDF. Must be Quijote!
    pdf = read_dist("quijote", run, "pdf", kwargs)
    return reader.kl_divergence(simname, run, nsim, pdf, nobs=nobs)


@cache_to_disk(7)
def make_ks(simname, run, nsim, nobs, kwargs):
    """
    Calculate the KS significance between the distribution of nearest neighbour
    distances of haloes in a reference simulation with respect to Quijote.

    Parameters
    ----------
    simname : str
        Simulation name. Must be either `csiborg` or `quijote`.
    run : str
        Run name.
    nsim : int
        Simulation index.
    nobs : int
        Fiducial Quijote observer index. For CSiBORG must be set to `None`.
    kwargs : dict
        Nearest neighbour reader keyword arguments.

    Returns
    -------
    ks : 1-dimensional array
        KS significance of the distribution of nearest neighbour distances of
        each halo in the reference simulation.
    """
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.summary.NearestNeighbourReader(**kwargs, paths=paths)
    # This is the reference CDF. Must be Quijote!
    cdf = read_dist("quijote", run, "cdf", kwargs)
    return reader.ks_significance(simname, run, nsim, cdf, nobs=nobs)


def get_cumulative_significance(simname, runs, nsim, nobs, kind, kwargs):
    """
    Calculate the cumulative significance of the distribution of nearest
    neighbours and evaluate it at the same points for all runs.

    Parameters
    ----------
    simname : str
        Simulation name. Must be either `csiborg` or `quijote`.
    runs : list of str
        Run names.
    nsim : int
        Simulation index.
    nobs : int
        Fiducial Quijote observer index. For CSiBORG must be set to `None`.
    kind : str
        Must be either `kl` (Kullback-Leibler diverge) or `ks`
        (Kolmogorov-Smirnov p-value).
    kwargs : dict
        Nearest neighbour reader keyword arguments.

    Returns
    -------
    z : 1-dimensional array
        Points where the cumulative significance is evaluated.
    cumsum : 2-dimensional array of shape `(len(runs), len(z)))`
        Cumulative significance of the distribution of nearest neighbours.
    """
    significances = []
    for run in runs:
        if kind == "kl":
            x = make_kl(simname, run, nsim, nobs, kwargs)
        else:
            x = make_ks(simname, run, nsim, nobs, kwargs)
            x = numpy.log10(x)
        x = x[numpy.isfinite(x)]
        x = numpy.sort(x)
        significances.append(x)
    z = numpy.hstack(significances).reshape(-1, )

    if kind == "ks":
        zmin, zmax = numpy.percentile(z, [1, 100])
    else:
        zmin, zmax = numpy.percentile(z, [0.0, 99.9])
    z = numpy.linspace(zmin, zmax, 1000, dtype=numpy.float32)

    cumsum = numpy.full((len(runs), z.size), numpy.nan, dtype=numpy.float32)
    for i, run in enumerate(runs):
        x = significances[i]
        y = numpy.linspace(0, 1, x.size)
        cumsum[i, :] = numpy.interp(z, x, y, left=0, right=1)

    return z, cumsum


def plot_significance(simname, runs, nsim, nobs, kind, kwargs, runs_to_mass):
    """
    Plot cumulative significance of the 1NN distribution.

    Parameters
    ----------
    simname : str
        Simulation name. Must be either `csiborg` or `quijote`.
    runs : list of str
        Run names.
    nsim : int
        Simulation index.
    nobs : int
        Fiducial Quijote observer index. For CSiBORG must be set to `None`.
    kind : str
        Must be either `kl` (Kullback-Leibler diverge) or `ks`
        (Kolmogorov-Smirnov p-value).
    kwargs : dict
        Nearest neighbour reader keyword arguments.
    runs_to_mass : dict
        Dictionary mapping run names to total halo mass range.
    upper_threshold : bool, optional

    Returns
    -------
    None
    """
    assert kind in ["kl", "ks"]
    runs_to_mass = [numpy.mean(runs_to_mass[run]) for run in runs]

    with plt.style.context(plt_utils.mplstyle):
        norm = mpl.colors.Normalize(vmin=min(runs_to_mass),
                                    vmax=max(runs_to_mass))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
        cmap.set_array([])

        fig, ax = plt.subplots(figsize=(3.5, 2.625 * 1.2), nrows=2,
                               sharex=True, height_ratios=[1, 0.5])
        fig.subplots_adjust(hspace=0, wspace=0)
        z, cumsum = get_cumulative_significance(simname, runs, nsim, nobs,
                                                kind, kwargs)

        for i in range(len(runs)):
            ax[0].plot(z, cumsum[i, :], color=cmap.to_rgba(runs_to_mass[i]))

            dy = cumsum[-1, :] - cumsum[i, :]
            if kind == "kl":
                dy *= -1
            ax[1].plot(z, dy, color=cmap.to_rgba(runs_to_mass[i]))

        cbar_ax = fig.add_axes([1.0, 0.125, 0.035, 0.85])
        fig.colorbar(cmap, cax=cbar_ax, ticks=runs_to_mass,
                     label=r"$\log M_{\rm tot} ~ [M_\odot / h]$")

        ax[0].set_xlim(z[0], z[-1])
        ax[0].set_ylim(1e-5, 1.)
        if kind == "ks":
            ax[1].set_xlabel(r"$\log p$-value of $r_{1\mathrm{NN}}$ distribution")  # noqa
        else:
            ax[1].set_xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")  # noqa
        ax[0].set_ylabel(r"Cumulative norm. counts")
        ax[1].set_ylabel(r"$\Delta f$")

        fig.tight_layout(h_pad=0, w_pad=0)
        for ext in ["png"]:
            if simname == "quijote":
                paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
                nsim = paths.quijote_fiducial_nsim(nsim, nobs)
            nsim = str(nsim).zfill(5)
            fout = join(
                plt_utils.fout,
                f"significance_{kind}_{simname}_{nsim}_{runs}.{ext}")
            print(f"Saving to `{fout}`.")
            fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def make_binlims(run, runs_to_mass, upper_threshold=None):
    """
    Make bin limits for the 1NN distance runs, corresponding to the first half
    of the log-mass bin.

    Parameters
    ----------
    run : str
        Run name.
    runs_to_mass : dict
        Dictionary mapping run names to total halo mass range.
    upper_threshold : float, optional
        Bin width in dex. If set to `None`, the bin width is taken from the
        `runs_to_mass` dictionary.

    Returns
    -------
    xmin, xmax : floats
    """
    xmin, xmax = runs_to_mass[run]
    if upper_threshold is not None:
        xmax = xmin + upper_threshold

    xmin, xmax = 10**xmin, 10**xmax
    if run == "mass009":
        xmax = numpy.infty
    return xmin, xmax


def plot_significance_vs_mass(simname, runs, nsim, nobs, kind, kwargs,
                              runs_to_mass, upper_threshold=False):
    """
    Plot significance of the 1NN distance as a function of the total mass.

    Parameters
    ----------
    simname : str
        Simulation name. Must be either `csiborg` or `quijote`.
    runs : list of str
        Run names.
    nsim : int
        Simulation index.
    nobs : int
        Fiducial Quijote observer index. For CSiBORG must be set to `None`.
    kind : str
        Must be either `kl` (Kullback-Leibler diverge) or `ks`
        (Kolmogorov-Smirnov p-value).
    kwargs : dict
        Nearest neighbour reader keyword arguments.
    runs_to_mass : dict
        Dictionary mapping run names to total halo mass range.
    upper_threshold : bool, optional
        Whether to enforce an upper threshold on halo mass.

    Returns
    -------
    None
    """
    print(f"Plotting {kind} significance vs mass.")
    assert kind in ["kl", "ks"]
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.summary.NearestNeighbourReader(**kwargs, paths=paths)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        xs, ys = [], []
        for run in runs:
            x = reader.read_single(simname, run, nsim, nobs)["mass"]
            if kind == "kl":
                y = make_kl(simname, run, nsim, nobs, kwargs)
            else:
                y = numpy.log10(make_ks(simname, run, nsim, nobs, kwargs))

            xmin, xmax = make_binlims(run, runs_to_mass, upper_threshold)

            mask = (x >= xmin) & (x < xmax)
            xs.append(numpy.log10(x[mask]))
            ys.append(y[mask])

        xs = numpy.concatenate(xs)
        ys = numpy.concatenate(ys)

        plt.hexbin(xs, ys, gridsize=75, mincnt=1, bins="log")
        mask = numpy.isfinite(xs) & numpy.isfinite(ys)
        corr = plt_utils.latex_float(*kendalltau(xs[mask], ys[mask]))
        plt.title(r"$\tau = {}, p = {}$".format(*corr), fontsize="small")

        plt.xlabel(r"$\log M_{\rm tot} ~ [M_\odot / h]$")
        if kind == "ks":
            plt.ylabel(r"$\log p$-value of $r_{1\mathrm{NN}}$ distribution")
            plt.ylim(top=0)
        else:
            plt.ylabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
            plt.ylim(bottom=0)
        plt.colorbar(label="Bin counts")

        plt.tight_layout()
        for ext in ["png"]:
            if simname == "quijote":
                nsim = paths.quijote_fiducial_nsim(nsim, nobs)
            nsim = str(nsim).zfill(5)
            fout = f"sgnf_vs_mass_{kind}_{simname}_{nsim}_{runs}.{ext}"
            if upper_threshold:
                fout = fout.replace(f".{ext}", f"_upper_threshold.{ext}")
            fout = join(plt_utils.fout, fout)
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def plot_kl_vs_ks(simname, runs, nsim, nobs, kwargs, runs_to_mass,
                  upper_threshold=False):
    """
    Plot Kullback-Leibler divergence vs Kolmogorov-Smirnov statistic p-value.

    Parameters
    ----------
    simname : str
        Simulation name. Must be either `csiborg` or `quijote`.
    runs : str
        Run names.
    nsim : int
        Simulation index.
    nobs : int
        Fiducial Quijote observer index. For CSiBORG must be set to `None`.
    kwargs : dict
        Nearest neighbour reader keyword arguments.
    runs_to_mass : dict
        Dictionary mapping run names to total halo mass range.
    upper_threshold : bool, optional
        Whether to enforce an upper threshold on halo mass.

    Returns
    -------
    None
    """
    paths = csiborgtools.read.Paths(**kwargs["paths_kind"])
    reader = csiborgtools.summary.NearestNeighbourReader(**kwargs, paths=paths)

    xs, ys, cs = [], [], []
    for run in runs:
        c = reader.read_single(simname, run, nsim, nobs)["mass"]
        x = make_kl(simname, run, nsim, nobs, kwargs)
        y = make_ks(simname, run, nsim, nobs, kwargs)

        cmin, cmax = make_binlims(run, runs_to_mass)
        mask = (c >= cmin) & (c < cmax if upper_threshold else True)
        xs.append(x[mask])
        ys.append(y[mask])
        cs.append(c[mask])

    xs = numpy.concatenate(xs)
    ys = numpy.log10(numpy.concatenate(ys))
    cs = numpy.log10(numpy.concatenate(cs))

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        plt.hexbin(xs, ys, C=cs, gridsize=50, mincnt=0,
                   reduce_C_function=numpy.median)
        mask = numpy.isfinite(xs) & numpy.isfinite(ys)
        corr = plt_utils.latex_float(*kendalltau(xs[mask], ys[mask]))
        plt.title(r"$\tau = {}, p = {}$".format(*corr), fontsize="small")
        plt.colorbar(label=r"$\log M_{\rm tot} / M_\odot$")

        plt.xlabel(r"$D_{\mathrm{KL}}$ of $r_{1\mathrm{NN}}$ distribution")
        plt.ylabel(r"$\log p$-value of $r_{1\mathrm{NN}}$ distribution")

        plt.tight_layout()
        for ext in ["png"]:
            if simname == "quijote":
                nsim = paths.quijote_fiducial_nsim(nsim, nobs)
            nsim = str(nsim).zfill(5)
            fout = join(
                plt_utils.fout,
                f"kl_vs_ks_{simname}_{run}_{nsim}_{runs}.{ext}")
            if upper_threshold:
                fout = fout.replace(f".{ext}", f"_upper_threshold.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
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
    nn_reader = csiborgtools.summary.NearestNeighbourReader(**kwargs, paths=paths)

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

    if False:
        plot_mass_vs_pairoverlap(7444 + 24, 8956 + 24 * 3)

    if False:
        plot_mass_vs_maxpairoverlap(7444 + 24, 8956 + 24 * 3)

    if False:
        plot_mass_vsmedmaxoverlap(7444)

    if False:
        plot_summed_overlap_vs_mass(7444)

    if True:
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
