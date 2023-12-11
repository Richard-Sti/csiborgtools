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

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
from h5py import File
import healpy

import scienceplots  # noqa
import plt_utils
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
    Open a CSiBORG halo catalogue. Applies mass and distance selection.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    bounds = {"totpartmass": (None, None), "dist": (0, 155)}
    return csiborgtools.read.CSiBORGHaloCatalogue(
        nsim, paths, bounds=bounds, load_fitted=True, load_initial=True,
        with_lagpatch=False)


def open_quijote(nsim, nobs=None):
    """
    Open a Quijote halo catalogue. Applies mass and distance selection.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    cat = csiborgtools.read.QuijoteHaloCatalogue(
        nsim, paths, nsnap=4, load_fitted=True, load_initial=True,
        with_lagpatch=False)
    if nobs is not None:
        cat = cat.pick_fiducial_observer(nobs, rmax=155.5)
    return cat


def plot_mass_vs_ncells(nsim, pdf=False):
    """
    Plot the halo mass vs. number of occupied cells in the initial snapshot.
    """
    cat = open_csiborg(nsim)
    mpart = 4.38304044e+09

    x = numpy.log10(cat["totpartmass"])
    y = numpy.log10(cat["lagpatch_ncells"])

    p = numpy.polyfit(x, y, 1)
    print("Fitted parameters are: ", p)

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        plt.scatter(cat["totpartmass"], cat["lagpatch_ncells"], s=0.25,
                    rasterized=True)
        plt.xscale("log")
        plt.yscale("log")
        for n in [1, 10, 100]:
            plt.axvline(n * 512 * mpart, c="black", ls="--", zorder=0, lw=0.8)
        plt.xlabel(r"$M_{\rm tot} ~ [M_\odot$ / h]")
        plt.ylabel(r"$N_{\rm cells}$")

        for ext in ["png"] if pdf is False else ["png", "pdf"]:
            fout = join(plt_utils.fout, f"init_mass_vs_ncells_{nsim}.{ext}")
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


###############################################################################
#                              HMF plot                                       #
###############################################################################


def plot_hmf(pdf=False):
    """
    Plot the FoF halo mass function of CSiBORG and Quijote.
    """
    print("Plotting the HMF...", flush=True)
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    csiborg_nsims = paths.get_ics("csiborg")
    print("Loading CSiBORG halo counts.", flush=True)
    for i, nsim in enumerate(tqdm(csiborg_nsims)):
        data = numpy.load(paths.halo_counts("csiborg", nsim))
        if i == 0:
            bins = data["bins"]
            csiborg_counts = numpy.full((len(csiborg_nsims), len(bins) - 1),
                                        numpy.nan, dtype=numpy.float32)
        csiborg_counts[i, :] = data["counts"]
    csiborg_counts /= numpy.diff(bins).reshape(1, -1)

    # csiborg5511 = numpy.load(paths.halo_counts("csiborg", 5511))["counts"]
    # csiborg5511 /= numpy.diff(data["bins"])

    print("Loading Quijote halo counts.", flush=True)
    quijote_nsims = paths.get_ics("quijote")
    for i, nsim in enumerate(tqdm(quijote_nsims)):
        data = numpy.load(paths.halo_counts("quijote", nsim))
        if i == 0:
            bins = data["bins"]
            nmax = data["counts"].shape[0]
            quijote_counts = numpy.full(
                (len(quijote_nsims) * nmax, len(bins) - 1), numpy.nan,
                dtype=numpy.float32)
        quijote_counts[i * nmax:(i + 1) * nmax, :] = data["counts"]
    quijote_counts /= numpy.diff(bins).reshape(1, -1)

    vol = 4 * numpy.pi / 3 * 155.5**3
    csiborg_counts /= vol
    quijote_counts /= vol
    # csiborg5511 /= vol

    x = 10**(0.5 * (bins[1:] + bins[:-1]))
    # Edit lower limits
    csiborg_counts[:, x < 10**13.1] = numpy.nan
    quijote_counts[:, x < 10**(13.1)] = numpy.nan
    # Edit upper limits
    csiborg_counts[:, x > 3e15] = numpy.nan
    quijote_counts[:, x > 3e15] = numpy.nan
    # csiborg5511[x > 3e15] = numpy.nan

    with plt.style.context(plt_utils.mplstyle):
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(nrows=1, sharex=True,
                               figsize=(3.5, 2.625))
        ax = [ax]
        # fig, ax = plt.subplots(nrows=2, sharex=True,
        #                        figsize=(3.5, 2.625 * 1.25),
        #                        gridspec_kw={"height_ratios": [1, 0.25]})
        # fig.subplots_adjust(hspace=0, wspace=0)

        # Upper panel data
        mean_csiborg = numpy.mean(csiborg_counts, axis=0)
        std_csiborg = numpy.std(csiborg_counts, axis=0)

        for i in range(len(csiborg_counts)):
            ax[0].plot(x, csiborg_counts[i, :], c="cornflowerblue", lw=0.5, zorder=0)

        ax[0].plot(x, mean_csiborg, label="CSiBORG", c="mediumblue", zorder=1)
        # ax[0].fill_between(x, mean_csiborg - std_csiborg,
        #                    mean_csiborg + std_csiborg,
        #                    alpha=0.5, color=cols[0])

        mean_quijote = numpy.mean(quijote_counts, axis=0)
        std_quijote = numpy.std(quijote_counts, axis=0)

        for i in range(len(quijote_counts)):
            ax[0].plot(x, quijote_counts[i, :], c="palegreen", lw=0.5, zorder=-1)



        ax[0].plot(x, mean_quijote, label="Quijote", c="darkgreen", zorder=1)
        # ax[0].fill_between(x, mean_quijote - std_quijote,
        #                    mean_quijote + std_quijote, alpha=0.5,
        #                    color=cols[1])

        # ax[0].plot(x, csiborg5511, label="CSiBORG 5511", c="k", ls="--")
        # std5511 = numpy.sqrt(csiborg5511)
        # ax[0].fill_between(x, csiborg5511 - std_csiborg, csiborg5511 + std5511,
        #                    alpha=0.2, color="k")

        # # Lower panel data
        # log_y = numpy.log10(mean_csiborg / mean_quijote)
        # err = numpy.sqrt((std_csiborg / mean_csiborg / numpy.log(10))**2
        #                  + (std_quijote / mean_quijote / numpy.log(10))**2)
        # ax[1].plot(x, 10**log_y, c=cols[0])
        # ax[1].fill_between(x, 10**(log_y - err), 10**(log_y + err), alpha=0.5,
        #                    color="k")

        # ax[1].plot(x, csiborg5511 / mean_quijote, c="k", ls="--")

        # Labels and accesories
        # ax[1].axhline(1, color="k", ls="--",
        #               lw=0.5 * plt.rcParams["lines.linewidth"], zorder=0)
        # ax[0].set_ylabel(r"$\frac{\mathrm{d}^2 N}{\mathrm{d} V \mathrm{d}\log M_{\rm tot}}~[\mathrm{dex}^{-1} (\mathrm{Mpc} / h)^{-3}]$",  # noqa
        #                  fontsize="small")
        m = numpy.isfinite(mean_quijote)
        ax[0].set_xlim(x[m].min(), x[m].max())
        ax[0].set_ylabel(r"$\mathrm{HMF}~[\mathrm{dex}^{-1} (\mathrm{Mpc} / h)^{-3}]$")
        ax[0].set_xlabel(r"$M_{\rm tot}~[M_\odot / h]$", fontsize="small")
        # ax[1].set_ylabel(r"$\mathrm{CSiBORG} / \mathrm{Quijote}$",
        #                  fontsize="small")

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        # ax[1].set_ylim(0.5, 1.5)
        # ax[1].set_yscale("log")
        ax[0].legend()

        fig.tight_layout(h_pad=0, w_pad=0)
        for ext in ["png"] if pdf is False else ["png", "pdf"]:
            fout = join(plt_utils.fout, f"hmf_comparison.{ext}")
            print(f"Saving to `{fout}`.")
            fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def plot_hmf_quijote_full(pdf=False):
    """
    Plot the FoF halo mass function of Quijote full run.

    Returns
    -------
    None
    """
    print("Plotting the HMF...", flush=True)
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    print("Loading Quijote halo counts.", flush=True)
    quijote_nsims = paths.get_ics("quijote")
    for i, nsim in enumerate(tqdm(quijote_nsims)):
        data = numpy.load(paths.halo_counts("quijote_full", nsim))
        if i == 0:
            bins = data["bins"]
            counts = numpy.full((len(quijote_nsims), len(bins) - 1), numpy.nan,
                                dtype=numpy.float32)
        counts[i, :] = data["counts"]
    counts /= numpy.diff(bins).reshape(1, -1)
    counts /= 1000**3

    x = 10**(0.5 * (bins[1:] + bins[:-1]))
    # Edit lower and upper limits
    counts[:, x < 10**(12.4)] = numpy.nan
    counts[:, x > 4e15] = numpy.nan

    with plt.style.context(plt_utils.mplstyle):
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(nrows=2, sharex=True,
                               figsize=(3.5, 2.625 * 1.25),
                               gridspec_kw={"height_ratios": [1, 0.65]})
        fig.subplots_adjust(hspace=0, wspace=0)

        # Upper panel data
        mean = numpy.mean(counts, axis=0)
        std = numpy.std(counts, axis=0)
        ax[0].plot(x, mean)
        ax[0].fill_between(x, mean - std, mean + std, alpha=0.5)
        # Lower panel data
        for i in range(counts.shape[0]):
            ax[1].plot(x, counts[i, :] / mean, c=cols[0])

        # Labels and accesories
        ax[1].axhline(1, color="k", ls="--",
                      lw=0.5 * plt.rcParams["lines.linewidth"], zorder=0)
        ax[0].set_ylabel(r"$\frac{\mathrm{d}^2 n}{\mathrm{d}\log M_{\rm tot} \mathrm{d} V}~[\mathrm{dex}^{-1} (\mathrm{Mpc / h})^{-3}]$",  # noqa
                         fontsize="small")
        ax[1].set_xlabel(r"$M_{\rm tot}~[$M_\odot / h]$", fontsize="small")
        ax[1].set_ylabel(r"$\mathrm{HMF} / \langle \mathrm{HMF} \rangle$",
                         fontsize="small")

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].legend()

        fig.tight_layout(h_pad=0, w_pad=0)
        for ext in ["png"] if pdf is False else ["png", "pdf"]:
            fout = join(plt_utils.fout, f"hmf_quijote_full.{ext}")
            print(f"Saving to `{fout}`.")
            fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def load_field(kind, nsim, grid, MAS, in_rsp=False, smooth_scale=None):
    r"""
    Load a single field.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    return numpy.load(paths.field(kind, MAS, grid, nsim, in_rsp=in_rsp,
                                  smooth_scale=smooth_scale))


###############################################################################
#                             Projected field                                 #
###############################################################################


def plot_projected_field(kind, nsim, grid, in_rsp, smooth_scale, MAS="PCS",
                         vel_component=0, highres_only=True, slice_find=None,
                         pdf=False):
    """
    Plot the mean projected field, however can also plot a single slice.
    """
    print(f"Plotting projected field `{kind}`. ", flush=True)
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, "csiborg"))
    box = csiborgtools.read.CSiBORG1Box(nsnap, nsim, paths)

    if kind == "overdensity":
        field = load_field("density", nsim, grid, MAS=MAS, in_rsp=in_rsp)
        density_gen = csiborgtools.field.DensityField(box, MAS)
        field = density_gen.overdensity_field(field) + 2

        field = numpy.log10(field)
    elif kind == "borg_density":
        field = File(paths.borg_mcmc(nsim), 'r')
        field = field["scalars"]["BORG_final_density"][...]
    else:
        field = load_field(kind, nsim, grid, MAS=MAS, in_rsp=in_rsp,
                           smooth_scale=smooth_scale)

    if kind == "velocity":
        field = field[vel_component, ...]
        field = box.box2vel(field)

    if highres_only:
        csiborgtools.field.fill_outside(field, numpy.nan, rmax=155.5,
                                        boxsize=677.7)
        start = round(field.shape[0] * 0.27)
        end = round(field.shape[0] * 0.73)
        field = field[start:end, start:end, start:end]

    if kind == "environment":
        cmap = mpl.colors.ListedColormap(
            ['red', 'lightcoral', 'limegreen', 'khaki'])
    else:
        cmap = "viridis"

    labels = [r"$y-z$", r"$x-z$", r"$x-y$"]
    with plt.style.context(plt_utils.mplstyle):
        fig, ax = plt.subplots(figsize=(3.5 * 2, 2.625), ncols=3, sharey=True,
                               sharex="col")
        fig.subplots_adjust(hspace=0, wspace=0)
        for i in range(3):
            if slice_find is None:
                img = numpy.nanmean(field, axis=i)
            else:
                ii = int(field.shape[i] * slice_find)
                img = numpy.take(field, ii, axis=i)

            if i == 0:
                vmin, vmax = numpy.nanpercentile(img, [1, 99])
                im = ax[i].imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
            else:
                ax[i].imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)

            k = img.shape[0] // 2
            ax[i].scatter(k, k, marker="x", s=5, zorder=2, c="red")

            frad = 155.5 / 677.7
            R = 155.5 / 677.7 * grid
            if slice_find is None:
                rad = R
                plot_circle = True
            elif (not highres_only and 0.5 - frad < slice_find < 0.5 + frad):
                z = (slice_find - 0.5) * grid
                rad = R * numpy.sqrt(1 - z**2 / R**2)
                plot_circle = True
            else:
                plot_circle = False

            if not highres_only and plot_circle:
                theta = numpy.linspace(0, 2 * numpy.pi, 100)
                ax[i].plot(rad * numpy.cos(theta) + grid // 2,
                           rad * numpy.sin(theta) + grid // 2,
                           lw=0.75 * plt.rcParams["lines.linewidth"], zorder=1,
                           c="red", ls="--")

            ax[i].set_title(labels[i])

        if highres_only:
            ncells = end - start
            size = ncells / grid * 677.7
        else:
            ncells = grid
            size = 677.7

        # Get beautiful ticks
        yticks = numpy.linspace(0, ncells, 6).astype(int)
        yticks = numpy.append(yticks, ncells // 2)
        ax[0].set_yticks(yticks)
        ax[0].set_yticklabels((yticks * size / ncells - size / 2).astype(int))
        ax[0].set_ylabel(r"$x_i ~ [\mathrm{Mpc} / h]$")

        for i in range(3):
            xticks = numpy.linspace(0, ncells, 6).astype(int)
            xticks = numpy.append(xticks, ncells // 2)
            xticks = numpy.sort(xticks)
            if i < 2:
                xticks = xticks[:-1]
            ax[i].set_xticks(xticks)
            ax[i].set_xticklabels(
                (xticks * size / ncells - size / 2).astype(int))
            ax[i].set_xlabel(r"$x_j ~ [\mathrm{Mpc} / h]$")

        cbar_ax = fig.add_axes([0.982, 0.155, 0.025, 0.75],
                               transform=ax[2].transAxes)
        if slice_find is None:
            clabel = "Mean projected field"
        else:
            clabel = "Sliced field"

        if kind == "environment":
            bounds = [0, 1, 2, 3, 4]
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(cmap=cmap, norm=norm), cax=cbar_ax,
                ticks=[0.5, 1.5, 2.5, 3.5])
            cbar.ax.set_yticklabels(["knot", "filament", "sheet", "void"],
                                    rotation=90, va="center")
        else:
            fig.colorbar(im, cax=cbar_ax, label=clabel)

        fig.tight_layout(h_pad=0, w_pad=0)
        for ext in ["png"] if pdf is False else ["png", "pdf"]:
            fout = join(
                plt_utils.fout,
                f"field_{kind}_{nsim}_rsp{in_rsp}_hres{highres_only}.{ext}")
            if smooth_scale is not None and smooth_scale > 0:
                smooth_scale = float(smooth_scale)
                fout = fout.replace(f".{ext}", f"_smooth{smooth_scale}.{ext}")
            print(f"Saving to `{fout}`.")
            fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


###############################################################################
#                        Command line interface                               #
###############################################################################


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()

    cached_funcs = ["load_field"]
    if args.clean:
        for func in cached_funcs:
            print(f"Cleaning cache for function {func}.")
            delete_disk_caches_for_function(func)

    if False:
        plot_mass_vs_ncells(7444, pdf=False)

    if True:
        plot_hmf(pdf=True)

    if False:
        plot_hmf_quijote_full(pdf=False)

    if False:
        kind = "overdensity"
        grid = 1024
        plot_sky_distribution(kind, 7444, grid, nside=64,
                              plot_groups=False, dmin=45, dmax=60,
                              plot_halos=5e13, volume_weight=True)

    if False:
        kind = "environment"
        grid = 512
        smooth_scale = 8.0
        # plot_projected_field("overdensity", 7444, grid, in_rsp=True,
        #                      highres_only=False)
        # nsims = [7444 + n * 24 for n in range(101)]
        nsim = 7444

        for in_rsp in [False]:
            plot_projected_field(kind, nsim, grid, in_rsp=in_rsp,
                                 smooth_scale=smooth_scale, slice_find=0.5,
                                 MAS="PCS", highres_only=True)

    if False:
        paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

        d = csiborgtools.read.read_h5(paths.particles(7444, "csiborg"))
        d = d["particles"]

        plt.figure()
        plt.hist(d[:100000, 4], bins="auto")

        plt.tight_layout()
        plt.savefig("../plots/velocity_distribution.png", dpi=450,
                    bbox_inches="tight")
