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

    Parameters
    ----------
    nsim : int
        Simulation index.

    Returns
    -------
    cat : csiborgtools.read.CSiBORGHaloCatalogue
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    bounds = {"totpartmass": (None, None), "dist": (0, 155/0.705)}
    return csiborgtools.read.CSiBORGHaloCatalogue(nsim, paths, bounds=bounds)


def open_quijote(nsim, nobs=None):
    """
    Open a Quijote halo catalogue. Applies mass and distance selection.

    Parameters
    ----------
    nsim : int
        Simulation index.
    nobs : int, optional
        Fiducial observer index.

    Returns
    -------
    cat : csiborgtools.read.QuijoteHaloCatalogue
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    cat = csiborgtools.read.QuijoteHaloCatalogue(nsim, paths, nsnap=4)
    if nobs is not None:
        cat = cat.pick_fiducial_observer(nobs, rmax=155.5 / 0.705)
    return cat


def plot_mass_vs_ncells(nsim, pdf=False):
    """
    Plot the halo mass vs. number of occupied cells in the initial snapshot.

    Parameters
    ----------
    nsim : int
        Simulation index.
    pdf : bool, optional
        Whether to save the figure as a PDF file.

    Returns
    -------
    None
    """
    cat = open_csiborg(nsim)
    mpart = 4.38304044e+09

    with plt.style.context(plt_utils.mplstyle):
        plt.figure()
        plt.scatter(cat["totpartmass"], cat["lagpatch_ncells"], s=0.25,
                    rasterized=True)
        plt.xscale("log")
        plt.yscale("log")
        for n in [1, 10, 100]:
            plt.axvline(n * 512 * mpart, c="black", ls="--", zorder=0, lw=0.8)
        plt.xlabel(r"$M_{\rm tot} / M_\odot$")
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
    Plot the (ultimate paretn) halo mass function of CSiBORG and Quijote.

    Parameters
    ----------
    pdf : bool, optional
        Whether to save the figure as a PDF file.

    Returns
    -------
    None
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

    x = 10**(0.5 * (bins[1:] + bins[:-1]))
    # Edit lower limits
    csiborg_counts[:, x < 1e12] = numpy.nan
    quijote_counts[:, x < 10**(12.4)] = numpy.nan
    # Edit upper limits
    csiborg_counts[:, x > 4e15] = numpy.nan
    quijote_counts[:, x > 4e15] = numpy.nan

    with plt.style.context(plt_utils.mplstyle):
        cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        fig, ax = plt.subplots(nrows=2, sharex=True,
                               figsize=(3.5, 2.625 * 1.25),
                               gridspec_kw={"height_ratios": [1, 0.65]})
        fig.subplots_adjust(hspace=0, wspace=0)

        # Upper panel data
        mean_csiborg = numpy.mean(csiborg_counts, axis=0)
        std_csiborg = numpy.std(csiborg_counts, axis=0)
        ax[0].plot(x, mean_csiborg, label="CSiBORG")
        ax[0].fill_between(x, mean_csiborg - std_csiborg,
                           mean_csiborg + std_csiborg, alpha=0.5)

        mean_quijote = numpy.mean(quijote_counts, axis=0)
        std_quijote = numpy.std(quijote_counts, axis=0)
        ax[0].plot(x, mean_quijote, label="Quijote")
        ax[0].fill_between(x, mean_quijote - std_quijote,
                           mean_quijote + std_quijote, alpha=0.5)
        # Lower panel data
        log_y = numpy.log10(mean_csiborg / mean_quijote)
        err = numpy.sqrt((std_csiborg / mean_csiborg / numpy.log(10))**2
                         + (std_quijote / mean_quijote / numpy.log(10))**2)
        ax[1].plot(x, 10**log_y, c=cols[2])
        ax[1].fill_between(x, 10**(log_y - err), 10**(log_y + err), alpha=0.5,
                           color=cols[2])

        # Labels and accesories
        ax[1].axhline(1, color="k", ls=plt.rcParams["lines.linestyle"],
                      lw=0.5 * plt.rcParams["lines.linewidth"], zorder=0)
        ax[0].set_ylabel(r"$\frac{\mathrm{d} n}{\mathrm{d}\log M_{\rm h}}~\mathrm{dex}^{-1}$")  # noqa
        ax[1].set_xlabel(r"$M_{\rm h}$ [$M_\odot$]")
        ax[1].set_ylabel(r"$\mathrm{CSiBORG} / \mathrm{Quijote}$")

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[1].set_yscale("log")
        ax[0].legend()

        fig.tight_layout(h_pad=0, w_pad=0)
        for ext in ["png"] if pdf is False else ["png", "pdf"]:
            fout = join(plt_utils.fout, f"hmf_comparison.{ext}")
            print(f"Saving to `{fout}`.")
            fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


def load_field(kind, nsim, grid, MAS, in_rsp=False, smooth_scale=None):
    r"""
    Load a single field.

    Parameters
    ----------
    kind : str
        Field kind.
    nsim : int
        Simulation index.
    grid : int
        Grid size.
    MAS : str
        Mass assignment scheme.
    in_rsp : bool, optional
        Whether to load the field in redshift space.
    smooth_scale : float, optional
        Smoothing scale in :math:`\mathrm{Mpc} / h`.

    Returns
    -------
    field : n-dimensional array
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
    r"""
    Plot the mean projected field, however can also plot a single slice.

    Parameters
    ----------
    kind : str
        Field kind.
    nsim : int
        Simulation index.
    grid : int
        Grid size.
    in_rsp : bool
        Whether to load the field in redshift space.
    smooth_scale : float
        Smoothing scale in :math:`\mathrm{Mpc} / h`.
    MAS : str, optional
        Mass assignment scheme.
    vel_component : int, optional
        Which velocity field component to plot.
    highres_only : bool, optional
        Whether to only plot the high-resolution region.
    slice_find : float, optional
        Which slice to plot in fractional units (i.e. 1. is the last slice)
    pdf : bool, optional
        Whether to save the figure as a PDF.

    Returns
    -------
    None
    """
    print(f"Plotting projected field `{kind}`. ", flush=True)
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)

    if kind == "overdensity":
        field = load_field("density", nsim, grid, MAS=MAS, in_rsp=in_rsp,
                           smooth_scale=smooth_scale)
        density_gen = csiborgtools.field.DensityField(box, MAS)
        field = density_gen.overdensity_field(field) + 1
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
#                             Sky distribution                                #
###############################################################################


def get_sky_label(kind, volume_weight):
    """
    Get the sky label for a given field kind.

    Parameters
    ----------
    kind : str
        Field kind.
    volume_weight : bool
        Whether to volume weight the field.

    Returns
    -------
    label : str
    """
    if volume_weight:
        if kind == "density":
            label = r"$\log \int_{0}^{R} r^2 \rho(r, \mathrm{RA}, \mathrm{dec}) \mathrm{d} r$"  # noqa
        if kind == "overdensity":
            label = r"$\log \int_{0}^{R} r^2 \left[\delta(r, \mathrm{RA}, \mathrm{dec}) + 1\right] \mathrm{d} r$"  # noqa
        elif kind == "potential":
            label = r"$\int_{0}^{R} r^2 \phi(r, \mathrm{RA}, \mathrm{dec}) \mathrm{d} r$"  # noqa
        elif kind == "radvel":
            label = r"$\int_{0}^{R} r^2 v_r(r, \mathrm{RA}, \mathrm{dec}) \mathrm{d} r$"  # noqa
        else:
            label = None
    else:
        if kind == "density":
            label = r"$\log \int_{0}^{R} \rho(r, \mathrm{RA}, \mathrm{dec}) \mathrm{d} r$"  # noqa
        if kind == "overdensity":
            label = r"$\log \int_{0}^{R} \left[\delta(r, \mathrm{RA}, \mathrm{dec}) + 1\right] \mathrm{d} r$"  # noqa
        elif kind == "potential":
            label = r"$\int_{0}^{R} \phi(r, \mathrm{RA}, \mathrm{dec}) \mathrm{d} r$"  # noqa
        elif kind == "radvel":
            label = r"$\int_{0}^{R} v_r(r, \mathrm{RA}, \mathrm{dec}) \mathrm{d} r$"  # noqa
        else:
            label = None
    return label


def plot_sky_distribution(field, nsim, grid, nside, smooth_scale=None,
                          MAS="PCS", plot_groups=True, dmin=0, dmax=220,
                          plot_halos=None, volume_weight=True, pdf=False):
    r"""
    Plot the sky distribution of a given field kind on the sky along with halos
    and selected observations.

    TODO
    ----
    - Add distance for groups.

    Parameters
    ----------
    field : str
        Field kind.
    nsim : int
        Simulation index.
    grid : int
        Grid size.
    nside : int
        Healpix nside of the sky projection.
    smooth_scale : float
        Smoothing scale in :math:`\mathrm{Mpc} / h`.
    MAS : str, optional
        Mass assignment scheme.
    plot_groups : bool, optional
        Whether to plot the 2M++ groups.
    dmin : float, optional
        Minimum projection distance in :math:`\mathrm{Mpc}/h`.
    dmax : float, optional
        Maximum projection distance in :math:`\mathrm{Mpc}/h`.
    plot_halos : list, optional
        Minimum halo mass to plot in :math:`M_\odot`.
    volume_weight : bool, optional
        Whether to volume weight the field.
    pdf : bool, optional
        Whether to save the figure as a pdf.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim))
    box = csiborgtools.read.CSiBORGBox(nsnap, nsim, paths)

    if kind == "overdensity":
        field = load_field("density", nsim, grid, MAS=MAS, in_rsp=False,
                           smooth_scale=smooth_scale)
        density_gen = csiborgtools.field.DensityField(box, MAS)
        field = density_gen.overdensity_field(field) + 1
    else:
        field = load_field(kind, nsim, grid, MAS=MAS, in_rsp=False,
                           smooth_scale=smooth_scale)

    angpos = csiborgtools.field.nside2radec(nside)
    dist = numpy.linspace(dmin, dmax, 500)
    out = csiborgtools.field.make_sky(field, angpos=angpos, dist=dist, box=box,
                                      volume_weight=volume_weight)

    with plt.style.context(plt_utils.mplstyle):
        label = get_sky_label(kind, volume_weight)
        if kind in ["density", "overdensity"]:
            out = numpy.log10(out)
        healpy.mollview(out, fig=0, title="", unit=label, rot=90)

        if plot_halos is not None:
            bounds = {"dist": (dmin, dmax),
                      "totpartmass": (plot_halos, None)}
            cat = csiborgtools.read.CSiBORGHaloCatalogue(nsim, paths,
                                                         bounds=bounds)
            X = cat.position(cartesian=False)
            healpy.projscatter(numpy.deg2rad(X[:, 2] + 90),
                               numpy.deg2rad(X[:, 1]),
                               s=5, c="red", label="CSiBORG haloes")

        if plot_groups:
            groups = csiborgtools.read.TwoMPPGroups(fpath="/mnt/extraspace/rstiskalek/catalogs/2M++_group_catalog.dat")  # noqa
            healpy.projscatter(numpy.deg2rad(groups["DEC"] + 90),
                               numpy.deg2rad(groups["RA"]), s=1, c="blue",
                               label="2M++ groups")

        if plot_halos is not None or plot_groups:
            plt.legend(markerscale=5)

        for ext in ["png"] if pdf is False else ["png", "pdf"]:
            fout = join(plt_utils.fout, f"sky_{kind}_{nsim}_from_{dmin}_to_{dmax}_vol{volume_weight}.{ext}")  # noqa
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

    cached_funcs = ["load_field"]
    if args.clean:
        for func in cached_funcs:
            print(f"Cleaning cache for function {func}.")
            delete_disk_caches_for_function(func)

    if False:
        plot_mass_vs_ncells(7444, pdf=False)

    if False:
        plot_hmf(pdf=False)

    if False:
        kind = "overdensity"
        grid = 1024
        plot_sky_distribution(kind, 7444, grid, nside=64,
                              plot_groups=False, dmin=45, dmax=60,
                              plot_halos=5e13, volume_weight=True)

    if True:
        kind = "overdensity"
        grid = 256
        smooth_scale = 0
        # plot_projected_field("overdensity", 7444, grid, in_rsp=True,
        #                      highres_only=False)
        plot_projected_field(kind, 7444, grid, in_rsp=False,
                             smooth_scale=smooth_scale, slice_find=0.5,
                             MAS="PCS",
                             highres_only=True)

    if False:
        paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

        d = csiborgtools.read.read_h5(paths.particles(7444))["particles"]

        plt.figure()
        plt.hist(d[:100000, 4], bins="auto")

        plt.tight_layout()
        plt.savefig("../plots/velocity_distribution.png", dpi=450,
                    bbox_inches="tight")
