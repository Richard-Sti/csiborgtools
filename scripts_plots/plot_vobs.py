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
from tqdm import tqdm

import csiborgtools
import plt_utils


def observer_peculiar_velocity(MAS, grid, ext="png"):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics("csiborg")

    for n, nsim in enumerate(nsims):
        fpath = paths.observer_peculiar_velocity(MAS, grid, nsim)
        f = numpy.load(fpath)

        if n == 0:
            data = numpy.full((len(nsims), *f["observer_vp"].shape), numpy.nan)
            smooth_scales = f["smooth_scales"]

        data[n] = f["observer_vp"]

    for n, smooth_scale in enumerate(tqdm(smooth_scales,
                                          desc="Plotting smooth scales")):
        with plt.style.context(plt_utils.mplstyle):
            fig, axs = plt.subplots(ncols=3, figsize=(3.5 * 3, 2.625),
                                    sharey=True)
            fig.subplots_adjust(wspace=0)
            for i, ax in enumerate(axs):
                ax.hist(data[:, n, i], bins="auto", histtype="step")
                mu, sigma = numpy.mean(data[:, n, i]), numpy.std(data[:, n, i])
                ax.set_title(r"$V_{{\rm obs, i}} = {:.3f} \pm {:.3f} ~ \mathrm{{km}} / \mathrm{{s}}$".format(mu, sigma)) # noqa

            axs[0].set_xlabel(r"$V_{\rm obs, x} ~ [\mathrm{km} / \mathrm{s}]$")
            axs[1].set_xlabel(r"$V_{\rm obs, y} ~ [\mathrm{km} / \mathrm{s}]$")
            axs[2].set_xlabel(r"$V_{\rm obs, z} ~ [\mathrm{km} / \mathrm{s}]$")
            axs[0].set_ylabel(r"Counts")

            fig.suptitle(r"$N_{{\rm grid}} = {}$, $\sigma_{{\rm smooth}} = {:.2f} ~ [\mathrm{{Mpc}} / h]$".format(grid, smooth_scale)) # noqa

            fig.tight_layout(w_pad=0)
            fout = join(plt_utils.fout, f"observer_vp_{smooth_scale}.{ext}")
            fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
            plt.close()

    with plt.style.context(plt_utils.mplstyle):
        fig, axs = plt.subplots(ncols=3, figsize=(3.5 * 3, 2.625))
        for i, ax in enumerate(axs):

            ymu = numpy.mean(data[..., i], axis=0)
            ystd = numpy.std(data[..., i], axis=0)
            ylow, yupp = ymu - ystd, ymu + ystd
            ax.plot(smooth_scales, ymu, c="k")
            ax.fill_between(smooth_scales, ylow, yupp, color="k", alpha=0.2)

            ax.set_xlabel(r"$\sigma_{{\rm smooth}} ~ [\mathrm{{Mpc}} / h]$")

        axs[0].set_ylabel(r"$V_{\rm obs, x} ~ [\mathrm{km} / \mathrm{s}]$")
        axs[1].set_ylabel(r"$V_{\rm obs, y} ~ [\mathrm{km} / \mathrm{s}]$")
        axs[2].set_ylabel(r"$V_{\rm obs, z} ~ [\mathrm{km} / \mathrm{s}]$")
        fig.suptitle(r"$N_{{\rm grid}} = {}$".format(grid))

        fig.tight_layout(w_pad=0)
        fout = join(plt_utils.fout, f"observer_vp_summary.{ext}")
        fig.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    observer_peculiar_velocity("PCS", 512)
