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
import utils

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


###############################################################################
#                Probability of matching a reference simulation halo          #
###############################################################################


def plot_knn(runname):
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    reader = csiborgtools.read.kNNCDFReader(paths)

    with plt.style.context(utils.mplstyle):
        plt.figure()

        # Quijote kNN
        rs, cdf, ndensity = reader.read("quijote", runname, kind="auto")
        pk = reader.prob_k(cdf)
        pk_poisson = reader.poisson_prob_k(rs, numpy.arange(pk.shape[1]),
                                           ndensity)

        for k in range(2):
            mu = numpy.mean(pk[:, k, :], axis=0)
            std = numpy.std(pk[:, k, :], axis=0)
            plt.plot(rs, mu, label=r"$k = {}$".format(k + 1),
                     c=cols[k % len(cols)])
            plt.fill_between(rs, mu - std, mu + std, alpha=0.15,
                             color=cols[k % len(cols)], zorder=0)

            mu = numpy.mean(pk_poisson[:, k, :], axis=0)
            std = numpy.std(pk_poisson[:, k, :], axis=0)
            plt.plot(rs, mu, c=cols[k % len(cols)], ls="dotted")
            plt.fill_between(rs, mu - std, mu + std, alpha=0.15,
                             color=cols[k % len(cols)], zorder=0, hatch="\\")

        # CSiBORG kNN
        rs, cdf, ndensity = reader.read("csiborg", runname, kind="auto")
        pk = reader.mean_prob_k(cdf)
        for k in range(2):
            mu = pk[k, :, 0]
            std = pk[k, :, 1]
            plt.plot(rs, mu, ls="--", c=cols[k % len(cols)])
            plt.fill_between(rs, mu - std, mu + std, alpha=0.15, hatch="\\",
                             color=cols[k % len(cols)], zorder=0)

        plt.legend()
        plt.xlabel(r"$r~[\mathrm{Mpc}]$")
        plt.ylabel(r"$P(k | V = 4 \pi r^3 / 3)$")

        for ext in ["png"]:
            fout = join(utils.fout, f"knn_{runname}.{ext}")
            print("Saving to `{fout}`.".format(fout=fout))
            plt.savefig(fout, dpi=utils.dpi, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":

    plot_knn("mass003")
