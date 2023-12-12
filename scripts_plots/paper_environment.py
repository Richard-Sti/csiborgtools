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
"""
Plots for paper where we correlate CSiBORG1 with properties of the observed
galaxies.
"""
from argparse import ArgumentParser
from os.path import join

import csiborgtools
import healpy
import matplotlib.pyplot as plt
import numpy
import scienceplots  # noqa
from cache_to_disk import cache_to_disk, delete_disk_caches_for_function

import plt_utils

###############################################################################
#                             Sky distribution                                #
###############################################################################


@cache_to_disk(30)
def _plot_sky_projected_density(nsim, simname, grid, nside, MAS="PCS",
                                dmin=0, dmax=220, volume_weight=True):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsnap = max(paths.get_snapshots(nsim, simname))
    # Some work here to get the box
    box = csiborgtools.read.CSiBORG1Box(nsnap, nsim, paths)

    field = numpy.load(paths.field("density", MAS, grid, nsim, in_rsp=False))

    field /= numpy.mean(field)
    field += 1

    angpos = csiborgtools.field.nside2radec(nside)
    dist = numpy.linspace(dmin, dmax, 1000)
    return csiborgtools.field.make_sky(field, angpos=angpos, dist=dist,
                                       box=box, volume_weight=volume_weight)


def plot_sky_projected_density(nsim, simname, grid, nside, MAS="PCS",
                               dmin=0, dmax=220, volume_weight=True,
                               pdf=False):
    r"""
    CSiBORG1
    Plot the sky distribution of a given field kind on the sky along with halos
    and selected observations.
    """

    dmap = _plot_sky_projected_density(nsim, simname, grid, nside, MAS,
                                       dmin, dmax, volume_weight)

    with plt.style.context(plt_utils.mplstyle):
        healpy.mollview(numpy.log10(dmap), fig=0, title="", unit="")

        # if plot_groups:
        #     groups = csiborgtools.read.TwoMPPGroups(fpath="/mnt/extraspace/rstiskalek/catalogs/2M++_group_catalog.dat")  # noqa
        #     healpy.projscatter(numpy.deg2rad(groups["DEC"] + 90),
        #                        numpy.deg2rad(groups["RA"]), s=1, c="blue",
        #                        label="2M++ groups")

        # if plot_halos is not None or plot_groups:
        #     plt.legend(markerscale=5)

        for ext in ["png"] if pdf is False else ["png", "pdf"]:
            fout = join(plt_utils.fout, f"sky_density_{simname}_{nsim}_from_{dmin}_to_{dmax}_vol{volume_weight}.{ext}")  # noqa
            print(f"Saving to `{fout}`.")
            plt.savefig(fout, dpi=plt_utils.dpi, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--clean', action='store_true')
    args = parser.parse_args()

    cached_funcs = ["_plot_sky_projected_density"]
    if args.clean:
        for func in cached_funcs:
            print(f"Cleaning cache for function {func}.")
            delete_disk_caches_for_function(func)

    if True:
        plot_sky_projected_density(7444, "csiborg", 512, 32, "PCS", dmin=10,
                                   dmax=30, volume_weight=False)

