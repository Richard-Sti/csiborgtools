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
A script to calculate the projected density field for a given simulation as a
sky map. The script is not parallelized in any way. The generated fields are
converted to galactic coordinates to match the CMB maps.
"""
from argparse import ArgumentParser
from os import remove
from os.path import exists, join

import csiborgtools
import numpy as np
from h5py import File


def get_field(simname, nsim, field_kind, MAS, grid):
    """Get the appropriate field reader for the simulation."""
    if simname == "csiborg1":
        reader = csiborgtools.read.CSiBORG1Field(nsim)
    elif "csiborg2" in simname:
        kind = simname.split("_")[-1]
        reader = csiborgtools.read.CSiBORG2Field(nsim, kind)
    elif simname == "csiborg2X":
        reader = csiborgtools.read.CSiBORG2XField(nsim)
    elif "quijote" in simname:
        reader = csiborgtools.read.QuijoteField(nsim)
    else:
        raise ValueError(f"Unknown simname: `{simname}`.")

    if field_kind == "density":
        return reader.density_field(MAS, grid)
    else:
        raise ValueError(f"Unknown field kind: `{field_kind}`.")


def main(simname, nsims, field_kind, nside, dist_ranges, MAS, grid,
         volume_weight, folder, normalize_to_overdensity=True):
    boxsize = csiborgtools.simname2boxsize(simname)
    Om0 = csiborgtools.simname2Omega_m(simname)
    matter_density = Om0 * 277.53662724583074  # Msun / kpc^3

    fname = join(folder, f"{simname}_{field_kind}.hdf5")
    if volume_weight:
        fname = fname.replace(".hdf5", "_volume_weighted.hdf5")
    print(f"Writing to `{fname}`...")
    if exists(fname):
        remove(fname)

    with File(fname, "w") as f:
        f.create_dataset("dist_ranges", data=np.asarray(dist_ranges))
        f.create_dataset("nsims", data=nsims)

    # These are at first generated in RA/dec but we can assume it is galactic
    # and convert it to RA/dec.
    pixel_angpos = csiborgtools.field.nside2radec(nside)
    RA, dec = csiborgtools.galactic_to_radec(*pixel_angpos.T)
    pixel_angpos = np.vstack([RA, dec]).T
    npix = len(pixel_angpos)

    Rmax = np.asanyarray(dist_ranges).reshape(-1).max()
    dr = 0.5 * boxsize / grid
    print(f"{'R_max:':<20} {Rmax} Mpc / h", flush=True)
    print(f"{'dr:':<20} {dr} Mpc / h", flush=True)

    for nsim in nsims:
        print(f"Interpolating at {npix} pixel for simulation {nsim}...",
              flush=True)

        field = get_field(simname, nsim, field_kind, MAS, grid)
        rdist, finterp = csiborgtools.field.make_sky(
            field, pixel_angpos, Rmax, dr, boxsize, return_full=True,
            interpolation_method="linear")

        with File(fname, "a") as f:
            grp = f.create_group(f"nsim_{nsim}")

            for n in range(len(dist_ranges)):
                dmin, dmax = dist_ranges[n]
                k_start = np.searchsorted(rdist, dmin)
                k_end = np.searchsorted(rdist, dmax)

                r = rdist[k_start:k_end + 1]
                y = r**2 * finterp[:, k_start:k_end + 1]
                skymap = np.trapz(y, r, axis=-1) / np.trapz(r**2, r)

                if normalize_to_overdensity:
                    skymap /= matter_density
                    skymap -= 1

                    grp.create_dataset(f"dist_range_{n}", data=skymap)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--simname", type=str, help="Simulation name.")
    args = parser.parse_args()

    fdir = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/field_projected"
    dx = 25
    dist_ranges = [[0, n * dx] for n in range(1, 5)]
    dist_ranges += [[n * dx, (n + 1) * dx] for n in range(0, 5)]

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(args.simname)
    print(f"{'Num. sims:':<20} {len(nsims)}", flush=True)

    MAS = "SPH"
    grid = 1024
    nside = 128
    field_kind = "density"
    volume_weight = True

    main(args.simname, nsims, field_kind, nside, dist_ranges, MAS, grid,
         volume_weight, fdir)
