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
"""Quick script to write halo catalogues as a HDF5 file."""
import csiborgtools
from h5py import File

from tqdm import tqdm


if __name__ == "__main__":
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    simname = "csiborg1"

    nsims = paths.get_ics(simname)
    print(f"Number of simulations: {nsims}.")

    fname_out = f"/mnt/users/rstiskalek/csiborgtools/data/halos_{simname}.hdf5"

    print(f"Writing to `{fname_out}`.")

    with File(fname_out, 'w') as f:
        for nsim in tqdm(nsims, desc="Simulations"):
            grp = f.create_group(f"sim_{nsim}")
            cat = csiborgtools.read.CSiBORG1Catalogue(nsim, paths)

            grp["pos"] = cat["cartesian_pos"]
            grp["totmass"] = cat["totmass"]
