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
Script to iteratively load particles of a TNG simulation and construct the DM
density field.
"""
from glob import glob
from os.path import join

import MAS_library as MASL
import numpy as np
from h5py import File
from tqdm import trange


if __name__ == "__main__":
    # Some parameters
    basepath = "/mnt/extraspace/rstiskalek/TNG300-1"
    snap = str(99).zfill(3)
    grid = 1024
    boxsize = 205000.0  # kpc/h
    mpart = 0.00398342749867548 * 1e10  # Msun/h, DM particles mass
    MAS = "PCS"

    # Get the snapshot files
    files = glob(join(basepath, "output", f"snapdir_{snap}", f"snap_{snap}.*"))
    print(f"Found {len(files)} snapshot files.")

    # Iterate over the snapshot files and construct the density field
    rho = np.zeros((grid, grid, grid), dtype=np.float32)
    for i in trange(len(files), desc="Reading snapshot files"):
        with File(files[i], 'r') as f:
            pos = f["PartType1/Coordinates"][...].astype(np.float32)

        MASL.MA(pos, rho, boxsize, MAS, verbose=False)

    # Convert to units h^2 Msun / kpc^3
    rho *= mpart / (boxsize / grid)**3

    # Save to file
    fname = join(basepath, "postprocessing", "density_field",
                 f"rho_dm_{snap}_{grid}_{MAS}.npy")
    print(f"Saving to {fname}.", flush=True)
    np.save(fname, rho)
