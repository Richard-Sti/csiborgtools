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
from os.path import join
import numpy as np
from h5py import File
import illustris_python as il


if __name__ == "__main__":
    fdir = "/mnt/extraspace/rstiskalek/TNG300-1/"
    basepath = join(fdir, "output")
    out_fname = join(fdir, "postprocessing/subhalo_catalogue_099.hdf5")

    # SUBFIND catalogue
    fields = ["SubhaloFlag", "SubhaloPos", "SubhaloMassType",
              "SubhaloGasMetallicity", "SubhaloStarMetallicity",
              "SubhaloSFR", "SubhaloSpin", "SubhaloStellarPhotometrics"]

    print("Loading the data.....")
    data = il.groupcat.loadSubhalos(basepath, 99, fields=fields)
    print("Finished loading!")

    is_subhalo = data["SubhaloFlag"] == 1

    print(f"Writing the subfind dataset to '{out_fname}'")
    with File(out_fname, 'w') as f:
        for key in fields:
            if key == "SubhaloFlag":
                continue

            f.create_dataset(key, data=data[key][is_subhalo])

    # HIH2 supplemetary catalogue
    print("Loading the HI & H2 supplementary catalogue.")
    fname = join(fdir, "postprocessing/hih2/hih2_galaxy_099.hdf5")
    with File(fname, "r") as f:
        _m_neutral_H = f["m_neutral_H"][:]
        _id_subhalo = np.array(f["id_subhalo"][:], dtype=int)

    m_neutral_H = np.full(data["count"], np.nan, dtype=float)
    for i, j in enumerate(_id_subhalo):
        m_neutral_H[j] = _m_neutral_H[i]

    print("Adding the HI & H2 supplementary catalogue.")
    with File(out_fname, 'r+') as f:
        f.create_dataset("m_neutral_H", data=m_neutral_H[is_subhalo])
