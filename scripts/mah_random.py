# Copyright (C) 2024 Richard Stiskalek
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
Script to extract the mass accretion histories in random simulations. Follows
the main progenitor of FoF haloes.
"""
from argparse import ArgumentParser
import csiborgtools
import numpy as np
from h5py import File
from mpi4py import MPI
from taskmaster import work_delegation  # noqa
from tqdm import trange
from cache_to_disk import cache_to_disk


@cache_to_disk(30)
def load_data(nsim, simname, min_logmass):
    """Load the data for a given simulation."""
    bnd = {"totmass": (10**min_logmass, None), "dist": (None, 135)}
    if "csiborg2_" in simname:
        kind = simname.split("_")[-1]
        cat = csiborgtools.read.CSiBORG2Catalogue(nsim, 99, kind, bounds=bnd)
        merger_reader = csiborgtools.read.CSiBORG2MergerTreeReader(nsim, kind)
    else:
        raise ValueError(f"Unknown simname: {simname}")

    return cat, merger_reader


def main_progenitor_mah(cat, merger_reader, simname, verbose=True):
    """Follow the main progenitor of each `z = 0` FoF halo."""
    indxs = cat["index"]

    # Main progenitor information as a function of time
    shape = (len(cat), cat.nsnap + 1)
    main_progenitor_mass = np.full(shape, np.nan, dtype=np.float32)
    group_mass = np.full(shape, np.nan, dtype=np.float32)

    for i in trange(len(cat), disable=not verbose, desc="Haloes"):
        d = merger_reader.main_progenitor(indxs[i])

        main_progenitor_mass[i, d["SnapNum"]] = d["MainProgenitorMass"]
        group_mass[i, d["SnapNum"]] = d["Group_M_Crit200"]

    return {"Redshift": [csiborgtools.snap2redshift(i, simname) for i in range(cat.nsnap + 1)],  # noqa
            "MainProgenitorMass": main_progenitor_mass,
            "GroupMass": group_mass,
            "FinalGroupMass": cat["totmass"],
            }


def save_output(data, nsim, simname, verbose=True):
    """Save the output to a file."""
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)

    fname = paths.random_mah(simname, nsim)
    if verbose:
        print(f"Saving output to `{fname}`")

    with File(fname, "w") as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)


if "__main__" == __name__:
    parser = ArgumentParser(description="Extract the mass accretion history in random simulations.")  # noqa
    parser.add_argument("--simname", help="Name of the simulation.", type=str,
                        choices=["csiborg2_random"])
    parser.add_argument("--min_logmass", type=float,
                        help="Minimum log mass of the haloes.")
    args = parser.parse_args()
    COMM = MPI.COMM_WORLD

    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(args.simname)

    def main(nsim):
        verbose = COMM.Get_size() == 1
        cat, merger_reader = load_data(nsim, args.simname, args.min_logmass)
        data = main_progenitor_mah(cat, merger_reader, args.simname,
                                   verbose=verbose)
        save_output(data, nsim, args.simname, verbose=verbose)

    work_delegation(main, list(nsims), MPI.COMM_WORLD)
