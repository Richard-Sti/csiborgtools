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

CB2_REDSHIFT = [69.0000210000063, 40.250007218751264, 28.24050991940438,
                21.6470609550175, 17.480001404480106, 14.608109099433955,
                12.508772664512199, 10.90721705951751, 9.64516173673259,
                8.625000360937513, 7.7832702592057235, 7.0769233254437935,
                6.475728365821477, 5.95783150553419, 5.50704240932355,
                5.111111246913583, 4.760598622974984, 4.448113312911626,
                4.1677853285437605, 3.914893700679041, 3.685598452365574,
                3.476744253718227, 3.285714346938776, 3.1103203402819117,
                2.9487179993425383, 2.7993421515051513, 2.6608558268213116,
                2.5321101306287352, 2.4121122957547967, 2.3000000330000008,
                2.1950207773798662, 2.096514773533915, 2.003901196522936,
                1.9166666909722223, 1.8343558508261513, 1.7565632668759008,
                1.6829268488994646, 1.613122190273029, 1.5468577900064306,
                1.4838709837669097, 1.4239244641145379, 1.366803292753544,
                1.3123123255056859, 1.2602739849878026, 1.210526327423823,
                1.162921359250726, 1.117323566656109, 1.0736086272735772,
                1.0316622782422846, 0.9913793189283591, 0.9526627299814432,
                0.9154228931957131, 0.8795768989699038, 0.8450479301016136,
                0.8117647122768166, 0.7796610229819017, 0.7486752517178681,
                0.7187500053710938, 0.6898317534223188, 0.6618705083794834,
                0.6348195374209455, 0.6086351017498701, 0.5832762206018658,
                0.5587044572276223, 0.5348837244997295, 0.5117801080759505,
                0.48936170529651424, 0.46759847820604516, 0.4464621192761633,
                0.42592592856652933, 0.4059647012034677, 0.3865546241790834,
                0.3676731815824261, 0.34929906746973005, 0.3314121056648591,
                0.31399317585528075, 0.2970241454144613, 0.28048780643961924,
                0.2643678175452504, 0.2486486499985392, 0.23331553782343795,
                0.21835443153641232, 0.20375195520916023, 0.18949536658248856,
                0.17557251998135315, 0.1619718318042056, 0.14868224838055033,
                0.13569321600925854, 0.122994653006949, 0.11057692361085425,
                0.09843081359419292, 0.08654750746436402, 0.0749185671253807,
                0.06353591189600438, 0.05239179978414388, 0.04147880992632613,
                0.03078982610853953, 0.020318021291547472,
                0.010056843069963017, 0.0]


def snap2redshift(snapnum, simname):
    """Convert a snapshot number to a redshift."""
    if "csiborg2_" in simname:
        return CB2_REDSHIFT[snapnum]
    else:
        raise ValueError(f"Unknown simname: {simname}")


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

    return {"Redshift": [snap2redshift(i, simname) for i in range(cat.nsnap)],
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

    work_delegation(main, nsims, MPI.COMM_WORLD)
