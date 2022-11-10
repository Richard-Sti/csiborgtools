# Copyright (C) 2022 Richard Stiskalek
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
Notebook utility functions.
"""


import numpy
from os.path import join
from astropy.cosmology import FlatLambdaCDM

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")


Nsplits = 200
dumpdir = "/mnt/extraspace/rstiskalek/csiborg/"


# Some chosen clusters
_coma = {"RA": (12 + 59/60 + 48.7 / 60**2) * 15,
         "DEC": 27 + 58 / 60 + 50 / 60**2,
         "COMDIST": 102.975}

_virgo = {"RA": (12 + 27 / 60) * 15,
          "DEC": 12 + 43/60,
          "COMDIST": 16.5}

specific_clusters = {"Coma": _coma, "Virgo": _virgo}


def load_processed(Nsim, Nsnap):
    simpath = csiborgtools.io.get_sim_path(Nsim)
    outfname = join(
        dumpdir, "ramses_out_{}_{}.npy".format(str(Nsim).zfill(5),
                                               str(Nsnap).zfill(5)))
    data = numpy.load(outfname)
    # Add mmain
    mmain = csiborgtools.io.read_mmain(Nsim, "/mnt/zfsusers/hdesmond/Mmain")
    data = csiborgtools.io.merge_mmain_to_clumps(data, mmain)
    csiborgtools.utils.flip_cols(data, "peak_x", "peak_z")
    # Cut on numbre of particles and finite m200
    data = data[(data["npart"] > 100) & numpy.isfinite(data["m200"])]

    # Do unit conversion
    boxunits = csiborgtools.units.BoxUnits(Nsnap, simpath)
    convert_cols = ["m200", "m500", "totpartmass", "mass_mmain",
                    "r200", "r500", "Rs", "rho0", "peak_x", "peak_y", "peak_z"]
    data = csiborgtools.units.convert_from_boxunits(
        data, convert_cols, boxunits)
    # Now calculate spherical coordinates
    d, ra, dec = csiborgtools.units.cartesian_to_radec(data)
    data = csiborgtools.utils.add_columns(
        data, [d, ra, dec], ["dist", "ra", "dec"])
    return data


def load_planck2015(max_comdist=214):
    cosmo = FlatLambdaCDM(H0=70.5, Om0=0.307, Tcmb0=2.728)
    fpath = ("/mnt/zfsusers/rstiskalek/csiborgtools/"
             + "data/HFI_PCCS_SZ-union_R2.08.fits")
    return csiborgtools.io.read_planck2015(fpath, cosmo, max_comdist)


def load_mcxc(max_comdist=214):
    cosmo = FlatLambdaCDM(H0=70.5, Om0=0.307, Tcmb0=2.728)
    fpath = ("/mnt/zfsusers/rstiskalek/csiborgtools/data/mcxc.fits")
    return csiborgtools.io.read_mcxc(fpath, cosmo, max_comdist)


def load_2mpp():
    cosmo = FlatLambdaCDM(H0=70.5, Om0=0.307, Tcmb0=2.728)
    return csiborgtools.io.read_2mpp("../data/2M++_galaxy_catalog.dat", cosmo)
