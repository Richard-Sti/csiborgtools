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
from tqdm import trange
from astropy.cosmology import FlatLambdaCDM

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")


Nsplits = 15
dumpdir = "/mnt/extraspace/rstiskalek/csiborg/"


def load_mmain_convert(n):
    srcdir = "/users/hdesmond/Mmain"
    arr = csiborgtools.io.read_mmain(n, srcdir)

    csiborgtools.utils.convert_mass_cols(arr, "mass_cl")
    csiborgtools.utils.convert_position_cols(
        arr, ["peak_x", "peak_y", "peak_z"])
    csiborgtools.utils.flip_cols(arr, "peak_x", "peak_z")

    d, ra, dec = csiborgtools.utils.cartesian_to_radec(arr)
    arr = csiborgtools.utils.add_columns(
        arr, [d, ra, dec], ["dist", "ra", "dec"])
    return arr


def load_mmains(N=None, verbose=True):
    ids = csiborgtools.io.get_csiborg_ids("/mnt/extraspace/hdesmond")
    N = ids.size if N is None else N
    if N > ids.size:
        raise ValueError("`N` cannot be larger than 101.")
    # If N less than num of CSiBORG, then radomly choose
    if N == ids.size:
        choices = numpy.arange(N)
    else:
        choices = numpy.random.choice(ids.size, N, replace=False)

    out = [None] * N
    iters = trange(N) if verbose else range(N)
    for i in iters:
        j = choices[i]
        out[i] = load_mmain_convert(ids[j])
    return out


def load_planck2015(max_comdist=214):
    cosmo = FlatLambdaCDM(H0=70.5, Om0=0.307, Tcmb0=2.728)
    fpath = ("/mnt/zfsusers/rstiskalek/csiborgtools/"
             + "data/HFI_PCCS_SZ-union_R2.08.fits")
    return csiborgtools.io.read_planck2015(fpath, cosmo, max_comdist)


def load_2mpp():
    cosmo = FlatLambdaCDM(H0=70.5, Om0=0.307, Tcmb0=2.728)
    return csiborgtools.io.read_2mpp("../data/2M++_galaxy_catalog.dat", cosmo)
