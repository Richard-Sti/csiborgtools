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
Various I/O functions for reading and writing data.
"""
from warnings import warn

import healpy as hp
from astropy.io import fits


def read_CMB_temperature(fname=None, nside_out=None,
                         convert_to_ring_ordering=True, normalize=False,
                         verbose=True):
    """
    Read the CMB temperature map from a FITS file.
    """
    if fname is None:
        warn("Using the glamdrnig path to the default temperature map.",
             UserWarning)
        fname = "/mnt/extraspace/rstiskalek/catalogs/CMB/COM_CMB_IQU-smica_2048_R3.00_full.fits"  # noqa

    f = fits.open(fname)
    if verbose:
        print(f"Reading CMB temperature map from `{fname}`.")

    skymap = f[1].data["I_STOKES"]
    mask = f[1].data["TMASK"]
    f.close()

    if nside_out is not None:
        if verbose:
            print(f"Moving to nside = {nside_out}...")
        skymap = hp.pixelfunc.ud_grade(skymap, nside_out, order_in="NESTED")
        mask = hp.pixelfunc.ud_grade(mask, nside_out, order_in="NESTED")

    if convert_to_ring_ordering:
        if verbose:
            print("Converting to RING ordering...")
        skymap = hp.reorder(skymap, n2r=True)
        mask = hp.reorder(mask, n2r=True)

    if normalize:
        skymap -= skymap.mean()
        skymap /= skymap.std()

    return skymap, mask
