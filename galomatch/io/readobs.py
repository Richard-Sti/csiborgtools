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

import numpy
from astropy.io import fits

from ..utils import (add_columns, cols_to_structured)


def read_planck2015(fpath, dist_cosmo, max_comdist=None):
    """
    Read the Planck 2nd Sunyaev-Zeldovich source catalogue [1]. The following
    is performed:
        - substracts 180 degrees from the right ascension,
        - removes clusters without a redshift estimate,
        - calculates the comoving distance with the provided cosmology.
        - Converts `MSZ` from units of :math:`1e14 M_\odot` to :math:`M_\odot`

    Parameters
    ----------
    fpath : str
        Path to the source catalogue.
    dist_cosmo : `astropy.cosmology` object
        The cosmology to calculate cluster comoving distance from redshift.
    max_comdist : float, optional
        Maximum comoving distance threshold in units of :math:`\mathrm{MPc}`.
        By default `None` and no threshold is applied.

    References
    ----------
    [1] https://heasarc.gsfc.nasa.gov/W3Browse/all/plancksz2.html

    Returns
    -------
    out : `astropy.io.fits.FITS_rec`
        The catalogue structured array.
    """
    data = fits.open(fpath)[1].data
    # Convert FITS to a structured array
    out = numpy.full(data.size, numpy.nan, dtype=data.dtype.descr)
    for name in out.dtype.names:
        out[name] = data[name]
    # Substract the RA and take only clusters with redshifts
    out["RA"] -= 180
    out = out[out["REDSHIFT"] >= 0]
    # Add comoving distance
    dist = dist_cosmo.comoving_distance(out["REDSHIFT"]).value
    out = add_columns(out, dist, "COMDIST")
    # Convert masses
    for p in ("MSZ", "MSZ_ERR_UP", "MSZ_ERR_LOW"):
        out[p] *= 1e14
    # Distance threshold
    if max_comdist is not None:
        out = out[out["COMDIST"] < max_comdist]

    return out


def read_2mpp(fpath):
    """
    Read in the 2M++ galaxy redshift catalogue [1], with the catalogue at [2].
    Removes fake galaxies used to fill the zone of avoidance.

    Parameters
    ----------
    fpath : str
        File path to the catalogue.

    Returns
    -------
    out : structured array
        The catalogue.
    """
    # Read the catalogue and select non-fake galaxies
    cat = numpy.genfromtxt(fpath, delimiter="|", )
    cat = cat[cat[:, 12] == 0, :]

    F64 = numpy.float64
    cols = [("RA", F64), ("DEC", F64), ("Ksmag", F64)]
    out = cols_to_structured(cat.shape[0], cols)
    out["RA"] = cat[:, 1] - 180
    out["DEC"] = cat[:, 2]
    out["Ksmag"] = cat[:, 5]

    return out
