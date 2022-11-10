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
Scripts to read in observation.
"""

import numpy
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from ..utils import (add_columns, cols_to_structured)

F64 = numpy.float64


def read_planck2015(fpath, cosmo, max_comdist=None):
    r"""
    Read the Planck 2nd Sunyaev-Zeldovich source catalogue [1]. The following
    is performed:
        - removes clusters without a redshift estimate,
        - calculates the comoving distance with the provided cosmology.
        - Converts `MSZ` from units of :math:`1e14 M_\odot` to :math:`M_\odot`

    Parameters
    ----------
    fpath : str
        Path to the source catalogue.
    cosmo : `astropy.cosmology` object
        The cosmology to calculate cluster comoving distance from redshift and
        convert their mass.
    max_comdist : float, optional
        Maximum comoving distance threshold in units of :math:`\mathrm{Mpc}`.
        By default `None` and no threshold is applied.

    Returns
    -------
    out : structured array
        The catalogue structured array.

    References
    ----------
    [1] https://heasarc.gsfc.nasa.gov/W3Browse/all/plancksz2.html
    """
    data = fits.open(fpath)[1].data
    hdata = 0.7
    # Convert FITS to a structured array
    out = numpy.full(data.size, numpy.nan, dtype=data.dtype.descr)
    for name in out.dtype.names:
        out[name] = data[name]
    # Take only clusters with redshifts
    out = out[out["REDSHIFT"] >= 0]
    # Add comoving distance
    dist = cosmo.comoving_distance(out["REDSHIFT"]).value
    out = add_columns(out, dist, "COMDIST")
    # Convert masses
    for par in ("MSZ", "MSZ_ERR_UP", "MSZ_ERR_LOW"):
        out[par] *= 1e14
        out[par] *= (hdata / cosmo.h)**2
    # Distance threshold
    if max_comdist is not None:
        out = out[out["COMDIST"] < max_comdist]

    return out


def read_mcxc(fpath, cosmo, max_comdist=None):
    r"""
    Read the MCXC Meta-Catalog of X-Ray Detected Clusters of Galaxies
    catalogue [1], with data description at [2] and download at [3].

    Note
    ----
    The exact mass conversion has non-trivial dependence on :math:`H(z)`, see
    [1] for more details. However, this should be negligible.

    Parameters
    ----------
    fpath : str
        Path to the source catalogue obtained from [3]. Expected to be the fits
        file.
    cosmo : `astropy.cosmology` object
        The cosmology to calculate cluster comoving distance from redshift and
        convert their mass.
    max_comdist : float, optional
        Maximum comoving distance threshold in units of :math:`\mathrm{Mpc}`.
        By default `None` and no threshold is applied.

    Returns
    -------
    out : structured array
        The catalogue structured array.

    References
    ----------
    [1] The MCXC: a meta-catalogue of x-ray detected clusters of galaxies
        (2011); Piffaretti, R. ;  Arnaud, M. ;  Pratt, G. W. ;  Pointecouteau,
        E. ;  Melin, J. -B.
    [2] https://heasarc.gsfc.nasa.gov/W3Browse/rosat/mcxc.html
    [3] https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/534/A109#/article
    """
    data = fits.open(fpath)[1].data
    hdata = 0.7  # Little h of the catalogue

    cols = [("RAdeg", F64), ("DEdeg", F64), ("z", F64),
            ("L500", F64), ("M500", F64), ("R500", F64)]
    out = cols_to_structured(data.size, cols)
    for col in cols:
        par = col[0]
        out[par] = data[par]
    # Get little h units to match the cosmology
    out["L500"] *= (hdata / cosmo.h)**2
    out["M500"] *= (hdata / cosmo.h)**2
    # Get the 10s back in
    out["L500"] *= 1e44  # ergs/s
    out["M500"] *= 1e14  # Msun

    dist = cosmo.comoving_distance(data["z"]).value
    out = add_columns(out, dist, "COMDIST")
    out = add_columns(out, data["MCXC"], "name")

    if max_comdist is not None:
        out = out[out["COMDIST"] < max_comdist]

    return out


def read_2mpp(fpath, dist_cosmo):
    """
    Read in the 2M++ galaxy redshift catalogue [1], with the catalogue at [2].
    Removes fake galaxies used to fill the zone of avoidance. Note that in
    principle additional care should be taken for calculating the distance
    to objects [3]. Currently calculated from the CMB redshift, so some
    distance estimates may be negative..

    Parameters
    ----------
    fpath : str
        File path to the catalogue.
    cosmo : `astropy.cosmology` object
        The cosmology to calculate distance from redshift.

    Returns
    -------
    out : structured array
        The catalogue.

    References
    ----------
    [1] The 2M++ galaxy redshift catalogue; Lavaux, Guilhem, Hudson, Michael J.
    [2] https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/416/2840#/article
    [3] Improving NASA/IPAC Extragalactic Database Redshift Calculations
        (2021); Anthony Carr and Tamara Davis
    """
    from scipy.constants import c
    # Read the catalogue and select non-fake galaxies
    cat = numpy.genfromtxt(fpath, delimiter="|", )
    cat = cat[cat[:, 12] == 0, :]

    cols = [("RA", F64), ("DEC", F64), ("Ksmag", F64), ("ZCMB", F64),
            ("DIST", F64)]
    out = cols_to_structured(cat.shape[0], cols)
    out["RA"] = cat[:, 1]
    out["DEC"] = cat[:, 2]
    out["Ksmag"] = cat[:, 5]
    out["ZCMB"] = cat[:, 7] / (c * 1e-3)
    out["DIST"] = cat[:, 7] / dist_cosmo.H0
    return out


def read_2mpp_groups(fpath, dist_cosmo):
    """
    Read in the 2M++ galaxy group catalogue [1], with the catalogue at [2].
    Note that the same caveats apply to the distance calculation as in
    py:function:`read_2mpp`. See that function for more details.

    Parameters
    ----------
    fpath : str
        File path to the catalogue.
    cosmo : `astropy.cosmology` object
        The cosmology to calculate distance from redshift.

    Returns
    -------
    out : structured array
        The catalogue.

    References
    ----------
    [1] The 2M++ galaxy redshift catalogue; Lavaux, Guilhem, Hudson, Michael J.
    [2] https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/416/2840#/article
    [3] Improving NASA/IPAC Extragalactic Database Redshift Calculations
        (2021); Anthony Carr and Tamara Davis
    """
    cat = numpy.genfromtxt(fpath, delimiter="|", )

    cols = [("RA", F64), ("DEC", F64), ("K2mag", F64), ("Rich", numpy.int64),
            ("dist", F64), ("sigma", F64)]
    out = cols_to_structured(cat.shape[0], cols)

    out["K2mag"] = cat[:, 3]
    out["Rich"] = cat[:, 4]
    out["sigma"] = cat[:, 7]
    out["dist"] = cat[:, 6] / dist_cosmo.H0

    # Convert galactic coordinates to RA, dec
    glon = cat[:, 1]
    glat = cat[:, 2]
    coords = SkyCoord(l=glon*u.degree, b=glat*u.degree, frame='galactic')
    coords = coords.transform_to("icrs")
    out["RA"] = coords.ra
    out["DEC"] = coords.dec

    return out


def match_planck_to_mcxc(planck, mcxc):
    """
    Return the MCXC catalogue indices of the Planck catalogue detections. Finds
    the index of the quoted Planck MCXC counterpart in the MCXC array. If not
    found throws an error. For this reason it may be better to make sure the
    MCXC catalogue reaches further.


    Parameters
    ----------
    planck : structured array
        The Planck cluster array.
    mcxc : structured array
        The MCXC cluster array.

    Returns
    -------
    indxs : 1-dimensional array
        The array of MCXC indices to match the Planck array. If no counterpart
        is found returns `numpy.nan`.
    """
    # Planck MCXC need to be decoded to str
    planck_names = [name.decode() for name in planck["MCXC"]]
    mcxc_names = [name for name in mcxc["name"]]

    indxs = [numpy.nan] * len(planck_names)
    for i, name in enumerate(planck_names):
        if name == "":
            continue
        if name in mcxc_names:
            indxs[i] = mcxc_names.index(name)
        else:
            raise ValueError("Planck MCXC identifies `{}` not found in the "
                             "MCXC catalogue.".format(name))
    return indxs
