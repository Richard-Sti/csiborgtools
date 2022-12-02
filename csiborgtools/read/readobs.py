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

TODO:
- [ ] Finish SDSS surv.
- [ ] Improve the rest of this file

"""

import numpy
from abc import ABC
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from astropy import units
from scipy import constants
from warnings import warn
from ..utils import (add_columns, cols_to_structured)

F64 = numpy.float64


class BaseSurvey:
    """
    Base survey class with some methods that are common to all survey classes.
    """
    _data = None
    _cosmo = None

    @property
    def data(self):
        """
        Cluster catalogue.

        Returns
        -------
        cat : structured array
            Catalogue.
        """
        if self._data is None:
            raise ValueError("`data` is not set!")
        return self._data

    @property
    def cosmo(self):
        """Desired cosmology."""
        if self._cosmo is None:
            raise ValueError("`cosmo` is not set!")
        return self._cosmo

    @property
    def keys(self):
        """Catalogue keys."""
        return self.data.dtype.names

    def __getitem__(self, key):
        return self._data[key]


class PlanckClusters(BaseSurvey):
    r"""
    Planck 2nd Sunyaev-Zeldovich source catalogue [1]. Automatically removes
    clusters without a redshift estimate.

    Parameters
    ----------
    fpath : str
        Path to the source catalogue.
    cosmo : `astropy.cosmology` object, optional
        Cosmology to convert masses (particularly :math:`H_0`). By default
        `FlatLambdaCDM(H0=70.5, Om0=0.307, Tcmb0=2.728)`.
    max_redshift: float, optional
        Maximum cluster redshift. By default `None` and no selection is
        performed.

    References
    ----------
    [1] https://heasarc.gsfc.nasa.gov/W3Browse/all/plancksz2.html
    """
    _hdata = 0.7  # little h value of the data

    def __init__(self, fpath, cosmo=None, max_redshift=None):
        if cosmo is None:
            self._cosmo = FlatLambdaCDM(H0=70.5, Om0=0.307, Tcmb0=2.728)
        else:
            self._cosmo = cosmo
        self.set_data(fpath, max_redshift)

    def set_data(self, fpath, max_redshift=None):
        """
        Set the catalogue, loads it and applies a maximum redshift cut.
        """
        cat = fits.open(fpath)[1].data
        # Convert FITS to a structured array
        data = numpy.full(cat.size, numpy.nan, dtype=cat.dtype.descr)
        for name in cat.dtype.names:
            data[name] = cat[name]
        # Take only clusters with redshifts
        data = data[data["REDSHIFT"] >= 0]
        # Convert masses
        for par in ("MSZ", "MSZ_ERR_UP", "MSZ_ERR_LOW"):
            data[par] *= 1e14
            data[par] *= (self._hdata / self.cosmo.h)**2
        # Redshift cut
        if max_redshift is not None:
            data = data["REDSHIFT" <= max_redshift]
        self._data = data

    def match_to_mcxc(self, mcxc):
        """
        Return the MCXC catalogue indices of the Planck catalogue detections.
        Finds the index of the quoted Planck MCXC counterpart in the MCXC
        array. If not found throws an error. For this reason it may be better
        to make sure the MCXC catalogue reaches further.

        Parameters
        ----------
        mcxc : :py:class`MCXCClusters`
            MCXC cluster object.

        Returns
        -------
        indxs : list of int
            Array of MCXC indices to match the Planck array. If no counterpart
            is found returns `numpy.nan`.
        """
        if not isinstance(mcxc, MCXCClusters):
            raise TypeError("`mcxc` must be `MCXCClusters` type.")

        # Planck MCXC need to be decoded to str
        planck_names = [name.decode() for name in self["MCXC"]]
        mcxc_names = [name for name in mcxc["name"]]

        indxs = [numpy.nan] * len(planck_names)
        for i, name in enumerate(planck_names):
            if name == "":
                continue
            if name in mcxc_names:
                indxs[i] = mcxc_names.index(name)
            else:
                raise ValueError("Planck MCXC identifier `{}` not found in "
                                 "the MCXC catalogue.".format(name))
        return indxs


class MCXCClusters(BaseSurvey):
    r"""
    MCXC Meta-Catalog of X-Ray Detected Clusters of Galaxies catalogue [1],
    with data description at [2] and download at [3].

    Note
    ----
    The exact mass conversion has non-trivial dependence on :math:`H(z)`, see
    [1] for more details. However, this should be negligible.

    Parameters
    ----------
    fpath : str
        Path to the source catalogue obtained from [3]. Expected to be the fits
        file.
    cosmo : `astropy.cosmology` object, optional
        The cosmology to to convert cluster masses (to first order). By default
        `FlatLambdaCDM(H0=70.5, Om0=0.307, Tcmb0=2.728)`.
    max_redshift: float, optional
        Maximum cluster redshift. By default `None` and no selection is
        performed.

    References
    ----------
    [1] The MCXC: a meta-catalogue of x-ray detected clusters of galaxies
        (2011); Piffaretti, R. ;  Arnaud, M. ;  Pratt, G. W. ;  Pointecouteau,
        E. ;  Melin, J. -B.
    [2] https://heasarc.gsfc.nasa.gov/W3Browse/rosat/mcxc.html
    [3] https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/534/A109#/article
    """
    _hdata = 0.7  # Little h of the catalogue

    def __init__(self, fpath, cosmo=None, max_redshift=None):
        if cosmo is None:
            self._cosmo = FlatLambdaCDM(H0=70.5, Om0=0.307, Tcmb0=2.728)
        else:
            self._cosmo = cosmo
        self._set_data(fpath, max_redshift)

    def _set_data(self, fpath, max_redshift):
        """
        Set the catalogue, loads it and applies a maximum redshift cut.
        """
        cat = fits.open(fpath)[1].data
        # Pre-allocate array and extract selected variables
        cols = [("RAdeg", F64), ("DEdeg", F64), ("z", F64),
                ("L500", F64), ("M500", F64), ("R500", F64)]
        data = cols_to_structured(cat.size, cols)
        for col in cols:
            par = col[0]
            data[par] = cat[par]
        # Add the cluster names
        data = add_columns(data, cat["MCXC"], "name")

        # Get little h units to match the cosmology
        data["L500"] *= (self._hdata / self.cosmo.h)**2
        data["M500"] *= (self._hdata / self.cosmo.h)**2
        # Get the 10s back in
        data["L500"] *= 1e44  # ergs/s
        data["M500"] *= 1e14  # Msun

        if max_redshift is not None:
            data = data["z" <= max_redshift]

        self._data = data


class TwoMPPGalaxies(BaseSurvey):
    """
    The 2M++ galaxy redshift catalogue [1], with the catalogue at [2].
    Removes fake galaxies used to fill the zone of avoidance. Note that the
    stated redshift is in the CMB frame.

    Parameters
    ----------
    fpath : str
        File path to the catalogue.

    References
    ----------
    [1] The 2M++ galaxy redshift catalogue; Lavaux, Guilhem, Hudson, Michael J.
    [2] https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/416/2840#/article
    [3] Improving NASA/IPAC Extragalactic Database Redshift Calculations
        (2021); Anthony Carr and Tamara Davis
    """

    def __init__(self, fpath):
        self._set_data(fpath)

    def _set_data(self, fpath):
        """
        Set the catalogue
        """
        from scipy.constants import c
        # Read the catalogue and select non-fake galaxies
        cat = numpy.genfromtxt(fpath, delimiter="|", )
        cat = cat[cat[:, 12] == 0, :]
        # Pre=allocate array and fillt it
        cols = [("RA", F64), ("DEC", F64), ("Ksmag", F64), ("ZCMB", F64),
                ("DIST", F64)]
        data = cols_to_structured(cat.shape[0], cols)
        data["RA"] = cat[:, 1]
        data["DEC"] = cat[:, 2]
        data["Ksmag"] = cat[:, 5]
        data["ZCMB"] = cat[:, 7] / (c * 1e-3)
        self._data = data


class TwoMPPGroups(BaseSurvey):
    """
    The 2M++ galaxy group catalogue [1], with the catalogue at [2].

    Parameters
    ----------
    fpath : str
        File path to the catalogue.

    References
    ----------
    [1] The 2M++ galaxy redshift catalogue; Lavaux, Guilhem, Hudson, Michael J.
    [2] https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/416/2840#/article
    [3] Improving NASA/IPAC Extragalactic Database Redshift Calculations
        (2021); Anthony Carr and Tamara Davis
    """

    def __init__(self, fpath):
        self._set_data(fpath)

    def _set_data(self, fpath):
        """
        Set the catalogue
        """
        cat = numpy.genfromtxt(fpath, delimiter="|", )
        # Pre-allocate and fill the array
        cols = [("RA", F64), ("DEC", F64), ("K2mag", F64),
                ("Rich", numpy.int64), ("sigma", F64)]
        data = cols_to_structured(cat.shape[0], cols)
        data["K2mag"] = cat[:, 3]
        data["Rich"] = cat[:, 4]
        data["sigma"] = cat[:, 7]

        # Convert galactic coordinates to RA, dec
        glon = data[:, 1]
        glat = data[:, 2]
        coords = SkyCoord(l=glon*units.degree, b=glat*units.degree,
                          frame='galactic')
        coords = coords.transform_to("icrs")
        data["RA"] = coords.ra
        data["DEC"] = coords.dec
        self._data = data


###############################################################################
#                The following do not inherit from `BaseSurvey`!              #
###############################################################################


class FitsSurvey(ABC):
    """
    Base class for extracting data from FITS files. Contains two sets of
    keys: `routine_keys` and `fits_keys`. The former are user-defined
    properties calculated from the FITS file data. Both are accesible via
    `self[key]`.
    """
    _file = None
    _h = None
    _routines = None

    @property
    def file(self):
        """
        The survey FITS file.

        Returns
        -------
        file : py:class:`astropy.io.fits.hdu.hdulist.HDUList`
        """
        if self._file is None:
            raise ValueError("`file` is not set!")
        return self._file

    @property
    def h(self):
        """
        Little h.

        Returns
        -------
        h : float
        """
        return self._h

    @h.setter
    def h(self, h):
        """Sets the little h."""
        self._h = h

    @staticmethod
    def _check_in_list(member, members, kind):
        """
        Checks that `member` is a member of a list `members`, `kind` is a
        member type name.
        """
        if member not in members:
            raise ValueError("Unknown {} `{}`, must be one of `{}`."
                             .format(kind, member, members))

    @property
    def routines(self):
        """
        Processing routines.

        Returns
        -------
        routines : dict
            Dictionary of routines. Keys are functions and values are their
            arguments.
        """
        return self._routines

    @property
    def fits_keys(self):
        """
        Keys of the FITS file `self.file`.

        Parameters
        ----------
        keys : list of str
        """
        return self.file[1].data.columns.names

    @property
    def routine_keys(self):
        """
        Routine keys.

        Parameters
        ----------
        keys : list of str
        """
        return list(self.routines.keys())

    def get_fitsitem(self, key):
        """
        Get a column `key` from the FITS file `self.file`.

        Parameters
        ----------
        key : str

        Returns
        -------
        col : 1-dimensional array
        """
        return self.file[1].data[key]

    @property
    def keys(self):
        """
        Routine and FITS keys.

        Returns
        -------
        keys : list of str
        """
        return self.routine_keys + self.fits_keys

    def __getitem__(self, key):
        """
        Return values for this `key`. If in both return from `routine_keys`.
        """
        # Check duplicates
        if key in self.routine_keys and key in self.fits_keys:
            warn("Key `{}` found in both `routine_keys` and `fits_keys`. "
                 "Returning `routine_keys` value.".format(key), UserWarning)

        if key in self.routine_keys:
            func, args = self.routines[key]
            return func(*args)
        elif key in self.fits_keys:
            warn("Returning a FITS property. Be careful about little h!",
                 UserWarning)
            return self.get_fitsitem(key)
        else:
            raise KeyError("Unrecognised key `{}`.".format(key))


class SDSS(FitsSurvey):
    """
    SDSS data manipulations. Carries routines for ABSMAG, APPMAG, COL, DIST,
    MTOL.

    TODO:
    - [ ] masking

    Parameters
    ----------
    fpath : str
        Path to the FITS file.
        By default `/mnt/extraspace/rstiskalek/catalogs/nsa_v1_0_1.fits`
    h : float
        Little h. By default `h = 1`. The catalogue assumes this value.
        The routine properties should take care of little h conversion.
    """

    def __init__(self, fpath=None, h=1):
        if fpath is None:
            fpath = "/mnt/extraspace/rstiskalek/catalogs/nsa_v1_0_1.fits"
        self._file = fits.open(fpath, memmap=False)
        self.h = h

        # Survey bands and photometries
        self._bands = ['F', 'N', 'u', 'g', 'r', 'i', 'z']
        self._photos = ["SERSIC", "ELPETRO"]

        self._routines = {}
        # Set ABSMAGroutines
        for photo in self._photos:
            for band in self._bands:
                # ABSMAG
                key = "{}_ABSMAG_{}".format(photo, band)
                val = (self._absmag, (photo, band))
                self.routines.update({key: val})
        # Set APPMAG routines
        for photo in self._photos:
            for band in self._bands:
                key = "{}_APPMAG_{}".format(photo, band)
                val = (self._appmag, (photo, band))
                self.routines.update({key: val})
        # Set COL routines
        for photo in self._photos:
            for band1 in self._bands:
                for band2 in self._bands:
                    key = "{}_COL_{}{}".format(photo, band1, band2)
                    val = (self._colour, (photo, band1, band2))
                    self.routines.update({key: val})
        # Set DIST routine
        self.routines.update({"DIST": (self._dist, ())})
        # Set MASS routines
        for photo in self._photos:
            key = "{}_MASS".format(photo)
            val = (self._solmass, (photo,))
            self.routines.update({key: val})
        # Set MTOL
        for photo in self._photos:
            for band in self._bands:
                key = "{}_MTOL_{}".format(photo, band)
                val = (self._mtol, (photo, band))
                self.routines.update({key: val})

    def _absmag(self, photo, band):
        """
        Get absolute magnitude of a given photometry and band. Converts to
        the right little h.
        """
        self._check_in_list(photo, self._photos, "photometry")
        self._check_in_list(band, self._bands, "band")
        k = self._bands.index(band)
        mag = self.get_fitsitem("{}_ABSMAG".format(photo))[:, k]
        return mag + 5 * numpy.log10(self.h)

    def _kcorr(self, photo, band):
        """
        Get K-correction of a given photometry and band.
        """
        self._check_in_list(photo, self._photos, "photometry")
        self._check_in_list(band, self._bands, "band")
        k = self._bands.index(band)
        return self.get_fitsitem("{}_KCORRECT".format(photo))[:, k]

    def _appmag(self, photo, band):
        """
        Get apparent magnitude of a given photometry and band.
        """
        lumdist = (1 + self.get_fitsitem("ZDIST")) * self._dist()
        absmag = self._absmag(photo, band)
        kcorr = self._kcorr(photo, band)
        return absmag + 25 + 5 * numpy.log10(lumdist) + kcorr

    def _colour(self, photo, band1, band2):
        """
        Get colour of a given photometry, i.e. `band1` - `band2` absolute
        magnitude.
        """
        return self._absmag(photo, band1) - self._absmag(photo, band2)

    def _dist(self):
        """
        Get the corresponding distance estimate from `ZDIST`, which is defined
        as:
            "Distance estimate using pecular velocity model of Willick et al.
            (1997), expressed as a redshift equivalent; multiply by c/H0 for
            Mpc"

        Converts little h.
        """
        return self.get_fitsitem("ZDIST") * constants.c * 1e-3 / (100 * self.h)

    def _solmass(self, photo):
        """
        Get solar mass of a given photometry. Converts little h.
        """
        self._check_in_list(photo, self._photos, "photometry")
        return self.get_fitsitem("{}_MASS".format(photo)) / self.h**2

    def _mtol(self, photo, band):
        """
        Get mass-to-light ratio of a given photometry. Converts little h.
        """
        self._check_in_list(photo, self._photos, "photometry")
        self._check_in_list(band, self._bands, "band")
        k = self._bands.index(band)
        return self.get_fitsitem("{}_MTOL".format(photo))[:, k] / self.h**2
