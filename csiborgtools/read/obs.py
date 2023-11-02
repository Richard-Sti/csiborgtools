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
from abc import ABC, abstractproperty
from os.path import join
from warnings import warn

import numpy
from astropy import units
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from scipy import constants

from .utils import cols_to_structured

###############################################################################
#                           Text survey base class                            #
###############################################################################


class TextSurvey:
    """
    Base survey class for extracting data from text files.
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


###############################################################################
#                              2M++ galaxies                                  #
###############################################################################


class TwoMPPGalaxies(TextSurvey):
    """
    The 2M++ galaxy redshift catalogue [1], with the catalogue at [2].
    Removes fake galaxies used to fill the zone of avoidance. Note that the
    stated redshift is in the CMB frame.

    Parameters
    ----------
    fpath : str, optional.
        File path to the catalogue. By default
        `/mnt/extraspace/rstiskalek/catalogs/2M++_galaxy_catalog.dat`.

    References
    ----------
    [1] The 2M++ galaxy redshift catalogue; Lavaux, Guilhem, Hudson, Michael J.
    [2] https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/416/2840#/article
    [3] Improving NASA/IPAC Extragalactic Database Redshift Calculations
        (2021); Anthony Carr and Tamara Davis
    """
    name = "2M++_galaxies"

    def __init__(self, fpath=None):
        if fpath is None:
            fpath = join("/mnt/extraspace/rstiskalek/catalogs/"
                         "2M++_galaxy_catalog.dat")
        self._set_data(fpath)

    def _set_data(self, fpath):
        from scipy.constants import c

        # Read the catalogue and select non-fake galaxies
        cat = numpy.genfromtxt(fpath, delimiter="|", )
        cat = cat[cat[:, 12] == 0, :]
        # Pre=allocate array and fillt it
        cols = [("RA", numpy.float64), ("DEC", numpy.float64),
                ("Ksmag", numpy.float64), ("ZCMB", numpy.float64),
                ("DIST", numpy.float64)]
        data = cols_to_structured(cat.shape[0], cols)
        data["RA"] = cat[:, 1]
        data["DEC"] = cat[:, 2]
        data["Ksmag"] = cat[:, 5]
        data["ZCMB"] = cat[:, 7] / (c * 1e-3)
        self._data = data


###############################################################################
#                             2M++ groups                                     #
###############################################################################


class TwoMPPGroups(TextSurvey):
    """
    The 2M++ galaxy group catalogue [1], with the catalogue at [2].

    Parameters
    ----------
    fpath : str, optional
        File path to the catalogue. By default
        `/mnt/extraspace/rstiskalek/catalogs/2M++_group_catalog.dat`

    References
    ----------
    [1] The 2M++ galaxy redshift catalogue; Lavaux, Guilhem, Hudson, Michael J.
    [2] https://cdsarc.cds.unistra.fr/viz-bin/cat/J/MNRAS/416/2840#/article
    [3] Improving NASA/IPAC Extragalactic Database Redshift Calculations
        (2021); Anthony Carr and Tamara Davis
    """
    name = "2M++_groups"

    def __init__(self, fpath=None):
        if fpath is None:
            fpath = join("/mnt/extraspace/rstiskalek/catalogs",
                         "2M++_group_catalog.dat")
        self._set_data(fpath)

    def _set_data(self, fpath):
        cat = numpy.genfromtxt(fpath, delimiter="|", )
        # Pre-allocate and fill the array
        cols = [("RA", numpy.float64), ("DEC", numpy.float64),
                ("K2mag", numpy.float64), ("Rich", numpy.int64),
                ("sigma", numpy.float64)]
        data = cols_to_structured(cat.shape[0], cols)
        data["K2mag"] = cat[:, 3]
        data["Rich"] = cat[:, 4]
        data["sigma"] = cat[:, 7]

        # Convert galactic coordinates to RA, dec
        glon = cat[:, 1]
        glat = cat[:, 2]
        coords = SkyCoord(l=glon * units.degree, b=glat * units.degree,
                          frame='galactic')
        coords = coords.transform_to("icrs")
        data["RA"] = coords.ra
        data["DEC"] = coords.dec
        self._data = data


###############################################################################
#                             FITS base class                                 #
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
    _selection_mask = None

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
        self._h = h

    @staticmethod
    def _check_in_list(member, members, kind):
        """
        Check that `member` is a member of a list `members`, `kind` is a
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

    @abstractproperty
    def size(self):
        """
        Return the number of samples in the catalogue.

        Returns
        -------
        size : int
        """
        pass

    @property
    def masked_size(self):
        if self.selection_mask is None:
            return self.size
        return numpy.sum(self.selection_mask)

    @property
    def selection_mask(self):
        """
        Selection mask, generated with `fmask` when initialised.

        Returns
        -------
        mask : 1-dimensional boolean array
        """
        return self._selection_mask

    @selection_mask.setter
    def selection_mask(self, mask):
        """Set the selection mask."""
        if not (isinstance(mask, numpy.ndarray)
                and mask.ndim == 1
                and mask.dtype == bool):
            raise TypeError("`selection_mask` must be a 1-dimensional boolean "
                            "array. Check output of `fmask`.")
        self._selection_mask = mask

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
            FITS key.

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
        return self.routine_keys + self.fits_keys + ["INDEX"]

    def make_mask(self, steps):
        """
        Make a survey mask from a series of steps, expected to look as below.

        ```
        def steps(cls):
            return [(lambda x: cls[x], ("IN_DR7_LSS",)),
                    (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
                   ]
        ```

        Parameters
        ----------
        steps : list of steps
            Selection steps.

        Returns
        -------
        mask : 1-dimensional boolean array
        """
        out = None
        steps = steps(self)
        for i, step in enumerate(steps):
            func, args = step
            if i == 0:
                out = func(*args)
            else:
                out = out & func(*args)
        return out

    def __getitem__(self, key):
        if key == "INDEX":
            mask = self.selection_mask
            if mask is None:
                return numpy.arange(self.size)
            else:
                return numpy.arange(mask.size)[mask]

        # Check duplicates
        if key in self.routine_keys and key in self.fits_keys:
            warn(f"Key `{key}` found in both `routine_keys` and `fits_keys`. "
                 "Returning `routine_keys` value.")

        if key in self.routine_keys:
            func, args = self.routines[key]
            out = func(*args)
        elif key in self.fits_keys:
            warn(f"Returning a FITS property `{key}`. "
                 "Be careful about little h!")
            out = self.get_fitsitem(key)
        else:
            raise KeyError(f"Unrecognised key `{key}`.")

        if self.selection_mask is None:
            return out
        return out[self.selection_mask]

    def __len__(self):
        return self.size


###############################################################################
#                            Planck clusters                                  #
###############################################################################


class PlanckClusters(FitsSurvey):
    r"""
    Planck 2nd Sunyaev-Zeldovich source catalogue [1].

    Parameters
    ----------
    fpath : str, optional
        Path to the FITS file. By default
        `/mnt/extraspace/rstiskalek/catalogs/HFI_PCCS_SZ-union_R2.08.fits`.
    h : float, optional
        Little h. By default `h = 0.7`. The catalogue assumes this value.
        The routine properties should take care of little h conversion.
    sel_steps : py:function:
        Steps to mask the survey. Expected to look for example like
        ```
        def steps(cls):
            return [(lambda x: cls[x], ("IN_DR7_LSS",)),
                    (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
                   ]
        ```

    References
    ----------
    [1] https://heasarc.gsfc.nasa.gov/W3Browse/all/plancksz2.html
    """
    name = "Planck_clusters"
    _hdata = 0.7  # little h value of the data

    def __init__(self, fpath=None, h=0.7, sel_steps=None):
        if fpath is None:
            fpath = join("/mnt/extraspace/rstiskalek/catalogs/",
                         "HFI_PCCS_SZ-union_R2.08.fits")
        self._file = fits.open(fpath, memmap=False)
        self.h = h

        self._routines = {}
        # Set MSZ routines
        for key in ("MSZ", "MSZ_ERR_UP", "MSZ_ERR_LOW"):
            self._routines.update({key: (self._mass, (key,))})

        # Add masking. Do this at the end!
        if sel_steps is not None:
            self.selection_mask = self.make_mask(sel_steps)

    @property
    def size(self):
        return self.get_fitsitem("MSZ").size

    def _mass(self, key):
        """Get mass. Puts in units of 1e14 and converts little h."""
        return self.get_fitsitem(key) * 1e14 * (self._hdata / self.h)**2

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
        mcxc_names = [name for name in mcxc["MCXC"]]

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


###############################################################################
#                             MCXC Clusters                                   #
###############################################################################


class MCXCClusters(FitsSurvey):
    r"""
    MCXC Meta-Catalog of X-Ray Detected Clusters of Galaxies catalogue [1],
    with data description at [2] and download at [3].

    Note
    ----
    The exact mass conversion has non-trivial dependence on :math:`H(z)`, see
    [1] for more details. However, this should be negligible.

    Parameters
    ----------
    fpath : str, optional
        Path to the source catalogue obtained from [3]. Expected to be the fits
        file. By default `/mnt/extraspace/rstiskalek/catalogs/mcxc.fits`.
    h : float, optional
        Little h. By default `h = 0.7`. The catalogue assumes this value.
        The routine properties should take care of little h conversion.
    sel_steps : py:function:
        Steps to mask the survey. Expected to look for example like
        ```
            steps = [(lambda x: cls[x], ("IN_DR7_LSS",)),
                     (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
                     ]
        ```.


    References
    ----------
    [1] The MCXC: a meta-catalogue of x-ray detected clusters of galaxies
        (2011); Piffaretti, R. ;  Arnaud, M. ;  Pratt, G. W. ;  Pointecouteau,
        E. ;  Melin, J. -B.
    [2] https://heasarc.gsfc.nasa.gov/W3Browse/rosat/mcxc.html
    [3] https://cdsarc.cds.unistra.fr/viz-bin/cat/J/A+A/534/A109#/article
    """
    name = "MCXC"
    _hdata = 0.7  # Little h of the catalogue

    def __init__(self, fpath=None, h=0.7, sel_steps=None):
        if fpath is None:
            fpath = "/mnt/extraspace/rstiskalek/catalogs/mcxc.fits"
        self._file = fits.open(fpath, memmap=False)
        self.h = h
        # Set mass and luminosity routines
        self._routines = {}
        self._routines.update({"M500": (self._mass, ("M500",))})
        self._routines.update({"L500": (self._lum, ("L500",))})

        if sel_steps is not None:
            self.selection_mask = self.make_mask(sel_steps)

    @property
    def size(self):
        return self.get_fitsitem("M500").size

    def _mass(self, key):
        """Get mass. Put in units of 1e14 Msun back and convert little h."""
        return self.get_fitsitem(key) * 1e14 * (self._hdata / self.h)**2

    def _lum(self, key):
        """Get luminosity, puts back units to be in ergs/s."""
        return self.get_fitsitem(key) * 1e44 * (self._hdata / self.h)**2


###############################################################################
#                              SDSS galaxies                                  #
###############################################################################


class SDSS(FitsSurvey):
    """
    SDSS data manipulations. Data obtained from [1]. Carries routines for
    ABSMAG, APPMAG, COL, DIST, MTOL.

    Parameters
    ----------
    fpath : str, optional
        Path to the FITS file.
    h : float, optional
        Little h. By default `h = 1`. The catalogue assumes this value.
        The routine properties should take care of little h conversion.
    Om0 : float, optional
        Matter density. By default `Om0 = 0.3175`, matching CSiBORG.
    sel_steps : py:function:
        Steps to mask the survey. Expected to look for example like
        ```
            steps = [(lambda x: cls[x], ("IN_DR7_LSS",)),
                     (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
                     ]
        ```.

    References
    ----------
    [1] https://www.sdss.org/dr13/manga/manga-target-selection/nsa/
    """
    name = "SDSS"

    def __init__(self, fpath, h=1, Om0=0.3175, sel_steps=None):
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
        self.routines.update(
            {"DIST_UNCORRECTED": (self._dist_uncorrected, (Om0,))})
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
        # Set IN_DR7_LSS
        self.routines.update({"IN_DR7_LSS": (self._in_dr7_lss, ())})

        # Add masking. Do this at the end!
        if sel_steps is not None:
            self.selection_mask = self.make_mask(sel_steps)

    @property
    def size(self):
        mask = self.selection_mask
        if mask is not None:
            return numpy.sum(mask)
        else:
            return self.get_fitsitem("ZDIST").size

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
        r"""
        Get the corresponding distance estimate from `ZDIST`, defined as below.

            "Distance estimate using pecular velocity model of Willick et al.
            (1997), expressed as a redshift equivalent; multiply by c/H0 for
            Mpc"

        Distance is converted to math:`h != 1` units.
        """
        return self.get_fitsitem("ZDIST") * constants.c * 1e-3 / (100 * self.h)

    def _dist_uncorrected(self, Om0):
        """
        Get the comoving distance estimate from `Z`, i.e. redshift uncorrected
        for peculiar motion in the heliocentric frame.
        """
        cosmo = FlatLambdaCDM(H0=100 * self.h, Om0=Om0)
        return cosmo.comoving_distance(self.get_fitsitem("Z")).value

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

    def _in_dr7_lss(self):
        """
        Get `IN_DR7_LSS` and turn to a boolean array.
        """
        return self.get_fitsitem("IN_DR7_LSS").astype(bool)
