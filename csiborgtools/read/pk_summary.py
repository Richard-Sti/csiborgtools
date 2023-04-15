# Copyright (C) 2022 Richard Stiskalek, Harry Desmond
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
"""Power spectrum reader."""
import numpy
import joblib
from tqdm import tqdm


class PKReader:
    """
    A shortcut object for reading in the power spectrum files.

    Parameters
    ----------
    get_ics : list of int
        IC IDs to be read.
    hw : float
        Box half-width.
    fskel : str, optional
        The skeleton path. By default
        `/mnt/extraspace/rstiskalek/csiborg/crosspk/out_{}_{}_{}.p`, where
        the formatting options are `ic0, ic1, hw`.
    dtype : dtype, optional
        Output precision. By default `numpy.float32`.
    """
    def __init__(self, get_ics, hw, fskel=None, dtype=numpy.float32):
        self.get_ics = get_ics
        self.hw = hw
        if fskel is None:
            fskel = "/mnt/extraspace/rstiskalek/csiborg/crosspk/out_{}_{}_{}.p"
        self.fskel = fskel
        self.dtype = dtype

    @staticmethod
    def _set_klim(kmin, kmax):
        """
        Sets limits on the wavenumber to 0 and infinity if `None`s provided.
        """
        if kmin is None:
            kmin = 0
        if kmax is None:
            kmax = numpy.infty
        return kmin, kmax

    def read_autos(self, kmin=None, kmax=None):
        """
        Read in the autocorrelation power spectra.

        Parameters
        ----------
        kmin : float, optional
            The minimum wavenumber. By default `None`, i.e. 0.
        kmin : float, optional
            The maximum wavenumber. By default `None`, i.e. infinity.

        Returns
        -------
        ks : 1-dimensional array
            Array of wavenumbers.
        pks : 2-dimensional array of shape `(len(self.get_ics), ks.size)`
            Autocorrelation of each simulation.
        """
        kmin, kmax = self._set_klim(kmin, kmax)
        ks, pks, sel = None, None, None
        for i, nsim in enumerate(self.get_ics):
            pk = joblib.load(self.fskel.format(nsim, nsim, self.hw))
            # Get cuts and pre-allocate arrays
            if i == 0:
                x = pk.k3D
                sel = (kmin < x) & (x < kmax)
                ks = x[sel].astype(self.dtype)
                pks = numpy.full((len(self.get_ics), numpy.sum(sel)), numpy.nan,
                                 dtype=self.dtype)
            pks[i, :] = pk.Pk[sel, 0, 0]

        return ks, pks

    def read_single_cross(self, ic0, ic1, kmin=None, kmax=None):
        """
        Read cross-correlation between IC IDs `ic0` and `ic1`.

        Parameters
        ----------
        ic0 : int
            The first IC ID.
        ic1 : int
            The second IC ID.
        kmin : float, optional
            The minimum wavenumber. By default `None`, i.e. 0.
        kmin : float, optional
            The maximum wavenumber. By default `None`, i.e. infinity.

        Returns
        -------
        ks : 1-dimensional array
            Array of wavenumbers.
        xpk : 1-dimensional array of shape `(ks.size, )`
            Cross-correlation.
        """
        if ic0 == ic1:
            raise ValueError("Requested cross correlation for the same ICs.")
        kmin, kmax = self._set_klim(kmin, kmax)
        # Check their ordering. The latter must be larger.
        ics = (ic0, ic1)
        if ic0 > ic1:
            ics = ics[::-1]

        pk = joblib.load(self.fskel.format(*ics, self.hw))
        ks = pk.k3D
        sel = (kmin < ks) & (ks < kmax)
        ks = ks[sel].astype(self.dtype)
        xpk = pk.XPk[sel, 0, 0].astype(self.dtype)

        return ks, xpk

    def read_cross(self, kmin=None, kmax=None):
        """
        Read cross-correlation between all IC pairs.

        Parameters
        ----------
        kmin : float, optional
            The minimum wavenumber. By default `None`, i.e. 0.
        kmin : float, optional
            The maximum wavenumber. By default `None`, i.e. infinity.

        Returns
        -------
        ks : 1-dimensional array
            Array of wavenumbers.
        xpks : 3-dimensional array of shape (`nics, nics - 1, ks.size`)
            Cross-correlations. The first column is the the IC and is being
            cross-correlated with the remaining ICs, in the second column.
        """
        nics = len(self.get_ics)

        ks, xpks = None, None
        for i, ic0 in enumerate(tqdm(self.get_ics)):
            k = 0
            for ic1 in self.get_ics:
                # We don't want cross-correlation
                if ic0 == ic1:
                    continue
                x, y = self.read_single_cross(ic0, ic1, kmin, kmax)
                # If in the first iteration pre-allocate arrays
                if ks is None:
                    ks = x
                    xpks = numpy.full((nics, nics - 1, ks.size), numpy.nan,
                                      dtype=self.dtype)
                xpks[i, k, :] = y
                # Bump up the iterator
                k += 1

        return ks, xpks
