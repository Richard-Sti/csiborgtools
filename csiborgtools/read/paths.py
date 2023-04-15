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
"""CSiBORG paths manager."""
from os.path import (join, isdir)
from glob import glob
import numpy


class CSiBORGPaths:
    """
    Paths manager for CSiBORG IC realisations.

    Parameters
    ----------
    srcdir : str
        Path to the folder where CSiBORG simulations are stored.
    dumpdir : str
        TODO: fix this docstring.
        Path to the folder where files from `run_fit_halos` are stored.
    _mmain_dir : str
        Path to folder where mmain files are stored.
    initmatch_path : str
        Path to the folder where particle ID match between the first and final
        snapshot is stored.
    """
    _srcdir = None
    _dumpdir = None
    _initmatch_path = None

    def __init__(self, srcdir=None, dumpdir=None, initmatch_path=None):
        self.srcdir = srcdir
        self.dumpdir = dumpdir
        self.initmatch_path = initmatch_path

    @staticmethod
    def _check_directory(path):
        if not isdir(path):
            raise IOError("Invalid directory `{}`!".format(path))

    @property
    def srcdir(self):
        """
        Path to the folder where CSiBORG simulations are stored.

        Returns
        -------
        path : str
        """
        if self._srcdir is None:
            raise ValueError("`srcdir` is not set!")
        return self._srcdir

    @srcdir.setter
    def srcdir(self, path):
        if path is None:
            return
        self._check_directory(path)
        self._srcdir = path

    @property
    def dumpdir(self):
        """
        Path to the folder where files from `run_fit_halos` are stored.

        Returns
        -------
        path : str
        """
        if self._dumpdir is None:
            raise ValueError("`dumpdir` is not set!")
        return self._dumpdir

    @dumpdir.setter
    def dumpdir(self, path):
        if path is None:
            return
        self._check_directory(path)
        self._dumpdir = path

    @property
    def temp_dumpdir(self):
        """
        Path to a temporary dumping folder.

        Returns
        -------
        path : str
        """
        fpath = join(self.dumpdir, "temp")
        if not isdir(fpath):
            raise IOError("Invalid directory `{}`.".format(fpath))
        return fpath

    def mmain_path(self, nsnap, nsim):
        """
        Path to the `mmain` files summed substructure files.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.

        Returns
        -------
        path : str
        """
        return join(
            self.dumpdir,
            "mmain",
            "mmain_{}_{}.npz".format(str(nsim).zfill(5), str(nsnap).zfill(5))
            )

    @property
    def initmatch_path(self):
        """
        Path to the folder where the match between the first and final
        snapshot is stored.

        Returns
        -------
        path : str
        """
        if self._initmatch_path is None:
            raise ValueError("`initmatch_path` is not set!")
        return self._initmatch_path

    @initmatch_path.setter
    def initmatch_path(self, path):
        if path is None:
            return
        self._check_directory(path)
        self._initmatch_path = path

    def get_ics(self, tonew):
        """
        Get CSiBORG IC realisation IDs from the list of folders in
        `self.srcdir`.

        Parameters
        ----------
        tonew : bool
            If `True`, path to the '_new' ICs is returned.
        Returns
        -------
        ids : 1-dimensional array
        """
        files = glob(join(self.srcdir, "ramses_out*"))
        files = [f.split("/")[-1] for f in files]  # Select only file names
        if tonew:
            files = [f for f in files if "_new" in f]
            ids = [int(f.split("_")[2]) for f in files]  # Take the IC IDs
        else:
            files = [f for f in files if "_inv" not in f]  # Remove inv. ICs
            files = [f for f in files if "_new" not in f]  # Remove _new
            files = [f for f in files if "OLD" not in f]   # Remove _old
            ids = [int(f.split("_")[-1]) for f in files]
            try:
                ids.remove(5511)
            except ValueError:
                pass
        return numpy.sort(ids)

    def ic_path(self, nsim, tonew=False):
        """
        Path to a CSiBORG IC realisation folder.

        Parameters
        ----------
        nsim : int
            IC realisation index.
        tonew : bool, optional
            Whether to return the path to the '_new' IC realisation.

        Returns
        -------
        path : str
        """
        fname = "ramses_out_{}"
        if tonew:
            fname += "_new"
        return join(self.srcdir, fname.format(nsim))

    def get_snapshots(self, nsim):
        """
        List of available snapshots of a CSiBORG IC realisation.

        Parameters
        ----------
        nsim : int
            IC realisation index.

        Returns
        -------
        snapshots : 1-dimensional array
        """
        simpath = self.ic_path(nsim, tonew=False)
        # Get all files in simpath that start with output_
        snaps = glob(join(simpath, "output_*"))
        # Take just the last _00XXXX from each file  and strip zeros
        snaps = [int(snap.split('_')[-1].lstrip('0')) for snap in snaps]
        return numpy.sort(snaps)

    def snapshot_path(self, nsnap, nsim):
        """
        Path to a CSiBORG IC realisation snapshot.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.

        Returns
        -------
        snappath : str
        """
        if nsnap == 1:
            tonew = True
        simpath = self.ic_path(nsim, tonew=tonew)
        return join(simpath, "output_{}".format(str(nsnap).zfill(5)))

    def clump0_path(self, nsim):
        """
        Path to a single dumped clump's particles. Expected to point to a
        dictonary whose keys are the clump indices and items structured
        arrays with the clump's particles in the initial snapshot.

        Parameters
        ----------
        nsim : int
            IC realisation index.

        Returns
        -------
        path : str
        """
        cdir = join(self.dumpdir, "initmatch")
        return join(cdir, "clump_{}_{}.npy".format(nsim, "particles"))

    def hcat_path(self, nsim):
        """
        Path to the final snapshot halo catalogue from `fit_halos.py`.

        Parameters
        ----------
        nsim : int
            IC realisation index.

        Returns
        -------
        path : str
        """
        nsnap = str(max(self.get_snapshots(nsim))).zfill(5)
        fname = "ramses_out_{}_{}.npy".format(str(self.nsim).zfill(5), nsnap)
        return join(self.dumpdir, fname)