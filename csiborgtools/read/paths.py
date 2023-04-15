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
from os import mkdir
from os.path import (join, isdir)
from warnings import warn
from glob import glob
import numpy


class CSiBORGPaths:
    """
    Paths manager for CSiBORG IC realisations.

    Parameters
    ----------
    srcdir : str
        Path to the folder where the RAMSES outputs are stored.
    postdir: str
        Path to the folder where post-processed files are stored.
    """
    _srcdir = None
    _postdir = None

    def __init__(self, srcdir=None, postdir=None):
        self.srcdir = srcdir
        self.postdir = postdir

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
    def postdir(self):
        """
        Path to the folder where post-processed files are stored.

        Returns
        -------
        path : str
        """
        if self._postdir is None:
            raise ValueError("`postdir` is not set!")
        return self._postdir

    @postdir.setter
    def postdir(self, path):
        if path is None:
            return
        self._check_directory(path)
        self._postdir = path

    @property
    def temp_dumpdir(self):
        """
        Path to a temporary dumping folder.

        TODO: this will soon go.

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
        fdir = join(self.postdir, "mmain")
        if not isdir(fdir):
            mkdir(fdir)
            warn("Created directory `{}`.".format(fdir), UserWarning)
        return join(
            fdir,
            "mmain_{}_{}.npz".format(str(nsim).zfill(5), str(nsnap).zfill(5))
            )

    def initmatch_path(self, nsim, kind):
        """
        Path to the `initmatch` files where the clump match between the
        initial and final snapshot is stored.

        Parameters
        ----------
        nsim : int
            IC realisation index.
        kind : str
            Type of match.  Can be either `cm` or `particles`.

        Returns
        -------
        """
        assert kind in ["cm", "particles"]
        fdir = join(self.postdir, "initmatch")
        if not isdir(fdir):
            mkdir(fdir)
            warn("Created directory `{}`.".format(fdir), UserWarning)
        return join(fdir, "{}_{}.npy".format(kind, str(nsim).zfill(5)))

    def split_path(self, clumpid, nsnap, nsim):
        """
        Path to the `split` files from `pre_splithalos`. Individual files
        contain particles belonging to a single clump.

        Parameters
        ----------
        clumpid : int
            Clump ID.
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.

        Returns
        -------
        path : str
        """
        fdir = join(self.postdir, "split", "ic_{}".format(str(nsim).zfill(5)))
        if not isdir(fdir):
            mkdir(fdir)
            warn("Created directory `{}`.".format(fdir), UserWarning)
        return join(
            fdir,
            "out_{}_{}.npy".format(str(nsnap).zfill(5), clumpid)
            )

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
        tonew = nsnap == 1
        simpath = self.ic_path(nsim, tonew=tonew)
        return join(simpath, "output_{}".format(str(nsnap).zfill(5)))

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
        return join(self.postdir, fname)
