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
from glob import glob
from os import makedirs, mkdir
from os.path import isdir, join
from warnings import warn

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

        Returns
        -------
        path : str
        """
        fpath = join(self.postdir, "temp")
        if not isdir(fpath):
            mkdir(fpath)
            warn(f"Created directory `{fpath}`.", UserWarning, stacklevel=1)
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
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        return join(fdir,
                    f"mmain_{str(nsim).zfill(5)}_{str(nsnap).zfill(5)}.npz")

    def initmatch_path(self, nsim, kind):
        """
        Path to the `initmatch` files where the clump match between the
        initial and final snapshot is stored.

        Parameters
        ----------
        nsim : int
            IC realisation index.
        kind : str
            Type of match. Must be one of `["particles", "fit", "halomap"]`.

        Returns
        -------
        path : str
        """
        assert kind in ["particles", "fit", "halomap"]
        ftype = "npy" if kind == "fit" else "h5"
        fdir = join(self.postdir, "initmatch")
        if not isdir(fdir):
            mkdir(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        return join(fdir, f"{kind}_{str(nsim).zfill(5)}.{ftype}")

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
            files = [f for f in files if "OLD" not in f]  # Remove _old
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
        snaps = [int(snap.split("_")[-1].lstrip("0")) for snap in snaps]
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
        return join(simpath, f"output_{str(nsnap).zfill(5)}")

    def structfit_path(self, nsnap, nsim, kind):
        """
        Path to the clump or halo catalogue from `fit_halos.py`.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.
        kind : str
            Type of catalogue.  Can be either `clumps` or `halos`.

        Returns
        -------
        path : str
        """
        assert kind in ["clumps", "halos"]
        fdir = join(self.postdir, "structfit")
        if not isdir(fdir):
            mkdir(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        fname = f"{kind}_out_{str(nsim).zfill(5)}_{str(nsnap).zfill(5)}.npy"
        return join(fdir, fname)

    def overlap_path(self, nsim0, nsimx):
        """
        Path to the overlap files between two simulations.

        Parameters
        ----------
        nsim0 : int
            IC realisation index of the first simulation.
        nsimx : int
            IC realisation index of the second simulation.

        Returns
        -------
        path : str
        """
        fdir = join(self.postdir, "overlap")
        if not isdir(fdir):
            mkdir(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        fname = f"overlap_{str(nsim0).zfill(5)}_{str(nsimx).zfill(5)}.npz"
        return join(fdir, fname)

    def radpos_path(self, nsnap, nsim):
        """
        Path to the files containing radial positions of halo particles (with
        summed substructure).

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
        fdir = join(self.postdir, "radpos")
        if not isdir(fdir):
            mkdir(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        fname = f"radpos_{str(nsim).zfill(5)}_{str(nsnap).zfill(5)}.npz"
        return join(fdir, fname)

    def particle_h5py_path(self, nsim, kind=None):
        """
        Path to the file containing all particles in a `.h5` file.

        Parameters
        ----------
        nsim : int
            IC realisation index.
        kind : str
            Type of output. Must be one of `[None, 'pos', 'clumpmap']`.
        pos_only : bool
            Whether this file contains only positions.

        Returns
        -------
        path : str
        """
        assert kind in [None, "pos", "clumpmap"]
        fdir = join(self.postdir, "particles")
        if not isdir(fdir):
            makedirs(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        if kind is None:
            fname = f"parts_{str(nsim).zfill(5)}.h5"
        else:
            fname = f"parts_{kind}_{str(nsim).zfill(5)}.h5"
        return join(fdir, fname)

    def density_field_path(self, mas, nsim):
        """
        Path to the files containing the calculated density fields.

        Parameters
        ----------
        mas : str
           Mass-assignment scheme. Currently only SPH is supported.
        nsim : int
            IC realisation index.

        Returns
        -------
        path : str
        """
        fdir = join(self.postdir, "environment")
        if not isdir(fdir):
            makedirs(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        fname = f"density_{mas}_{str(nsim).zfill(5)}.npy"
        return join(fdir, fname)

    def knnauto_path(self, run, nsim=None):
        """
        Path to the `knn` auto-correlation files. If `nsim` is not specified
        returns a list of files for this run for all available simulations.

        Parameters
        ----------
        run : str
            Type of run.
        nsim : int, optional
            IC realisation index.

        Returns
        -------
        path : str
        """
        fdir = join(self.postdir, "knn", "auto")
        if not isdir(fdir):
            makedirs(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        if nsim is not None:
            return join(fdir, f"knncdf_{str(nsim).zfill(5)}_{run}.p")

        files = glob(join(fdir, "knncdf*"))
        run = "__" + run
        return [f for f in files if run in f]

    def knncross_path(self, run, nsims=None):
        """
        Path to the `knn` cross-correlation files. If `nsims` is not specified
        returns a list of files for this run for all available simulations.

        Parameters
        ----------
        run : str
            Type of run.
        nsims : len-2 tuple of int, optional
            IC realisation indices.

        Returns
        -------
        path : str
        """
        fdir = join(self.postdir, "knn", "cross")
        if not isdir(fdir):
            makedirs(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        if nsims is not None:
            assert isinstance(nsims, (list, tuple)) and len(nsims) == 2
            nsim0 = str(nsims[0]).zfill(5)
            nsimx = str(nsims[1]).zfill(5)
            return join(fdir, f"knncdf_{nsim0}_{nsimx}__{run}.p")

        files = glob(join(fdir, "knncdf*"))
        run = "__" + run
        return [f for f in files if run in f]

    def tpcfauto_path(self, run, nsim=None):
        """
        Path to the `tpcf` auto-correlation files. If `nsim` is not specified
        returns a list of files for this run for all available simulations.

        Parameters
        ----------
        run : str
            Type of run.
        nsim : int, optional
            IC realisation index.

        Returns
        -------
        path : str
        """
        fdir = join(self.postdir, "tpcf", "auto")
        if not isdir(fdir):
            makedirs(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        if nsim is not None:
            return join(fdir, f"tpcf{str(nsim).zfill(5)}_{run}.p")

        files = glob(join(fdir, "tpcf*"))
        run = "__" + run
        return [f for f in files if run in f]
