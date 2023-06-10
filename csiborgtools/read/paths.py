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


class Paths:
    """
    Paths manager for CSiBORG and Quijote simulations.

    Parameters
    ----------
    srcdir : str, optional
        Path to the folder where the RAMSES outputs are stored.
    postdir: str, optional
        Path to the folder where post-processed files are stored.
    quiote_dir : str, optional
        Path to the folder where Quijote simulations are stored.
    """
    _srcdir = None
    _postdir = None
    _quijote_dir = None

    def __init__(self, srcdir=None, postdir=None, quijote_dir=None):
        self.srcdir = srcdir
        self.postdir = postdir
        self.quijote_dir = quijote_dir

    @staticmethod
    def _check_directory(path):
        if not isdir(path):
            raise IOError(f"Invalid directory `{path}`!")

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
    def quijote_dir(self):
        """
        Path to the folder where Quijote simulations are stored.

        Returns
        -------
        path : str
        """
        if self._quijote_dir is None:
            raise ValueError("`quijote_dir` is not set!")
        return self._quijote_dir

    @quijote_dir.setter
    def quijote_dir(self, path):
        if path is None:
            return
        self._check_directory(path)
        self._quijote_dir = path

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

    @staticmethod
    def quijote_fiducial_nsim(nsim, nobs=None):
        """
        Fiducial Quijote simulation ID. Combines the IC realisation and
        observer placement.

        Parameters
        ----------
        nsim : int
            IC realisation index.
        nobs : int, optional
            Fiducial observer index.

        Returns
        -------
        id : str
        """
        if nobs is None:
            assert isinstance(nsim, str)
            assert len(nsim) == 5
            return nsim
        return f"{str(nobs).zfill(2)}{str(nsim).zfill(3)}"

    def mmain(self, nsnap, nsim):
        """
        Path to the `mmain` CSiBORG files of summed substructure.

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

    def initmatch(self, nsim, kind):
        """
        Path to the `initmatch` files where the halo match between the
        initial and final snapshot of a CSiBORG realisaiton is stored.

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

    def get_ics(self, simname):
        """
        Get available IC realisation IDs for either the CSiBORG or Quijote
        simulation suite.

        Parameters
        ----------
        simname : str
            Simulation name. Must be one of `["csiborg", "quijote"]`.

        Returns
        -------
        ids : 1-dimensional array
        """
        assert simname in ["csiborg", "quijote"]
        if simname == "csiborg":
            files = glob(join(self.srcdir, "ramses_out*"))
            files = [f.split("/")[-1] for f in files]      # Only file names
            files = [f for f in files if "_inv" not in f]  # Remove inv. ICs
            files = [f for f in files if "_new" not in f]  # Remove _new
            files = [f for f in files if "OLD" not in f]   # Remove _old
            ids = [int(f.split("_")[-1]) for f in files]
            try:
                ids.remove(5511)
            except ValueError:
                pass
            return numpy.sort(ids)
        else:
            return numpy.arange(100, dtype=int)

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
            return join(self.postdir, "output", fname.format(nsim))

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

    def snapshot(self, nsnap, nsim):
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

    def structfit(self, nsnap, nsim, kind):
        """
        Path to the clump or halo catalogue from `fit_halos.py`. Only CSiBORG
        is supported.

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

    def overlap(self, nsim0, nsimx, smoothed):
        """
        Path to the overlap files between two CSiBORG simulations.

        Parameters
        ----------
        nsim0 : int
            IC realisation index of the first simulation.
        nsimx : int
            IC realisation index of the second simulation.
        smoothed : bool
            Whether the overlap is smoothed or not.

        Returns
        -------
        path : str
        """
        fdir = join(self.postdir, "overlap")
        if not isdir(fdir):
            mkdir(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        fname = f"overlap_{str(nsim0).zfill(5)}_{str(nsimx).zfill(5)}.npz"
        if smoothed:
            fname = fname.replace("overlap", "overlap_smoothed")
        return join(fdir, fname)

    def particles(self, nsim):
        """
        Path to the files containing all particles of a CSiBORG realisation at
        :math:`z = 0`.

        Parameters
        ----------
        nsim : int
            IC realisation index.

        Returns
        -------
        path : str
        """
        fdir = join(self.postdir, "particles")
        if not isdir(fdir):
            makedirs(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        fname = f"parts_{str(nsim).zfill(5)}.h5"
        return join(fdir, fname)

    def field(self, kind, MAS, grid, nsim, in_rsp):
        """
        Path to the files containing the calculated density fields in CSiBORG.

        Parameters
        ----------
        kind : str
            Field type. Must be one of: `density`, `velocity`, `potential`,
            `radvel`.
        MAS : str
           Mass-assignment scheme.
        grid : int
            Grid size.
        nsim : int
            IC realisation index.
        in_rsp : bool
            Whether the calculation is performed in redshift space.

        Returns
        -------
        path : str
        """
        fdir = join(self.postdir, "environment")
        assert kind in ["density", "velocity", "potential", "radvel"]
        if not isdir(fdir):
            makedirs(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        if in_rsp:
            kind = kind + "_rsp"
        fname = f"{kind}_{MAS}_{str(nsim).zfill(5)}_grid{grid}.npy"
        return join(fdir, fname)

    def halo_counts(self, simname, nsim):
        """
        Path to the files containing the binned halo counts.

        Parameters
        ----------
        simname : str
            Simulation name. Must be `csiborg` or `quijote`.
        nsim : int
            IC realisation index.

        Returns
        -------
        path : str
        """
        fdir = join(self.postdir, "HMF")
        if not isdir(fdir):
            makedirs(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        fname = f"halo_counts_{simname}_{str(nsim).zfill(5)}.npz"
        return join(fdir, fname)

    def cross_nearest(self, simname, run, kind, nsim=None, nobs=None):
        """
        Path to the files containing distance from a halo in a reference
        simulation to the nearest halo from a cross simulation.

        Parameters
        ----------
        simname : str
            Simulation name. Must be one of: `csiborg`, `quijote`.
        run : str
            Run name.
        kind : str
            Whether raw distances or counts in bins. Must be one of `dist` or
            `bin_dist`.
        nsim : int, optional
            IC realisation index.
        nobs : int, optional
            Fiducial observer index in Quijote simulations.

        Returns
        -------
        path : str
        """
        assert simname in ["csiborg", "quijote"]
        assert kind in ["dist", "bin_dist"]
        fdir = join(self.postdir, "nearest_neighbour")
        if not isdir(fdir):
            makedirs(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        if nsim is not None:
            if simname == "csiborg":
                nsim = str(nsim).zfill(5)
            else:
                nsim = self.quijote_fiducial_nsim(nsim, nobs)
            return join(fdir, f"{simname}_nn_{kind}_{nsim}_{run}.npz")

        files = glob(join(fdir, f"{simname}_nn_{kind}_*"))
        run = "_" + run
        return [f for f in files if run in f]

    def knnauto(self, simname, run, nsim=None, nobs=None):
        """
        Path to the `knn` auto-correlation files. If `nsim` is not specified
        returns a list of files for this run for all available simulations.

        Parameters
        ----------
        simname : str
            Simulation name. Must be either `csiborg` or `quijote`.
        run : str
            Run name.
        nsim : int, optional
            IC realisation index.
        nobs : int, optional
            Fiducial observer index in Quijote simulations.

        Returns
        -------
        path : str
        """
        assert simname in ["csiborg", "quijote"]
        fdir = join(self.postdir, "knn", "auto")
        if not isdir(fdir):
            makedirs(fdir)
            warn(f"Created directory `{fdir}`.", UserWarning, stacklevel=1)
        if nsim is not None:
            if simname == "csiborg":
                nsim = str(nsim).zfill(5)
            else:
                nsim = self.quijote_fiducial_nsim(nsim, nobs)
            return join(fdir, f"{simname}_knncdf_{nsim}_{run}.p")

        files = glob(join(fdir, f"{simname}_knncdf*"))
        run = "_" + run
        return [f for f in files if run in f]

    def knncross(self, simname, run, nsims=None):
        """
        Path to the `knn` cross-correlation files. If `nsims` is not specified
        returns a list of files for this run for all available simulations.

        Parameters
        ----------
        simname : str
            Simulation name. Must be either `csiborg` or `quijote`.
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
            return join(fdir, f"{simname}_knncdf_{nsim0}_{nsimx}__{run}.p")

        files = glob(join(fdir, f"{simname}_knncdf*"))
        run = "_" + run
        return [f for f in files if run in f]

    def tpcfauto(self, simname, run, nsim=None):
        """
        Path to the `tpcf` auto-correlation files. If `nsim` is not specified
        returns a list of files for this run for all available simulations.

        Parameters
        ----------
        simname : str
            Simulation name. Must be either `csiborg` or `quijote`.
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
            return join(fdir, f"{simname}_tpcf{str(nsim).zfill(5)}_{run}.p")

        files = glob(join(fdir, f"{simname}_tpcf*"))
        run = "__" + run
        return [f for f in files if run in f]
