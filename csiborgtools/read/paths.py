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
"""
CSiBORG paths manager.
"""
from glob import glob
from os import makedirs
from os.path import basename, isdir, join
from warnings import warn

import numpy


def check_directory(path):
    if not isdir(path):
        raise IOError(f"Invalid directory `{path}`!")


def try_create_directory(fdir):
    if not isdir(fdir):
        makedirs(fdir)
        warn(f"Created directory `{fdir}`.")


class Paths:
    """
    Paths manager for CSiBORG and Quijote simulations.

    Parameters
    ----------
    # HERE EDIT EVERYTHING
    srcdir : str, optional
        Path to the folder where the RAMSES outputs are stored.
    postdir: str, optional
        Path to the folder where post-processed files are stored.
    borg_dir : str, optional
        Path to the folder where BORG MCMC chains are stored.
    quiote_dir : str, optional
        Path to the folder where Quijote simulations are stored.
    """
    def __init__(self,
                 csiborg1_srcdir=None,
                 csiborg2_main_srcdir=None,
                 csiborg2_random_srcdir=None,
                 csiborg2_varysmall_srcdir=None,
                 postdir=None,
                 quijote_dir=None
                 ):
        self.csiborg1_srcdir = csiborg1_srcdir
        self.csiborg2_main_srcdir = csiborg2_main_srcdir
        self.csiborg2_random_srcdir = csiborg2_random_srcdir
        self.csiborg2_varysmall_srcdir = csiborg2_varysmall_srcdir
        self.quijote_dir = quijote_dir

        self.postdir = postdir

    def get_ics(self, simname, from_quijote_backup=False):
        """
        Get available IC realisation IDs for a given simulation.

        Parameters
        ----------
        simname : str
            Simulation name.
        from_quijote_backup : bool, optional
            Whether to return the ICs from the Quijote backup.

        Returns
        -------
        ids : 1-dimensional array
        """
        if simname == "csiborg":
            files = glob(join(self.csiborg1_srcdir, "chain_*"))
            files = [int(basename(f).replace("chain_", "") for f in files)]
        elif simname == "csiborg2_main":
            files = glob(join(self.csiborg2_main_srcdir, "chain_*"))
            files = [int(basename(f).replace("chain_", "") for f in files)]
        elif simname == "csiborg2_random":
            raise NotImplementedError("TODO")
        elif simname == "csiborg2_varysmall":
            raise NotImplementedError("TODO")
        elif simname == "quijote" or simname == "quijote_full":
            if from_quijote_backup:
                files = glob(join(self.quijote_dir, "halos_backup", "*"))
            else:
                warn(("Taking only the snapshots that also have "
                     "a FoF catalogue!"))
                files = glob(join(self.quijote_dir, "Snapshots_fiducial", "*"))
            files = [int(f.split("/")[-1]) for f in files]
            files = [f for f in files if f < 100]
        else:
            raise ValueError(f"Unknown simulation name `{simname}`.")

        return numpy.sort(files)

    def snapshots(self, nsim, simname):
        if simname == "csiborg":
            fname = "ramses_out_{}"
            if tonew:
                fname += "_new"
                return join(self.postdir, "output", fname.format(nsim))
            return join(self.csiborg1_srcdir, fname.format(nsim))
        elif simname == "csiborg2_main":
            return join(self.csiborg2_main_srcdir, f"chain_{nsim}", "output")
        elif simname == "csiborg2_random":
            raise NotImplementedError("TODO")
        elif simname == "csiborg2_varysmall":
            raise NotImplementedError("TODO")
        elif simname == "quijote":
            return join(self.quijote_dir, "Snapshots_fiducial", str(nsim))
        else:
            raise ValueError(f"Unknown simulation name `{simname}`.")

    def snapshot(self, nsnap, nsim, simname):
        """
        Path to an IC realisation snapshot.

        Parameters
        ----------
        nsnap : int
            Snapshot index. For Quijote, `-1` indicates the IC snapshot.
        nsim : inlot
            IC realisation index.
        simname : str
            Simulation name.

        Returns
        -------
        str
        """
        if simname == "csiborg":
            return join(self.csiborg1_srcdir, f"chain_{nsim}",
                        f"snapshot_{str(nsnap).zfill(5)}")
        elif simname == "csiborg2_main":
            return join(self.csiborg1_srcdir, f"chain_{nsim}",
                        f"snapshot_{str(nsnap).zfill(5)}")
        elif simname == "csiborg2_random":
            raise NotImplementedError("TODO")
        elif simname == "csiborg2_varysmall":
            raise NotImplementedError("TODO")
        elif simname == "quijote":
            raise NotImplementedError("TODO")
        else:
            raise ValueError(f"Unknown simulation name `{simname}`.")


        # simpath = self.snapshots(nsim, simname, tonew=nsnap == 1)
        # if simname == "csiborg":
        #     return join(simpath, f"output_{str(nsnap).zfill(5)}")
        # else:
        #     if nsnap == -1:
        #         return join(simpath, "ICs", "ics")
        #     nsnap = str(nsnap).zfill(3)
        #     return join(simpath, f"snapdir_{nsnap}", f"snap_{nsnap}")

    def get_snapshots(self, nsim, simname):
        """
        List of available snapshots of simulation.

        Parameters
        ----------
        nsim : int
            IC realisation index.
        simname : str
            Simulation name.

        Returns
        -------
        snapshots : 1-dimensional array
        """
        simpath = self.snapshots(nsim, simname, tonew=False)

        if simname == "csiborg":
            # Get all files in simpath that start with output_
            snaps = glob(join(simpath, "output_*"))
            # Take just the last _00XXXX from each file  and strip zeros
            snaps = [int(snap.split("_")[-1].lstrip("0")) for snap in snaps]
        elif simname == "csiborg2_main":
            snaps = glob(join(simpath, "snapshot_*"))
            snaps = [basename(snap) for snap in snaps]
            snaps = [int(snap.split("_")[1]) for snap in snaps]
        elif simname == "csiborg2_random":
            raise NotImplementedError("TODO")
        elif simname == "csiborg2_varysmall":
            raise NotImplementedError("TODO")
        elif simname == "quijote":
            snaps = glob(join(simpath, "snapdir_*"))
            snaps = [int(snap.split("/")[-1].split("snapdir_")[-1])
                     for snap in snaps]
        else:
            raise ValueError(f"Unknown simulation name `{simname}`.")
        return numpy.sort(snaps)

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
        str
        """
        if nobs is None:
            assert isinstance(nsim, str)
            assert len(nsim) == 5
            return nsim
        return f"{str(nobs).zfill(2)}{str(nsim).zfill(3)}"

    def overlap(self, simname, nsim0, nsimx, min_logmass, smoothed):
        """
        Path to the overlap files between two CSiBORG simulations.

        Parameters
        ----------
        simname : str
            Simulation name. Must be one of `csiborg` or `quijote`.
        nsim0 : int
            IC realisation index of the first simulation.
        nsimx : int
            IC realisation index of the second simulation.
        min_logmass : float
            Minimum log mass of halos to consider.
        smoothed : bool
            Whether the overlap is smoothed or not.

        Returns
        -------
        str
        """
        if simname == "csiborg":
            fdir = join(self.postdir, "overlap")
        elif simname == "quijote":
            fdir = join(self.quijote_dir, "overlap")
        else:
            raise ValueError(f"Unknown simulation name `{simname}`.")

        try_create_directory(fdir)

        nsim0 = str(nsim0).zfill(5)
        nsimx = str(nsimx).zfill(5)
        min_logmass = float('%.4g' % min_logmass)

        fname = f"overlap_{nsim0}_{nsimx}_{min_logmass}.npz"
        if smoothed:
            fname = fname.replace("overlap", "overlap_smoothed")
        return join(fdir, fname)

    def match_max(self, simname, nsim0, nsimx, min_logmass, mult):
        """
        Path to the files containing matching based on [1].

        Parameters
        ----------
        simname : str
            Simulation name.
        nsim0 : int
            IC realisation index of the first simulation.
        nsimx : int
            IC realisation index of the second simulation.
        min_logmass : float
            Minimum log mass of halos to consider.
        mult : float
            Multiplicative search radius factor.

        Returns
        -------
        str

        References
        ----------
        [1] Maxwell L Hutt, Harry Desmond, Julien Devriendt, Adrianne Slyz; The
        effect of local Universe constraints on halo abundance and clustering;
        Monthly Notices of the Royal Astronomical Society, Volume 516, Issue 3,
        November 2022, Pages 3592–3601, https://doi.org/10.1093/mnras/stac2407
        """
        if simname == "csiborg":
            fdir = join(self.postdir, "match_max")
        elif simname == "quijote":
            fdir = join(self.quijote_dir, "match_max")
        else:
            ValueError(f"Unknown simulation name `{simname}`.")

        try_create_directory(fdir)

        nsim0 = str(nsim0).zfill(5)
        nsimx = str(nsimx).zfill(5)
        min_logmass = float('%.4g' % min_logmass)
        fname = f"match_max_{nsim0}_{nsimx}_{min_logmass}_{str(mult)}.npz"

        return join(fdir, fname)

    def field(self, kind, MAS, grid, nsim, in_rsp, smooth_scale=None):
        r"""
        Path to the files containing the calculated fields in CSiBORG.

        Parameters
        ----------
        kind : str
            Field type. Must be one of: `density`, `velocity`, `potential`,
            `radvel`, `environment`.
        MAS : str
           Mass-assignment scheme.
        grid : int
            Grid size.
        nsim : int
            IC realisation index.
        in_rsp : bool
            Whether the calculation is performed in redshift space.
        smooth_scale : float, optional
            Smoothing scale in Mpc/h.

        Returns
        -------
        str
        """
        assert kind in ["density", "velocity", "potential", "radvel",
                        "environment"]
        fdir = join(self.postdir, "environment")

        try_create_directory(fdir)

        if in_rsp:
            kind = kind + "_rsp"

        fname = f"{kind}_{MAS}_{str(nsim).zfill(5)}_grid{grid}.npy"

        if smooth_scale is not None:
            fname = fname.replace(".npy", f"_smooth{smooth_scale}.npy")

        return join(fdir, fname)

    def field_interpolated(self, survey, kind, MAS, grid, nsim, in_rsp,
                           smooth_scale=None):
        """
        Path to the files containing the CSiBORG interpolated field for a given
        survey.

        Parameters
        ----------
        survey : str
            Survey name.
        kind : str
            Field type. Must be one of: `density`, `velocity`, `potential`,
            `radvel`, `environment`.
        MAS : str
           Mass-assignment scheme.
        grid : int
            Grid size.
        nsim : int
            IC realisation index.
        in_rsp : bool
            Whether the calculation is performed in redshift space.
        smooth_scale : float, optional
            Smoothing scale in Mpc/h.

        Returns
        -------
        str
        """
        assert kind in ["density", "velocity", "potential", "radvel",
                        "environment"]
        fdir = join(self.postdir, "environment_interpolated")

        try_create_directory(fdir)

        if in_rsp:
            kind = kind + "_rsp"

        fname = f"{survey}_{kind}_{MAS}_{str(nsim).zfill(5)}_grid{grid}.npz"

        if smooth_scale is not None:
            fname = fname.replace(".npz", f"_smooth{smooth_scale}.npz")

        return join(fdir, fname)

    def observer_peculiar_velocity(self, MAS, grid, nsim):
        """
        Path to the files containing the observer peculiar velocity.

        Parameters
        ----------
        MAS : str
           Mass-assignment scheme.
        grid : int
            Grid size.
        nsim : int
            IC realisation index.

        Returns
        -------
        str
        """
        fdir = join(self.postdir, "environment")
        try_create_directory(fdir)

        fname = f"obs_vp_{MAS}_{str(nsim).zfill(5)}_{grid}.npz"
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
            Whether raw distances or counts in bins. Must be one of `dist`,
            `bin_dist` or `tot_counts`.
        nsim : int, optional
            IC realisation index.
        nobs : int, optional
            Fiducial observer index in Quijote simulations.

        Returns
        -------
        str
        """
        assert simname in ["csiborg", "quijote"]
        assert kind in ["dist", "bin_dist", "tot_counts"]
        fdir = join(self.postdir, "nearest_neighbour")
        try_create_directory(fdir)

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
        str
        """
        assert simname in ["csiborg", "quijote"]
        fdir = join(self.postdir, "knn", "auto")
        try_create_directory(fdir)

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
        str
        """
        fdir = join(self.postdir, "knn", "cross")
        try_create_directory(fdir)

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
        str
        """
        fdir = join(self.postdir, "tpcf", "auto")
        try_create_directory(fdir)

        if nsim is not None:
            return join(fdir, f"{simname}_tpcf{str(nsim).zfill(5)}_{run}.p")

        files = glob(join(fdir, f"{simname}_tpcf*"))
        run = "__" + run
        return [f for f in files if run in f]
