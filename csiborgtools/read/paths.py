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
from os.path import isdir, join
from warnings import warn
from re import search

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
    csiborg1_srcdir : str
        Path to the CSiBORG1 simulation directory.
    csiborg2_main_srcdir : str
        Path to the CSiBORG2 main simulation directory.
    csiborg2_random_srcdir : str
        Path to the CSiBORG2 random simulation directory.
    csiborg2_varysmall_srcdir : str
        Path to the CSiBORG2 varysmall simulation directory.
    postdir : str
        Path to the CSiBORG post-processing directory.
    quijote_dir : str
        Path to the Quijote simulation directory.
    borg1_dir : str
        Path to the BORG1 simulation directory.
    borg2_dir : str
        Path to the BORG2 simulation directory.
    tng300_1_dir : str
        Path to the TNG300-1 simulation directory.
    """
    def __init__(self,
                 csiborg1_srcdir,
                 csiborg2_main_srcdir,
                 csiborg2_random_srcdir,
                 csiborg2_varysmall_srcdir,
                 postdir,
                 quijote_dir,
                 borg1_dir,
                 borg2_dir,
                 tng300_1_dir
                 ):
        self.csiborg1_srcdir = csiborg1_srcdir
        self.csiborg2_main_srcdir = csiborg2_main_srcdir
        self.csiborg2_random_srcdir = csiborg2_random_srcdir
        self.csiborg2_varysmall_srcdir = csiborg2_varysmall_srcdir
        self.quijote_dir = quijote_dir
        self.borg1_dir = borg1_dir
        self.borg2_dir = borg2_dir
        self.tng300_1_dir = tng300_1_dir
        self.postdir = postdir

    def get_ics(self, simname):
        """
        Get available IC realisation IDs for a given simulation.

        Parameters
        ----------
        simname : str
            Simulation name.

        Returns
        -------
        ids : 1-dimensional array
        """
        if simname == "csiborg1" or simname == "borg1":
            files = glob(join(self.csiborg1_srcdir, "chain_*"))
            files = [int(search(r'chain_(\d+)', f).group(1)) for f in files]
        elif simname == "csiborg2_main" or simname == "borg2":
            files = glob(join(self.csiborg2_main_srcdir, "chain_*"))
            files = [int(search(r'chain_(\d+)', f).group(1)) for f in files]
        elif simname == "csiborg2_random":
            files = glob(join(self.csiborg2_random_srcdir, "chain_*"))
            files = [int(search(r'chain_(\d+)', f).group(1)) for f in files]
        elif simname == "csiborg2_varysmall":
            files = glob(join(self.csiborg2_varysmall_srcdir, "chain_*"))
            files = [int(search(r'chain_16417_(\d+)', f).group(1))
                     for f in files]
        elif simname == "quijote":
            files = glob(join(self.quijote_dir, "fiducial_processed",
                              "chain_*"))
            files = [int(search(r'chain_(\d+)', f).group(1)) for f in files]
        else:
            raise ValueError(f"Unknown simulation name `{simname}`.")

        return numpy.sort(files)

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
        if simname == "csiborg1":
            snaps = glob(join(self.csiborg1_srcdir, f"chain_{nsim}",
                              "snapshot_*"))
            snaps = [int(search(r'snapshot_(\d+)', f).group(1)) for f in snaps]
            snaps = sorted(snaps)
        elif simname == "csiborg2_main":
            snaps = glob(join(self.csiborg2_main_srcdir, f"chain_{nsim}",
                              "output", "snapshot_*"))
            snaps = [int(search(r'snapshot_(\d+)', f).group(1))
                     for f in snaps]
            snaps = sorted(snaps)
        elif simname == "csiborg2_random":
            snaps = glob(join(self.csiborg2_random_srcdir, f"chain_{nsim}",
                              "output", "snapshot_*"))
            snaps = [int(search(r'snapshot_(\d+)', f).group(1))
                     for f in snaps]
            snaps = sorted(snaps)
        elif simname == "csiborg2_varysmall":
            snaps = glob(join(self.csiborg2_random_srcdir,
                              f"chain_16417_{str(nsim).zfill(3)}",
                              "snapshot_*"))
            snaps = [int(search(r'snapshot_16417_(\d+)', f).group(1))
                     for f in snaps]
            snaps = sorted(snaps)
        elif simname == "quijote":
            snaps = glob(join(self.quijote_dir, "fiducial_processed",
                              f"chain_{nsim}", "snapshot_*"))
            has_ics = any("snapshot_ICs" in f for f in snaps)
            snaps = [int(match.group(1))
                     for f in snaps if (match := search(r'snapshot_(\d+)', f))]
            snaps = sorted(snaps)
            if has_ics:
                snaps.insert(0, "ICs")
        else:
            raise ValueError(f"Unknown simulation name `{simname}`.")
        return snaps

    def snapshot(self, nsnap, nsim, simname):
        """
        Path to a simulation snapshot.

        Parameters
        ----------
        nsnap : int
            Snapshot index. For Quijote, `ICs` indicates the IC snapshot.
        nsim : int
            IC realisation index.
        simname : str
            Simulation name.

        Returns
        -------
        str
        """
        if simname == "csiborg1":
            fpath = join(self.csiborg1_srcdir, f"chain_{nsim}",
                         f"snapshot_{str(nsnap).zfill(5)}.hdf5")
        elif simname == "csiborg2_main":
            fpath = join(self.csiborg2_main_srcdir, f"chain_{nsim}", "output",
                         f"snapshot_{str(nsnap).zfill(3)}.hdf5")
        elif simname == "csiborg2_random":
            fpath = join(self.csiborg2_random_srcdir, f"chain_{nsim}",
                         "output", f"snapshot_{str(nsnap).zfill(3)}.hdf5")
        elif simname == "csiborg2_varysmall":
            fpath = join(self.csiborg2_varysmall_srcdir,
                         f"chain_16417_{str(nsim).zfill(3)}", "output",
                         f"snapshot_{str(nsnap).zfill(3)}.hdf5")
        elif simname == "quijote":
            fpath = join(self.quijote_dir, "fiducial_processed",
                         f"chain_{nsim}",
                         f"snapshot_{str(nsnap).zfill(3)}.hdf5")
        else:
            raise ValueError(f"Unknown simulation name `{simname}`.")

        return fpath

    def snapshot_catalogue(self, nsnap, nsim, simname):
        """
        Path to the halo catalogue of a simulation snapshot.

        Parameters
        ----------
        nsnap : int
            Snapshot index.
        nsim : int
            IC realisation index.
        simname : str
            Simulation name.

        Returns
        -------
        str
        """
        if simname == "csiborg1":
            return join(self.csiborg1_srcdir, f"chain_{nsim}",
                        f"fof_{str(nsnap).zfill(5)}.hdf5")
        elif simname == "csiborg2_main":
            return join(self.csiborg2_main_srcdir, f"chain_{nsim}", "output",
                        f"fof_subhalo_tab_{str(nsnap).zfill(3)}.hdf5")
        elif simname == "csiborg2_random":
            return join(self.csiborg2_random_srcdir, f"chain_{nsim}", "output",
                        f"fof_subhalo_tab_{str(nsnap).zfill(3)}.hdf5")
        elif simname == "csiborg2_varysmall":
            return join(self.csiborg2_varysmall_srcdir,
                        f"chain_16417_{str(nsim).zfill(3)}", "output",
                        f"fof_subhalo_tab_{str(nsnap).zfill(3)}.hdf5")
        elif simname == "quijote":
            return join(self.quijote_dir, "fiducial_processed",
                        f"chain_{nsim}",
                        f"fof_{str(nsnap).zfill(3)}.hdf5")
        else:
            raise ValueError(f"Unknown simulation name `{simname}`.")

    def initial_lagpatch(self, nsim, simname):
        """
        Path to the Lagrangain patch information of a simulation for halos
        defined at z = 0.

        Parameters
        ----------
        nsim : int
            IC realisation index.
        simname : str
            Simulation name.

        Returns
        -------
        str
        """
        if simname == "csiborg1":
            return join(self.csiborg1_srcdir, f"chain_{nsim}",
                        "initial_lagpatch.npy")
        elif simname == "csiborg2_main":
            return join(self.csiborg2_main_srcdir, "catalogues",
                        f"initial_lagpatch_{nsim}.npy")
        elif simname == "csiborg2_random":
            return join(self.csiborg2_random_srcdir, "catalogues",
                        f"initial_lagpatch_{nsim}.npy")
        elif simname == "csiborg2_varysmall":
            return join(self.csiborg2_varysmall_srcdir, "catalogues",
                        f"initial_lagpatch_{nsim}.npy")
        elif simname == "quijote":
            return join(self.quijote_dir, "fiducial_processed",
                        f"chain_{nsim}", "initial_lagpatch.npy")
        else:
            raise ValueError(f"Unknown simulation name `{simname}`.")

    def trees(self, nsim, simname):
        """
        Path to the halo trees of a simulation snapshot.

        Parameters
        ----------
        nsim : int
            IC realisation index.
        simname : str
            Simulation name.

        Returns
        -------
        str
        """
        if simname == "csiborg1":
            raise ValueError("Trees not available for CSiBORG1.")
        elif simname == "csiborg2_main":
            return join(self.csiborg2_main_srcdir, f"chain_{nsim}", "output",
                        "trees.hdf5")
        elif simname == "csiborg2_random":
            return join(self.csiborg2_ranodm_srcdir, f"chain_{nsim}", "output",
                        "trees.hdf5")
        elif simname == "csiborg2_varysmall":
            return join(self.csiborg2_varysmall_srcdir,
                        f"chain_16417_{str(nsim).zfill(3)}", "output",
                        "trees.hdf5")
        elif simname == "quijote":
            raise ValueError("Trees not available for Quijote.")
        else:
            raise ValueError(f"Unknown simulation name `{simname}`.")

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
        if "csiborg" in simname:
            fdir = join(self.postdir, "overlap")
        elif simname == "quijote":
            fdir = join(self.quijote_dir, "overlap")
        else:
            raise ValueError(f"Unknown simulation name `{simname}`.")

        try_create_directory(fdir)

        nsim0 = str(nsim0).zfill(5)
        nsimx = str(nsimx).zfill(5)
        min_logmass = float('%.4g' % min_logmass)

        fname = f"overlap_{simname}_{nsim0}_{nsimx}_{min_logmass}.npz"
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

    def field(self, kind, MAS, grid, nsim, simname):
        r"""
        Path to the files containing the calculated fields in CSiBORG.

        Parameters
        ----------
        kind : str
            Field type.
        MAS : str
           Mass-assignment scheme.
        grid : int
            Grid size.
        nsim : int
            IC realisation index.
        simname : str
            Simulation name.

        Returns
        -------
        str
        """
        if simname == "borg2":
            return join(self.borg2_dir, f"mcmc_{nsim}.h5")

        if simname == "borg1":
            #
            return f"/mnt/zfsusers/hdesmond/BORG_final/mcmc_{nsim}.h5"

        if MAS == "SPH" and kind in ["density", "velocity"]:
            if simname == "csiborg1":
                return join(self.csiborg1_srcdir, "field",
                            f"sph_ramses_{str(nsim).zfill(5)}_{grid}.hdf5")
            elif simname == "csiborg2_main":
                return join(self.csiborg2_main_srcdir, "field",
                            f"chain_{nsim}_{grid}.hdf5")
            elif simname == "csiborg2_random":
                return join(self.csiborg2_random_srcdir, "field",
                            f"chain_{nsim}_{grid}.hdf5")
            elif simname == "csiborg2_varysmall":
                return join(self.csiborg2_varysmall_srcdir, "field",
                            f"chain_{nsim}_{grid}.hdf5")
            elif simname == "quijote":
                raise ValueError("SPH field not available for CSiBORG1.")

        fdir = join(self.postdir, "environment")
        try_create_directory(fdir)

        fname = f"{kind}_{simname}_{MAS}_{str(nsim).zfill(5)}_{grid}.npy"

        return join(fdir, fname)

    def observer_peculiar_velocity(self, MAS, grid, nsim, simname):
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
        simname : str
            Simulation name.

        Returns
        -------
        str
        """
        fdir = join(self.postdir, "environment")
        try_create_directory(fdir)
        fname = f"observer_peculiar_velocity_{simname}_{MAS}_{str(nsim).zfill(5)}_{grid}.npz"  # noqa
        return join(fdir, fname)

    def field_interpolated(self, survey, simname, nsim, kind, MAS, grid,
                           radial_scatter=None):
        """
        Path to the files containing the interpolated field for a given
        survey.

        Parameters
        ----------
        survey : str
            Survey name.
        simname : str
            Simulation name.
        nsim : int
            IC realisation index.
        kind : str
            Field type.
        MAS : str
           Mass-assignment scheme.
        grid : int
            Grid size.
        radial_scatter : float, optional
            Radial scatter added to the galaxy positions, only supported for
            TNG300-1.

        Returns
        -------
        str
        """
        # # In case the galaxy positions of TNG300-1 were scattered..
        if kind not in ["density", "potential", "radvel"]:
            raise ValueError("Unsupported field type.")

        fdir = join(self.postdir, "field_interpolated")
        try_create_directory(fdir)

        nsim = str(nsim).zfill(5)
        fname = join(fdir, f"{survey}_{simname}_{kind}_{MAS}_{nsim}_{grid}.npz")  # noqa

        if simname == "TNG300-1" and radial_scatter is not None:
            fname = fname.replace(".npz", f"_scatter{radial_scatter}.npz")

        return fname

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

    def tng300_1(self):
        """
        Path to the TNG300-1 simulation directory.

        Returns
        -------
        str
        """
        return self.tng300_1_dir
