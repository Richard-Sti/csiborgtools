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
Utility functions for scripts.
"""
from datetime import datetime

import numpy

from tqdm import tqdm

try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools


###############################################################################
#                           Reading functions                                 #
###############################################################################


def get_nsims(args, paths):
    """
    Get simulation indices from the command line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments. Must include `nsims` and `simname`. If `nsims`
        is `None` or `-1`, all simulations in `simname` are used.
    paths : :py:class`csiborgtools.paths.Paths`
        Paths object.

    Returns
    -------
    nsims : list of int
        Simulation indices.
    """
    if args.nsims is None or args.nsims[0] == -1:
        nsims = paths.get_ics(args.simname)
    else:
        nsims = args.nsims
    return list(nsims)


def read_single_catalogue(args, config, nsim, run, rmax, paths, nobs=None):
    """
    Read a single halo catalogue and apply selection criteria to it.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments. Must include `simname`.
    config : dict
        Configuration dictionary.
    nsim : int
        Simulation index.
    run : str
        Run name.
    rmax : float
        Maximum radial distance of the halo catalogue.
    paths : csiborgtools.paths.Paths
        Paths object.
    nobs : int, optional
        Fiducial Quijote observer index.

    Returns
    -------
    cat : csiborgtools.read.HaloCatalogue or csiborgtools.read.QuijoteHaloCatalogue  # noqa
        Halo catalogue with selection criteria applied.
    """
    selection = config.get(run, None)
    if selection is None:
        raise KeyError(f"No configuration for run {run}.")
    # We first read the full catalogue without applying any bounds.
    if args.simname == "csiborg":
        cat = csiborgtools.read.HaloCatalogue(nsim, paths)
    else:
        cat = csiborgtools.read.QuijoteHaloCatalogue(nsim, paths, nsnap=4)
        if nobs is not None:
            # We may optionally already here pick a fiducial observer.
            cat = cat.pick_fiducial_observer(nobs, args.Rmax)

    cat.apply_bounds({"dist": (0, rmax)})
    # We then first read off the primary selection bounds.
    sel = selection["primary"]
    pname = None
    xs = sel["name"] if isinstance(sel["name"], list) else [sel["name"]]
    for _name in xs:
        if _name in cat.keys:
            pname = _name
    if pname is None:
        raise KeyError(f"Invalid names `{sel['name']}`.")

    cat.apply_bounds({pname: (sel.get("min", None), sel.get("max", None))})

    # Now the secondary selection bounds. If needed transfrom the secondary
    # property before applying the bounds.
    if "secondary" in selection:
        sel = selection["secondary"]
        sname = None
        xs = sel["name"] if isinstance(sel["name"], list) else [sel["name"]]
        for _name in xs:
            if _name in cat.keys:
                sname = _name
        if sname is None:
            raise KeyError(f"Invalid names `{sel['name']}`.")

        if sel.get("toperm", False):
            cat[sname] = numpy.random.permutation(cat[sname])

        if sel.get("marked", False):
            cat[sname] = csiborgtools.clustering.normalised_marks(
                cat[pname], cat[sname], nbins=config["nbins_marks"])
        cat.apply_bounds({sname: (sel.get("min", None), sel.get("max", None))})

    return cat


def open_catalogues(args, config, paths, comm):
    """
    Read all halo catalogues on the zeroth rank and broadcast them to all
    higher ranks.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments.
    config : dict
        Configuration dictionary.
    paths : csiborgtools.paths.Paths
        Paths object.
    comm : mpi4py.MPI.Comm
        MPI communicator.

    Returns
    -------
    cats : dict
        Dictionary of halo catalogues. Keys are simulation indices, values are
        the catalogues.
    """
    nsims = get_nsims(args, paths)
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    if args.verbose:
        print(f"{datetime.now()}: opening catalogues.", flush=True)

    if rank == 0:
        cats = {}
        if args.simname == "csiborg":
            for nsim in tqdm(nsims) if args.verbose else nsims:
                cat = read_single_catalogue(args, config, nsim, args.run,
                                            rmax=args.Rmax, paths=paths)
                cats.update({nsim: cat})
        else:
            for nsim in tqdm(nsims) if args.verbose else nsims:
                ref_cat = read_single_catalogue(args, config, nsim, args.run,
                                                rmax=None, paths=paths)

                nmax = int(ref_cat.box.boxsize // (2 * args.Rmax))**3
                for nobs in range(nmax):
                    name = paths.quijote_fiducial_nsim(nsim, nobs)
                    cat = ref_cat.pick_fiducial_observer(nobs, rmax=args.Rmax)
                    cats.update({name: cat})

        if nproc > 1:
            for i in range(1, nproc):
                comm.send(cats, dest=i, tag=nproc + i)
    else:
        cats = comm.recv(source=0, tag=nproc + rank)
    return cats


###############################################################################
#                               Clusters                                      #
###############################################################################

_coma = {"RA": (12 + 59 / 60 + 48.7 / 60**2) * 15,
         "DEC": 27 + 58 / 60 + 50 / 60**2,
         "COMDIST": 102.975}

_virgo = {"RA": (12 + 27 / 60) * 15,
          "DEC": 12 + 43 / 60,
          "COMDIST": 16.5}

specific_clusters = {"Coma": _coma, "Virgo": _virgo}

###############################################################################
#                                 Surveys                                     #
###############################################################################


class SDSS:
    @staticmethod
    def steps(cls):
        return [(lambda x: cls[x], ("IN_DR7_LSS",)),
                (lambda x: cls[x] < 17.6, ("ELPETRO_APPMAG_r", )),
                (lambda x: cls[x] < 155, ("DIST", ))
                ]

    def __call__(self):
        return csiborgtools.read.SDSS(h=1, sel_steps=self.steps)
