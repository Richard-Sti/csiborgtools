# Copyright (C) 2024 Richard Stiskalek
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
"""Script to help with plots in `flow_calibration.ipynb`."""
from os.path import exists, join

import csiborgtools
import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

FDIR = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/field_shells"


def read_enclosed_density(simname):
    fname = join(FDIR, f"enclosed_mass_{simname}.npz")

    if exists(fname):
        data = np.load(fname)
    else:
        raise FileNotFoundError(f"File `{fname}` not found.")

    Om0 = csiborgtools.simname2Omega_m(simname)
    cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
    rho_matter = Om0 * cosmo.critical_density(0).to(u.M_sun / u.Mpc**3).value

    r = data["distances"]
    volume = 4 * np.pi / 3 * r**3

    overdensity = data["enclosed_mass"] / volume / rho_matter - 1

    return r, overdensity


def read_enclosed_flow(simname):
    fname = join(FDIR, f"enclosed_mass_{simname}.npz")

    if exists(fname):
        data = np.load(fname)
    else:
        raise FileNotFoundError(f"File {fname} not found.")

    r = data["distances"]
    V = data["cumulative_velocity"]
    nsim, nbin = V.shape[:2]
    Vmag = np.linalg.norm(V, axis=-1)
    l = np.empty((nsim, nbin), dtype=V.dtype)  # noqa
    b = np.empty_like(l)

    for n in range(nsim):
        V_n = csiborgtools.cartesian_to_radec(V[n])
        l[n], b[n] = csiborgtools.flow.radec_to_galactic(V_n[:, 1], V_n[:, 2])

    return r, Vmag, l, b
