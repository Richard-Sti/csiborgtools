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
from copy import copy

import numpy as np
from getdist import MCSamples
from h5py import File

import csiborgtools


def read_samples(catalogue, simname, ksmooth, include_calibration=False,
                 return_MCsamples=False):
    print(f"\nReading {catalogue} fitted to {simname} with ksmooth = {ksmooth}.", flush=True)  # noqa
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(simname)

    Vx, Vy, Vz, beta, sigma_v, alpha = [], [], [], [], [], []
    BIC, AIC, logZ = [], [], []

    if catalogue in ["LOSS", "Foundation", "Pantheon+"]:
        alpha_cal, beta_cal, mag_cal, e_mu_intrinsic = [], [], [], []
    elif catalogue in ["2MTF", "SFI_gals"]:
        a, b, e_mu_intrinsic = [], [], []
    else:
        raise ValueError(f"Catalogue {catalogue} not recognized.")

    fname = f"/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity/flow_samples_{catalogue}_{simname}_smooth_{ksmooth}.hdf5"  # noqa
    with File(fname, 'r') as f:
        for __, nsim in enumerate(nsims):
            Vx.append(f[f"sim_{nsim}/Vext_x"][:])
            Vy.append(f[f"sim_{nsim}/Vext_y"][:])
            Vz.append(f[f"sim_{nsim}/Vext_z"][:])
            alpha.append(f[f"sim_{nsim}/alpha"][:])
            beta.append(f[f"sim_{nsim}/beta"][:])
            sigma_v.append(f[f"sim_{nsim}/sigma_v"][:])

            BIC.append(f[f"sim_{nsim}/BIC"][...])
            AIC.append(f[f"sim_{nsim}/AIC"][...])
            logZ.append(f[f"sim_{nsim}/logZ"][...])

            if catalogue in ["LOSS", "Foundation", "Pantheon+"]:
                alpha_cal.append(f[f"sim_{nsim}/alpha_cal"][:])
                beta_cal.append(f[f"sim_{nsim}/beta_cal"][:])
                mag_cal.append(f[f"sim_{nsim}/mag_cal"][:])
                e_mu_intrinsic.append(f[f"sim_{nsim}/e_mu_intrinsic"][:])
            elif catalogue in ["2MTF", "SFI_gals"]:
                a.append(f[f"sim_{nsim}/a"][:])
                b.append(f[f"sim_{nsim}/b"][:])
                e_mu_intrinsic.append(f[f"sim_{nsim}/e_mu_intrinsic"][:])
            else:
                raise ValueError(f"Catalogue {catalogue} not recognized.")

    Vx, Vy, Vz, alpha, beta, sigma_v = np.hstack(Vx), np.hstack(Vy), np.hstack(Vz), np.hstack(alpha), np.hstack(beta), np.hstack(sigma_v)  # noqa

    gof = np.hstack(BIC), np.hstack(AIC), np.hstack(logZ)

    if catalogue in ["LOSS", "Foundation", "Pantheon+"]:
        alpha_cal, beta_cal, mag_cal, e_mu_intrinsic = np.hstack(alpha_cal), np.hstack(beta_cal), np.hstack(mag_cal), np.hstack(e_mu_intrinsic)  # noqa
    elif catalogue in ["2MTF", "SFI_gals"]:
        a, b, e_mu_intrinsic = np.hstack(a), np.hstack(b), np.hstack(e_mu_intrinsic)  # noqa
    else:
        raise ValueError(f"Catalogue {catalogue} not recognized.")

    # Calculate magnitude of V_ext
    Vmag = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    # Calculate direction in galactic coordinates of V_ext
    V = np.vstack([Vx, Vy, Vz]).T
    V = csiborgtools.cartesian_to_radec(V)
    l, b = csiborgtools.flow.radec_to_galactic(V[:, 1], V[:, 2])

    data = [alpha, beta, Vmag, l, b, sigma_v]
    names = ["alpha", "beta", "Vmag", "l", "b", "sigma_v"]

    if include_calibration:
        if catalogue in ["LOSS", "Foundation", "Pantheon+"]:
            data += [alpha_cal, beta_cal, mag_cal, e_mu_intrinsic]
            names += ["alpha_cal", "beta_cal", "mag_cal", "e_mu_intrinsic"]
        elif catalogue in ["2MTF", "SFI_gals"]:
            data += [a, b, e_mu_intrinsic]
            names += ["a", "b", "e_mu_intrinsic"]
        else:
            raise ValueError(f"Catalogue {catalogue} not recognized.")

    print("BIC  = {:4f} +- {:4f}".format(np.mean(gof[0]), np.std(gof[0])))
    print("AIC  = {:4f} +- {:4f}".format(np.mean(gof[1]), np.std(gof[1])))
    print("logZ = {:4f} +- {:4f}".format(np.mean(gof[2]), np.std(gof[2])))

    data = np.vstack(data).T

    if return_MCsamples:
        return MCSamples(samples=data, names=names,
                         labels=names_to_latex(names), label=simname)

    return data, names, gof


def print_gof(gof):
    print("Carrick+2015, kappa = 0")
    print("BIC = {:.4f}".format(np.mean(gof[0])))
    print("AIC = {:.4f}".format(np.mean(gof[1])))
    print("logZ = {:.4f}".format(np.mean(gof[2])))
    print()


def names_to_latex(names, for_corner=False):
    ltx = {"alpha": "\\alpha",
           "beta": "\\beta",
           "Vmag": "V_{\\rm ext} ~ [\\mathrm{km} / \\mathrm{s}]",
           "sigma_v": "\\sigma_v ~ [\\mathrm{km} / \\mathrm{s}]",
           }

    ltx_corner = {"alpha": r"$\alpha$",
                  "beta": r"$\beta$",
                  "Vmag": r"$V_{\rm ext}$",
                  "sigma_v": r"$\sigma_v$",
                  }

    labels = copy(names)
    for i, label in enumerate(names):
        if label in ltx:
            labels[i] = ltx_corner[label] if for_corner else ltx[label]

    return labels
