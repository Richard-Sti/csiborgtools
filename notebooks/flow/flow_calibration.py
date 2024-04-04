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
from os.path import join, exists

import numpy as np
from getdist import MCSamples
from h5py import File

import csiborgtools


def read_samples(catalogue, simname, ksmooth, include_calibration=False,
                 return_MCsamples=False, subtract_LG_velocity=-1):
    print(f"\nReading {catalogue} fitted to {simname} with ksmooth = {ksmooth}.", flush=True)  # noqa
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(simname)

    # The last simulation was used to draw the mocks.
    if catalogue in ["CB2_small", "CB2_large"]:
        nsims = nsims[:-5]

    FDIR_LG = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity/observer"  # noqa

    Vx, Vy, Vz, beta, sigma_v, alpha = [], [], [], [], [], []
    BIC, AIC, logZ, chi2 = [], [], [], []

    if catalogue in ["LOSS", "Foundation", "Pantheon+"]:
        alpha_cal, beta_cal, mag_cal, e_mu_intrinsic = [], [], [], []
    elif catalogue in ["2MTF", "SFI_gals", "SFI_gals_masked"]:
        a, b, e_mu_intrinsic = [], [], []
    elif catalogue == "SFI_groups":
        h = []
    elif catalogue in ["CB2_small", "CB2_large"]:
        pass
    else:
        raise ValueError(f"Catalogue {catalogue} not recognized.")

    fname = f"/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity/flow_samples_{catalogue}_{simname}_smooth_{ksmooth}.hdf5"  # noqa
    with File(fname, 'r') as f:
        for i, nsim in enumerate(nsims):
            Vx.append(f[f"sim_{nsim}/Vext_x"][:])
            Vy.append(f[f"sim_{nsim}/Vext_y"][:])
            Vz.append(f[f"sim_{nsim}/Vext_z"][:])

            alpha.append(f[f"sim_{nsim}/alpha"][:])
            beta.append(f[f"sim_{nsim}/beta"][:])
            sigma_v.append(f[f"sim_{nsim}/sigma_v"][:])

            if subtract_LG_velocity >= 0:
                fname = join(FDIR_LG, f"{simname}_{nsim}_observer_velocity.npz")  # noqa
                if not exists(fname):
                    raise FileNotFoundError(f"File {fname} not found.")
                d = np.load(fname)
                R = d["smooth_scales"][subtract_LG_velocity]
                if i == 0:
                    print(f"Subtracting LG velocity with kernel {R} Mpc / h.", flush=True)  # noqa
                Vx_LG, Vy_LG, Vz_LG = d["vobs"][subtract_LG_velocity]
                if simname == "Carrick2015":
                    Vx[-1] += beta[-1] * Vx_LG
                    Vy[-1] += beta[-1] * Vy_LG
                    Vz[-1] += beta[-1] * Vz_LG
                else:
                    Vx[-1] += Vx_LG
                    Vy[-1] += Vy_LG
                    Vz[-1] += Vz_LG

            BIC.append(f[f"sim_{nsim}/BIC"][...])
            AIC.append(f[f"sim_{nsim}/AIC"][...])
            logZ.append(f[f"sim_{nsim}/logZ"][...])
            try:
                chi2.append(f[f"sim_{nsim}/chi2"][...])
            except KeyError:
                chi2.append([0.])

            if catalogue in ["LOSS", "Foundation", "Pantheon+"]:
                alpha_cal.append(f[f"sim_{nsim}/alpha_cal"][:])
                beta_cal.append(f[f"sim_{nsim}/beta_cal"][:])
                mag_cal.append(f[f"sim_{nsim}/mag_cal"][:])
                e_mu_intrinsic.append(f[f"sim_{nsim}/e_mu_intrinsic"][:])
            elif catalogue in ["2MTF", "SFI_gals"]:
                a.append(f[f"sim_{nsim}/a"][:])
                b.append(f[f"sim_{nsim}/b"][:])
                e_mu_intrinsic.append(f[f"sim_{nsim}/e_mu_intrinsic"][:])
            elif catalogue == "SFI_groups":
                h.append(f[f"sim_{nsim}/h"][:])
            elif catalogue in ["CB2_small", "CB2_large"]:
                pass
            else:
                raise ValueError(f"Catalogue {catalogue} not recognized.")

    Vx, Vy, Vz, alpha, beta, sigma_v = np.hstack(Vx), np.hstack(Vy), np.hstack(Vz), np.hstack(alpha), np.hstack(beta), np.hstack(sigma_v)  # noqa

    gof = np.hstack(BIC), np.hstack(AIC), np.hstack(logZ), np.hstack(chi2)

    if catalogue in ["LOSS", "Foundation", "Pantheon+"]:
        alpha_cal, beta_cal, mag_cal, e_mu_intrinsic = np.hstack(alpha_cal), np.hstack(beta_cal), np.hstack(mag_cal), np.hstack(e_mu_intrinsic)  # noqa
    elif catalogue in ["2MTF", "SFI_gals", "SFI_gals_masked"]:
        a, b, e_mu_intrinsic = np.hstack(a), np.hstack(b), np.hstack(e_mu_intrinsic)  # noqa
    elif catalogue == "SFI_groups":
        h = np.hstack(h)
    elif catalogue in ["CB2_small", "CB2_large"]:
        pass
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
        elif catalogue in ["2MTF", "SFI_gals", "SFI_gals_masked"]:
            data += [a, b, e_mu_intrinsic]
            names += ["a", "b", "e_mu_intrinsic"]
        elif catalogue == "SFI_groups":
            data += [h]
            names += ["h"]
        else:
            raise ValueError(f"Catalogue {catalogue} not recognized.")

    print("BIC  = {:4f} +- {:4f}".format(np.mean(gof[0]), np.std(gof[0])))
    print("AIC  = {:4f} +- {:4f}".format(np.mean(gof[1]), np.std(gof[1])))
    print("logZ = {:4f} +- {:4f}".format(np.mean(gof[2]), np.std(gof[2])))
    print("chi2 = {:4f} +- {:4f}".format(np.mean(gof[3]), np.std(gof[3])))

    data = np.vstack(data).T

    if return_MCsamples:
        simname = simname_to_pretty(simname)
        if ksmooth == 1:
            simname = fr"{simname} (2)"

        if subtract_LG_velocity >= 0:
            simname += " (LG)"

        label = fr"{catalogue}, {simname}, $\log \mathcal{{Z}} = {np.mean(gof[2]):.1f}$"                 # noqa

        return MCSamples(samples=data, names=names,
                         labels=names_to_latex(names), label=label)

    return data, names, gof


def simname_to_pretty(simname):
    ltx = {"Carrick2015": "C+15",
           "csiborg1": "CB1",
           "csiborg2_main": "CB2",
           }
    return ltx[simname] if simname in ltx else simname


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
                  "h": r"$h$",
                  }

    labels = copy(names)
    for i, label in enumerate(names):
        if label in ltx:
            labels[i] = ltx_corner[label] if for_corner else ltx[label]

    return labels
