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
from copy import copy, deepcopy

import numpy as np
from scipy.stats import norm
from jax import numpy as jnp
from getdist import MCSamples
from h5py import File

import csiborgtools


###############################################################################
#                       Convert between coordinate systems                    #
###############################################################################


def cartesian_to_radec(x, y, z):
    d = (x**2 + y**2 + z**2)**0.5
    dec = np.arcsin(z / d)
    ra = np.arctan2(y, x)
    ra[ra < 0] += 2 * np.pi

    ra *= 180 / np.pi
    dec *= 180 / np.pi

    return d, ra, dec


###############################################################################
#                          Convert names to LaTeX                             #
###############################################################################


def names_to_latex(names, for_corner=False):
    """Convert the names of the parameters to LaTeX."""
    ltx = {"alpha": "\\alpha",
           "beta": "\\beta",
           "Vmag": "V_{\\rm ext} ~ [\\mathrm{km} / \\mathrm{s}]",
           "Vx": "V_x ~ [\\mathrm{km} / \\mathrm{s}]",
           "Vy": "V_y ~ [\\mathrm{km} / \\mathrm{s}]",
           "Vz": "V_z ~ [\\mathrm{km} / \\mathrm{s}]",
           "sigma_v": "\\sigma_v ~ [\\mathrm{km} / \\mathrm{s}]",
           "alpha_cal": "\\mathcal{A}",
           "beta_cal": "\\mathcal{B}",
           "mag_cal": "\\mathcal{M}",
           # "l": "V_{\\rm ext}^{\\ell} ~ [\\mathrm{deg}]",
           # "b": "V_{\\rm ext}^{b} ~ [\\mathrm{deg}]",
           "l": "\\ell ~ [\\mathrm{deg}]",
           "b": "b ~ [\\mathrm{deg}]",
           "rLG": "R_{\\rm offset} ~ [\\mathrm{Mpc}]",
           "rLG_deterministic": "R_{\\rm offset} ~ [\\mathrm{Mpc}]",
           "Vext_axis_mag": "V_{\\rm axis} ~ [\\mathrm{km} / \\mathrm{s}]",
           "Vvoid": "V_{\\rm void} ~ [\\mathrm{km} / \\mathrm{s}]",
           "void_size": "r_{\\rm void}",
           "hubble": "h",
           "mag_dipole_mag": "\\Delta m",
           "mag_dipole_l": "\\ell_{\\Delta m} ~ [\\mathrm{deg}]",
           "mag_dipole_b": "b_{\\Delta m} ~ [\\mathrm{deg}]",
           }

    ltx_corner = {"alpha": r"$\alpha$",
                  "beta": r"$\beta$",
                  "Vmag": r"$V_{\mathrm{ext}}$",
                  # "l": r"$V_{\mathrm{ext},\ell}$",
                  # "b": r"$V_{\mathrm{ext},b}$",
                  "l": r"$\ell$",
                  "b": r"$b$",
                  "sigma_v": r"$\sigma_v$",
                  "alpha_cal": r"$\mathcal{A}$",
                  "beta_cal": r"$\mathcal{B}$",
                  "mag_cal": r"$\mathcal{M}$",
                  "Vvoid": r"$V_{\rm void}$",
                  "hubble": r"$h$",
                  "rLG": r"$R_{\rm offset}$",
                  "void_size": r"$r_{\rm void}$",
                  "aTFR": r"$a_{\rm TFR}$",
                  "bTFR": r"$b_{\rm TFR}$",
                  "cTFR": r"$c_{\rm TFR}$",
                  "mag_dipole_mag": r"$\Delta m$",
                  "mag_dipole_l": r"$\ell_{\Delta m}$",
                  "mag_dipole_b": r"$b_{\Delta m}$",
                  }

    names = copy(names)
    for i, name in enumerate(names):
        if "SFI_gals" in name:
            names[i] = names[i].replace("SFI_gals", "SFI")

        if "CF4_GroupAll" in name:
            names[i] = names[i].replace("CF4_GroupAll", "CF4Group")

        if "CF4_TFR_i" in name:
            names[i] = names[i].replace("CF4_TFR_i", "CF4,i")

        if "CF4_TFR_w1" in name:
            names[i] = names[i].replace("CF4_TFR_w1", "CF4,W1")

        if "CF4_TFR_w2" in name:
            names[i] = names[i].replace("CF4_TFR_w2", "CF4,W2")

        if "CF4_TFR_notSDSS_w1" in name:
            names[i] = names[i].replace("CF4_TFR_notSDSS_w1", "CF4,W1")

        if "IndranilVoidTFRMock_" in name:
            for n in range(20):
                name_test = f"_IndranilVoidTFRMock_{n}"
                if name_test in name:
                    names[i] = names[i].replace(name_test, "")

    for cat in ["2MTF", "SFI", "CF4,i", "CF4,W2", "CF4,W1"]:
        ltx[f"aTFR_{cat}"] = f"a_{{\\rm TFR}}^{{\\rm {cat}}}"
        ltx[f"bTFR_{cat}"] = f"b_{{\\rm TFR}}^{{\\rm {cat}}}"
        ltx[f"cTFR_{cat}"] = f"c_{{\\rm TFR}}^{{\\rm {cat}}}"
        ltx[f"alpha_{cat}"] = f"\\alpha^{{\\rm {cat}}}"
        ltx[f"corr_mag_eta_{cat}"] = f"\\rho_{{m,\\eta}}^{{\\rm {cat}}}"
        ltx[f"eta_mean_{cat}"] = f"\\widehat{{\\eta}}^{{\\rm {cat}}}"
        ltx[f"eta_std_{cat}"] = f"\\widehat{{\\sigma}}_\\eta^{{\\rm {cat}}}"
        ltx[f"mag_mean_{cat}"] = f"\\widehat{{m}}^{{\\rm {cat}}}"
        ltx[f"mag_std_{cat}"] = f"\\widehat{{\\sigma}}_m^{{\\rm {cat}}}"

        ltx_corner[f"aTFR_{cat}"] = rf"$a_{{\rm TFR}}^{{\rm {cat}}}$"
        ltx_corner[f"bTFR_{cat}"] = rf"$b_{{\rm TFR}}^{{\rm {cat}}}$"
        ltx_corner[f"cTFR_{cat}"] = rf"$c_{{\rm TFR}}^{{\rm {cat}}}$"
        ltx_corner[f"alpha_{cat}"] = rf"$\alpha^{{\rm {cat}}}$"
        ltx_corner[f"corr_mag_eta_{cat}"] = rf"$\rho_{{m,\eta}}^{{\rm {cat}}}$"
        ltx_corner[f"eta_mean_{cat}"] = rf"$\widehat{{\eta}}^{{\rm {cat}}}$"
        ltx_corner[f"eta_std_{cat}"] = rf"$\widehat{{\sigma}}_\eta^{{\rm {cat}}}$"  # noqa
        ltx_corner[f"mag_mean_{cat}"] = rf"$\widehat{{m}}^{{\rm {cat}}}$"
        ltx_corner[f"mag_std_{cat}"] = rf"$\widehat{{\sigma}}_m^{{\rm {cat}}}$"
        ltx_corner[f"aTFR_dipole_{cat}_mag"] = fr"$\tilde{{a}}_{{\rm mag}}^{{\rm {cat}}}$"    # noqa
        ltx_corner[f"aTFR_dipole_{cat}_l"] = fr"$\tilde{{a}}_{{\ell}}^{{\rm {cat}}}$"         # noqa
        ltx_corner[f"aTFR_dipole_{cat}_b"] = fr"$\tilde{{a}}_{{b}}^{{\rm {cat}}}$"            # noqa

    for cat in ["2MTF", "SFI", "Foundation", "LOSS", "CF4Group", "CF4_TFR_w1",
                "CF4_TFR_w2"]:
        ltx[f"alpha_{cat}"] = f"\\alpha^{{\\rm {cat}}}"
        ltx[f"e_mu_{cat}"] = f"\\sigma_{{\\mu}}^{{\\rm {cat}}}"

        ltx_corner[f"alpha_{cat}"] = rf"$\alpha^{{\rm {cat}}}$"
        ltx_corner[f"e_mu_{cat}"] = rf"$\sigma_{{\mu}}^{{\rm {cat}}}$"

    for cat in ["Foundation", "LOSS"]:
        ltx[f"alpha_cal_{cat}"] = f"\\mathcal{{A}}^{{\\rm {cat}}}"
        ltx[f"beta_cal_{cat}"] = f"\\mathcal{{B}}^{{\\rm {cat}}}"
        ltx[f"mag_cal_{cat}"] = f"\\mathcal{{M}}^{{\\rm {cat}}}"

        ltx_corner[f"alpha_cal_{cat}"] = rf"$\mathcal{{A}}^{{\rm {cat}}}$"
        ltx_corner[f"beta_cal_{cat}"] = rf"$\mathcal{{B}}^{{\rm {cat}}}$"
        ltx_corner[f"mag_cal_{cat}"] = rf"$\mathcal{{M}}^{{\rm {cat}}}$"

    for cat in ["CF4Group"]:
        ltx[f"dmu_{cat}"] = f"\\Delta\\mu^{{\\rm {cat}}}"
        ltx[f"dmu_dipole_mag_{cat}"] = f"\\epsilon_\\mu_{{\\rm mag}}^{{\\rm {cat}}}"                  # noqa
        ltx[f"dmu_dipole_l_{cat}"] = f"\\epsilon_\\mu_{{\\ell}}^{{\\rm {cat}}} ~ [\\mathrm{{deg}}]"   # noqa
        ltx[f"dmu_dipole_b_{cat}"] = f"\\epsilon_\\mu_{{b}}^{{\\rm {cat}}} ~ [\\mathrm{{deg}}]"       # noqa

        ltx_corner[f"dmu_{cat}"] = rf"$\Delta\mu_{{0}}^{{\rm {cat}}}$"
        ltx_corner[f"dmu_dipole_mag_{cat}"] = rf"$\epsilon_{{\rm mag}}^{{\rm {cat}}}$"  # noqa
        ltx_corner[f"dmu_dipole_l_{cat}"] = rf"$\epsilon_{{\ell}}^{{\rm {cat}}}$"       # noqa
        ltx_corner[f"dmu_dipole_b_{cat}"] = rf"$\epsilon_{{b}}^{{\rm {cat}}}$"          # noqa

    labels = copy(names)
    for i, label in enumerate(names):
        if for_corner:
            labels[i] = ltx_corner[label] if label in ltx_corner else label
        else:
            labels[i] = ltx[label] if label in ltx else label
    return labels


def simname_to_pretty(simname):
    if "no_field" in simname:
        return "No field"

    ltx = {"Carrick2015": "Carrick+15",
           "Lilow2024": "Lilow+24",
           "csiborg1": r"\texttt{CSiBORG}1",
           "csiborg2_main": r"\texttt{CSiBORG}2",
           "csiborg2X": "Manticore V0",
           "manticore_2MPP_N128_DES_V1": "N128_DES_V1",
           "manticore_2MPP_MULTIBIN_N128_DES_V1": "MULTIBIN_N128_DES_V1",
           "manticore_2MPP_MULTIBIN_N128_DES_V2": "MULTIBIN_N128_DES_V2",
           "manticore_2MPP_MULTIBIN_N256_DES_V2": "MULTIBIN_N256_DES_V2",
           "CF4": "Courtois+23",
           "CF4gp": "CF4group",
           "CLONES": "Sorce+2018",
           "IndranilVoid_exp": "Exponential",
           "IndranilVoid_gauss": "Gaussian",
           "IndranilVoid_mb": "Maxwell-Boltzmann",
           "IndranilVoidSizeVar_exp": "Exponential",
           "IndranilVoidSizeVar_gauss": "Gaussian",
           "IndranilVoidSizeVar_mb": "Maxwell-Boltzmann",
           "no_field": r"$\mathbf{V}_{\rm ext}$ only"
           }

    if isinstance(simname, list):
        names = [ltx[s] if s in ltx else s for s in simname]
        return "".join([f"{n}, " for n in names]).rstrip(", ")

    return ltx[simname] if simname in ltx else simname


def catalogue_to_pretty(catalogue):
    ltx = {"SFI_gals": r"SFI\texttt{++}",
           "CF4_TFR_not2MTForSFI_i": r"CF4 $i$-band",
           "CF4_TFR_i": r"CF4 TFR $i$",
           "CF4_TFR_w1": r"CF4 TFR W1",
           "CF4_TFR_w2": r"CF4 TFR W2",
           }

    if isinstance(catalogue, list):
        names = [ltx[s] if s in ltx else s for s in catalogue]
        return "".join([f"{n}, " for n in names]).rstrip(", ")

    return ltx[catalogue] if catalogue in ltx else catalogue


###############################################################################
#                       Read in goodness-of-fit                               #
###############################################################################

def get_gof(kind, fname):
    """Read in the goodness-of-fit statistics `kind`."""
    if kind not in ["BIC", "AIC", "neg_lnZ_harmonic", "logZ_harmonic"]:
        raise ValueError("`kind` must be one of 'BIC', 'AIC', "
                         "'neg_lnZ_harmonic', 'logZ_harmonic'. "
                         f"Received: `{kind}`.")

    with File(fname, 'r') as f:
        if kind == "logZ_harmonic":
            return -f["gof/neg_lnZ_harmonic"][()] / np.log(10)

        return f[f"gof/{kind}"][()]


###############################################################################
#                           Read in samples                                   #
###############################################################################

def get_samples(fname):
    """Read in the samples from the HDF5 file."""
    samples = {}
    with File(fname, 'r') as f:
        grp = f["samples"]
        for key in grp.keys():
            samples[key] = grp[key][...]

    if "Vext" in samples:
        Vext_mag = samples.pop("Vext_mag")
        Vext_phi = samples.pop("Vext_phi")
        Vext_cos_theta = samples.pop("Vext_cos_theta")

        samples["Vmag"] = Vext_mag
        samples["l"], samples["b"] = csiborgtools.radec_to_galactic(
            np.rad2deg(Vext_phi),
            np.rad2deg(np.pi / 2 - np.arccos(Vext_cos_theta)))

    keys = list(samples.keys())
    for key in keys:

        if "dmu_dipole_" in key:
            dmu = samples.pop(key)

            dmu = csiborgtools.cartesian_to_radec(dmu)
            dmu_mag = dmu[:, 0]
            l, b = csiborgtools.radec_to_galactic(dmu[:, 1], dmu[:, 2])

            samples[key.replace("dmu_dipole_", "dmu_dipole_mag_")] = dmu_mag
            samples[key.replace("dmu_dipole_", "dmu_dipole_l_")] = l
            samples[key.replace("dmu_dipole_", "dmu_dipole_b_")] = b

        if "a_dipole" in key:
            adipole = samples.pop(key)
            adipole = csiborgtools.cartesian_to_radec(adipole)
            adipole_mag = adipole[:, 0]
            l, b = csiborgtools.radec_to_galactic(adipole[:, 1], adipole[:, 2])
            samples[key.replace("a_dipole", "a_dipole_mag")] = adipole_mag
            samples[key.replace("a_dipole", "a_dipole_l")] = l
            samples[key.replace("a_dipole", "a_dipole_b")] = b

    return samples


def get_some_samples(fname, labels):
    """Read in the samples from the HDF5 file."""
    if not isinstance(labels, list) and all(isinstance(label, str) for label in labels):  # noqa
        raise ValueError("`labels` must be a list of strings.")

    samples = {}
    with File(fname, 'r') as f:
        grp = f["samples"]

        if "Vext" in labels:
            samples["Vmag"] = grp["Vext_mag"][...]

            samples["l"], samples["b"] = csiborgtools.radec_to_galactic(
                np.rad2deg(grp["Vext_phi"][...]),
                np.rad2deg(np.pi / 2 - np.arccos(grp["Vext_cos_theta"][...])))

        if "mag_dipole" in labels:
            samples["mag_dipole_mag"] = grp["mag_mag_dipole"][...]
            samples["mag_dipole_l"], samples["mag_dipole_b"] = csiborgtools.radec_to_galactic(  # noqa
                np.rad2deg(grp["phi_mag_dipole"][...]),
                np.rad2deg(np.pi / 2 - np.arccos(grp[f"cos_theta_mag_dipole"][...])))           # noqa

        for label in labels:
            if "Vext" in label:
                continue

            if "mag_dipole" in label:
                continue

            for key in grp.keys():
                if "aTFR_dipole" in key and label in key:
                    if "skipZ" not in key:
                        adip = grp[key][...]
                        samples[f"{key}_mag"] = np.linalg.norm(adip, axis=1)
                        a_dipole = csiborgtools.cartesian_to_radec(adip)
                        ldir, bdir = csiborgtools.radec_to_galactic(
                            a_dipole[:, 1], a_dipole[:, 2])
                        samples[f"{key}_l"], samples[f"{key}_b"] = ldir, bdir
                elif label in key:
                    x = grp[key][...]
                    if x.ndim > 1:
                        raise ValueError("All samples must be 1D arrays.")
                    samples[key] = grp[key][...]

    return samples


###############################################################################
#                         Bulk flow plotting                                  #
###############################################################################


def get_bulkflow_simulation(simname, convert_to_galactic=True):
    f = np.load(f"/mnt/extraspace/rstiskalek/csiborg_postprocessing/field_shells/enclosed_mass_{simname}.npz")  # noqa
    r, B = f["distances"], f["cumulative_velocity"]

    if convert_to_galactic:
        Bmag, Bl, Bb = cartesian_to_radec(B[..., 0], B[..., 1], B[..., 2])
        Bl, Bb = csiborgtools.radec_to_galactic(Bl, Bb)
        B = np.stack([Bmag, Bl, Bb], axis=-1)

    return r, B


def get_bulkflow(fname, simname, convert_to_galactic=True, downsample=1,
                 Rmax=125):
    # Read in the samples
    with File(fname, "r") as f:
        grp = f["samples"]

        # This should still be double-checked.
        Vext = csiborgtools.radec_to_cartesian(np.asarray([
            grp["Vext_mag"][...],
            grp["Vext_phi"][...],
            np.rad2deg(np.pi / 2 - np.arccos(grp["Vext_cos_theta"][...]))]).T)

        try:
            beta = grp["beta"][...]
        except KeyError:
            beta = jnp.ones(len(Vext))

        sigma_v = grp["sigma_v"][...]

    # Read in the bulk flow
    f = np.load(f"/mnt/extraspace/rstiskalek/csiborg_postprocessing/field_shells/enclosed_mass_{simname}.npz")  # noqa
    r = f["distances"]

    # Shape of B_i is (nsims, nradial)
    Bx, By, Bz = (f["cumulative_velocity"][..., i] for i in range(3))

    # Mask out the unconstrained large scales
    Rmax = Rmax  # Mpc/h
    mask = r < Rmax
    r = r[mask]
    Bx = Bx[:, mask]
    By = By[:, mask]
    Bz = Bz[:, mask]

    Vext = Vext[::downsample]
    beta = beta[::downsample]

    # Multiply the simulation velocities by beta.
    Bx = Bx[..., None] * beta
    By = By[..., None] * beta
    Bz = Bz[..., None] * beta

    # Add V_ext, shape of B_i is `(nsims, nradial, nsamples)``
    Bx = Bx + Vext[:, 0]
    By = By + Vext[:, 1]
    Bz = Bz + Vext[:, 2]

    Bcart = np.stack([Bx, By, Bz], axis=-1)

    # Bulk flow in Cartesian coordinates at the origin, `(nsims, nsamples, 3)`.
    # We need to find the first finite point in radial distance.
    k = np.where(np.isfinite(Bcart[0, :, 0, 0]))[0][0]
    Bcart_origin = Bcart[:, k, ...]

    # Add sigma_v scatter to it
    nsim, nsample, __ = Bcart_origin.shape
    for i in range(nsample):
        Bcart_origin[:, i, :] += norm(0, sigma_v[i]).rvs(size=(nsim, 3))

    if convert_to_galactic:
        Bmag, Bl, Bb = cartesian_to_radec(Bx, By, Bz)
        Bl, Bb = csiborgtools.radec_to_galactic(Bl, Bb)
        B = np.stack([Bmag, Bl, Bb], axis=-1)

        Bmag, Bl, Bb = cartesian_to_radec(
            Bcart_origin[..., 0], Bcart_origin[..., 1], Bcart_origin[..., 2])
        Bl, Bb = csiborgtools.radec_to_galactic(Bl, Bb)
        Borigin = np.stack([Bmag, Bl, Bb], axis=-1)[0, ...]
    else:
        B = Bcart
        Borigin = Bcart_origin

    # Stack over the simulations
    B = np.hstack([B[i] for i in range(len(B))])
    return r, B, Borigin

###############################################################################
#                      Prepare samples for plotting                           #
###############################################################################


def samples_for_corner(samples):
    samples = deepcopy(samples)

    # Remove the true parameters of each galaxy.
    keys = list(samples.keys())
    for key in keys:
        # Generally don't want to plot the true latent parameters..
        if "x_TFR" in key or "_true_" in key:
            samples.pop(key)

    keys = list(samples.keys())

    if any(x.ndim > 1 for x in samples.values()):
        raise ValueError("All samples must be 1D arrays.")

    data = np.vstack([x for x in samples.values()]).T
    labels = names_to_latex(list(samples.keys()), for_corner=True)

    return data, labels, keys


def samples_to_getdist(samples, label, ranges=None, settings={}):
    data, __, keys = samples_for_corner(samples)

    return MCSamples(
        samples=data, names=keys,
        labels=names_to_latex(keys, for_corner=False),
        label=label,
        ranges=ranges,
        settings=settings,
        )
