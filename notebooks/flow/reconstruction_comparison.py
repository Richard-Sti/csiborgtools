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
           "Vmag": "V_{\\rm ext}",
           "sigma_v": "\\sigma_v",
           "alpha_cal": "\\mathcal{A}",
           "beta_cal": "\\mathcal{B}",
           "mag_cal": "\\mathcal{M}",
           }

    ltx_corner = {"alpha": r"$\alpha$",
                  "beta": r"$\beta$",
                  "Vmag": r"$V_{\rm ext}$",
                  "l": r"$\ell$",
                  "b": r"$b$",
                  "sigma_v": r"$\sigma_v$",
                  "alpha_cal": r"$\mathcal{A}$",
                  "beta_cal": r"$\mathcal{B}$",
                  "mag_cal": r"$\mathcal{M}$",
                  }

    names = copy(names)
    for i, name in enumerate(names):
        if "SFI_gals" in name:
            names[i] = names[i].replace("SFI_gals", "SFI")

    for cat in ["2MTF", "SFI"]:
        ltx[f"a_{cat}"] = f"a_{{\\rm TF}}^{{\\rm {cat}}}"
        ltx[f"b_{cat}"] = f"b_{{\\rm TF}}^{{\\rm {cat}}}"
        ltx[f"c_{cat}"] = f"c_{{\\rm TF}}^{{\\rm {cat}}}"
        ltx[f"corr_mag_eta_{cat}"] = f"\\rho_{{m,\\eta}}^{{\\rm {cat}}}"
        ltx[f"eta_mean_{cat}"] = f"\\widehat{{\\eta}}^{{\\rm {cat}}}"
        ltx[f"eta_std_{cat}"] = f"\\widehat{{\\sigma}}_\\eta^{{\\rm {cat}}}"
        ltx[f"mag_mean_{cat}"] = f"\\widehat{{m}}^{{\\rm {cat}}}"
        ltx[f"mag_std_{cat}"] = f"\\widehat{{\\sigma}}_m^{{\\rm {cat}}}"

        ltx_corner[f"a_{cat}"] = rf"$a_{{\rm TF}}^{{\rm {cat}}}$"
        ltx_corner[f"b_{cat}"] = rf"$b_{{\rm TF}}^{{\rm {cat}}}$"
        ltx_corner[f"c_{cat}"] = rf"$c_{{\rm TF}}^{{\rm {cat}}}$"
        ltx_corner[f"corr_mag_eta_{cat}"] = rf"$\rho_{{m,\eta}}^{{\rm {cat}}}$"
        ltx_corner[f"eta_mean_{cat}"] = rf"$\widehat{{\eta}}^{{\rm {cat}}}$"
        ltx_corner[f"eta_std_{cat}"] = rf"$\widehat{{\sigma}}_\eta^{{\rm {cat}}}$"  # noqa
        ltx_corner[f"mag_mean_{cat}"] = rf"$\widehat{{m}}^{{\rm {cat}}}$"
        ltx_corner[f"mag_std_{cat}"] = rf"$\widehat{{\sigma}}_m^{{\rm {cat}}}$"

    for cat in ["2MTF", "SFI", "Foundation", "LOSS"]:
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

    labels = copy(names)
    for i, label in enumerate(names):
        if for_corner:
            labels[i] = ltx_corner[label] if label in ltx_corner else label
        else:
            labels[i] = ltx[label] if label in ltx else label
    return labels


def simname_to_pretty(simname):
    ltx = {"Carrick2015": "Carrick+15",
           "Lilow2024": "Lilow+24",
           "csiborg1": "CB1",
           "csiborg2_main": "CB2",
           "csiborg2X": "Manticore",
           "CF4": "CF4",
           "CF4gp": "CF4group",
           }

    if isinstance(simname, list):
        return [ltx[s] if s in ltx else s for s in simname]

    return ltx[simname] if simname in ltx else simname


def catalogue_to_pretty(catalogue):
    ltx = {"SFI_gals": "SFI",
           "CF4_TFR_not2MTForSFI_i": r"CF4 $i$-band",
           }

    if isinstance(catalogue, list):
        return [ltx[s] if s in ltx else s for s in catalogue]

    return ltx[catalogue] if catalogue in ltx else catalogue


###############################################################################
#                       Read in goodness-of-fit                               #
###############################################################################

def get_gof(kind, fname):
    """Read in the goodness-of-fit statistics `kind`."""
    if kind not in ["BIC", "AIC", "neg_lnZ_harmonic"]:
        raise ValueError("`kind` must be one of 'BIC', 'AIC', 'neg_lnZ_harmonic'.")  # noqa

    with File(fname, 'r') as f:
        return f[f"gof/{kind}"][()]


###############################################################################
#                           Read in samples                                   #
###############################################################################

def get_samples(fname, convert_Vext_to_galactic=True):
    """Read in the samples from the HDF5 file."""
    samples = {}
    with File(fname, 'r') as f:
        grp = f["samples"]
        for key in grp.keys():
            samples[key] = grp[key][...]

    if convert_Vext_to_galactic:
        Vext = samples.pop("Vext")
        samples["Vmag"] = np.linalg.norm(Vext, axis=1)
        Vext = csiborgtools.cartesian_to_radec(Vext)
        samples["l"], samples["b"] = csiborgtools.radec_to_galactic(
            Vext[:, 1], Vext[:, 2])

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


def get_bulkflow(fname, simname, sample_beta=True, convert_to_galactic=True,
                 weight_simulations=True, downsample=1, Rmax=125):
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

    # Read in the samples
    with File(fname, 'r') as f:
        # Shape of Vext_i is (nsamples,)
        Vext_x, Vext_y, Vext_z = (f["samples/Vext"][...][::downsample, i] for i in range(3))  # noqa
        nsamples = len(Vext_x)

        if weight_simulations:
            simulation_weights = jnp.exp(f["simulation_weights"][...])[::downsample]  # noqa
        else:
            nsims = len(Bx)
            simulation_weights = jnp.ones((nsamples, nsims)) / nsims

        if sample_beta:
            beta = f["samples/beta"][...][::downsample]
        else:
            beta = jnp.ones(nsamples)

    # Multiply the simulation velocities by beta
    Bx = Bx[..., None] * beta
    By = By[..., None] * beta
    Bz = Bz[..., None] * beta

    # Shape of B_i is (nsims, nradial, nsamples)
    Bx = Bx + Vext_x
    By = By + Vext_y
    Bz = Bz + Vext_z

    simulation_weights = simulation_weights.T[:, None, :]
    Bx = jnp.sum(Bx * simulation_weights, axis=0)
    By = jnp.sum(By * simulation_weights, axis=0)
    Bz = jnp.sum(Bz * simulation_weights, axis=0)

    if convert_to_galactic:
        Bmag, Bl, Bb = cartesian_to_radec(Bx, By, Bz)
        Bl, Bb = csiborgtools.radec_to_galactic(Bl, Bb)
        B = np.stack([Bmag, Bl, Bb], axis=-1)
    else:
        B = np.stack([Bx, By, Bz], axis=-1)

    return r, B

###############################################################################
#                      Prepare samples for plotting                           #
###############################################################################


def samples_for_corner(samples):
    samples = deepcopy(samples)

    # Remove the true parameters of each galaxy.
    keys = list(samples.keys())
    for key in keys:
        if "x_TFR" in key or "_true_" in key:
            samples.pop(key)

        if "a_dipole_" in key:
            samples.pop(key)

    keys = list(samples.keys())

    if any(x.ndim > 1 for x in samples.values()):
        raise ValueError("All samples must be 1D arrays.")

    data = np.vstack([x for x in samples.values()]).T
    labels = names_to_latex(list(samples.keys()), for_corner=True)

    return data, labels, keys


def samples_to_getdist(samples, label):
    data, __, keys = samples_for_corner(samples)

    return MCSamples(
        samples=data, names=keys,
        labels=names_to_latex(keys, for_corner=False),
        label=label)
