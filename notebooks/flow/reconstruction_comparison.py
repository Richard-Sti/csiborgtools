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
from os.path import join

import numpy as np
from getdist import MCSamples
from h5py import File

import csiborgtools


###############################################################################
#                       Get the filename of the samples                       #
###############################################################################


def get_fname(simname, catalogue, ksmooth=0, nsim=None, sample_beta=True):
    """Get the filename of the HDF5 file containing the posterior samples."""
    FDIR = "/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity/"  # noqa
    fname = join(FDIR, f"samples_{simname}_{catalogue}_ksmooth{ksmooth}.hdf5")

    if nsim is not None:
        fname = fname.replace(".hdf5", f"_nsim{nsim}.hdf5")

    if sample_beta:
        fname = fname.replace(".hdf5", "_sample_beta.hdf5")

    return fname


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
           "e_mu": "\\sigma_\\mu",
           "aTF": "a_{\\rm TF}",
           "bTF": "b_{\\rm TF}",
           }

    ltx_corner = {"alpha": r"$\alpha$",
                  "beta": r"$\beta$",
                  "Vmag": r"$V_{\rm ext}$",
                  "l": r"$\ell_{V_{\rm ext}}$",
                  "b": r"$b_{V_{\rm ext}}$",
                  "sigma_v": r"$\sigma_v$",
                  "alpha_cal": r"$\mathcal{A}$",
                  "beta_cal": r"$\mathcal{B}$",
                  "mag_cal": r"$\mathcal{M}$",
                  "e_mu": r"$\sigma_\mu$",
                  "aTF": r"$a_{\rm TF}$",
                  "bTF": r"$b_{\rm TF}$",
                  }

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
           }

    if isinstance(simname, list):
        return [ltx[s] if s in ltx else s for s in simname]

    return ltx[simname] if simname in ltx else simname


###############################################################################
#                       Read in goodness-of-fit                               #
###############################################################################

def get_gof(kind, simname, catalogue, ksmooth=0, nsim=None, sample_beta=True):
    """Read in the goodness-of-fit statistics `kind`."""
    if kind not in ["BIC", "AIC", "lnZ"]:
        raise ValueError("`kind` must be one of 'BIC', 'AIC', 'lnZ'")

    fname = get_fname(simname, catalogue, ksmooth, nsim, sample_beta)
    with File(fname, 'r') as f:
        return f[f"gof/{kind}"][()]


###############################################################################
#                       Read in samples                                      #
###############################################################################

def get_samples(simname, catalogue, ksmooth=0, nsim=None, sample_beta=True,
                convert_Vext_to_galactic=True):
    """Read in the samples from the HDF5 file."""
    fname = get_fname(simname, catalogue, ksmooth, nsim, sample_beta)
    samples = {}
    with File(fname, 'r') as f:
        grp = f["samples"]
        for key in grp.keys():
            samples[key] = grp[key][...]

    # Rename TF parameters
    if "a" in samples:
        samples["aTF"] = samples.pop("a")

    if "b" in samples:
        samples["bTF"] = samples.pop("b")

    if convert_Vext_to_galactic:
        Vext = samples.pop("Vext")
        samples["Vmag"] = np.linalg.norm(Vext, axis=1)
        Vext = csiborgtools.cartesian_to_radec(Vext)
        samples["l"], samples["b"] = csiborgtools.radec_to_galactic(
            Vext[:, 1], Vext[:, 2])

    return samples


def samples_for_corner(samples):
    if any(x.ndim > 1 for x in samples.values()):
        raise ValueError("All samples must be 1D arrays.")

    data = np.vstack([x for x in samples.values()]).T
    labels = names_to_latex(list(samples.keys()), for_corner=True)

    return data, labels


def samples_to_getdist(samples, simname, catalogue=None):
    data, __ = samples_for_corner(samples)
    names = list(samples.keys())

    if catalogue is None:
        label = simname_to_pretty(simname)
    else:
        label = catalogue

    return MCSamples(
        samples=data, names=names,
        labels=names_to_latex(names, for_corner=False),
        label=label)
