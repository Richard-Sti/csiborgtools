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
"""Code to sample the chi2 grid from Sergij so that I can have MCMC samples."""

from argparse import ArgumentParser

import numpy as np
from h5py import File
from interpax import Interpolator2D
from jax import random
from numpyro import factor, sample
from numpyro.distributions import Uniform
from numpyro.infer import MCMC, NUTS, init_to_sample


def load_interpolator(kind):
    if kind not in ["exp", "gauss", "mb"]:
        raise ValueError("kind must be one of 'exp', 'gauss', 'mb'")

    kind = kind.upper()

    fpath = f"/mnt/extraspace/rstiskalek/catalogs/IndranilVoid/ChiSq2D_{kind}profileresolution_3501x601.dat"  # noqa
    print(f"Reading the chi2 grid from `{fpath}`.")

    data = np.genfromtxt(fpath)
    nx, ny = data.shape
    xdata = np.arange(0, nx)
    ydata = np.arange(0, ny)

    if kind in ["EXP", "GAUSS"]:
        ymin = 0
        ymax = 100
    else:
        ymin = 100
        ymax = 250

    xmin = 100
    xmax = len(xdata)

    print(f"Selecting only the region where R_LG < {ymax} Mpc to avoid some "
          "unexpected behavior.")

    # We want to interpolate the log-likelihood
    f = Interpolator2D(xdata, ydata, -0.5 * data, method="cubic")

    return f, xmin, xmax, ymin, ymax, fpath


if __name__ == "__main__":
    parser = ArgumentParser(description="Sample the chi2 grid from Sergij.")
    parser.add_argument("kind", type=str, help="The profile kind to sample.",
                        choices=["exp", "gauss", "mb"],)
    args = parser.parse_args()

    nwarm = 5000
    nsamp = 15_000

    f, xmin, xmax, ymin, ymax, fpath = load_interpolator(args.kind)

    print("Sampling the chi2 grid:")
    print(f"xmin = {xmin}, xmax = {xmax}")
    print(f"ymin = {ymin}, ymax = {ymax}")

    def model():
        x = sample("Vmag", Uniform(xmin, xmax))
        y = sample("rLG", Uniform(ymin, ymax))

        factor("lnL", f(x, y))

    nuts_kernel = NUTS(model, init_strategy=init_to_sample())
    mcmc = MCMC(nuts_kernel, num_warmup=nwarm, num_samples=nsamp)
    rng_key = random.PRNGKey(42)

    mcmc.run(rng_key,)
    mcmc.print_summary()

    samples = mcmc.get_samples()

    fout = fpath.replace(".dat", "_samples.hdf5")
    print(f"Writing samples to `{fout}`.")
    with File(fout, "w") as f:
        grp = f.create_group("samples")
        for key, value in samples.items():
            grp.create_dataset(key, data=value)
