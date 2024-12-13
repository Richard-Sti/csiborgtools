# Copyright (C) 2024 Richard Stiskalek, Deaglan Bartlett
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
Formulas to convert the non-linear `sigma8` to linear `sigma8`.
"""
import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from tqdm import trange


def nonlinear_sigma8_from_pk(As_fid, h, Om, Ob, ns, mnu, w0, wa, **kwargs):
    """
    Compute the non-linear sigma_8 from the input cosmological parameters
    using `symbolic_pofk`.

    `As_fid` is assumed to be in units of 1e-9.
    """
    try:
        from symbolic_pofk import syren_new  # noqa
    except ImportError as e:
        raise ImportError("`syren_new` not found. Please install `symbolic_pofk` package.") from e  # noqa

    k = np.logspace(np.log10(9e-3), np.log10(9), 2500)  # h Mpc^-1
    Pk = syren_new.pnl_new_emulated(k, As_fid, Om, Ob, h, ns, mnu, w0, wa, a=1)

    R = 8.0  # 8 Mpc / h

    # Compute integrand for sigma8
    kR = k * R
    Wk = 3 * (np.sin(kR) - kR * np.cos(kR)) / (kR**3)
    integrand = k**2 * Pk * Wk**2  # k^2 * P(k) * |W(kR)|^2

    # Integrate using the trapezium rule in log-space
    sigma8_squared = simpson(integrand * k, x=np.log(k)) / (2 * np.pi**2)
    return np.sqrt(sigma8_squared)


def find_linear_sigma8(target_sigma8_nl, h, Om, Ob, ns, mnu, w0, wa,
                       verbose=True, return_full=True, **kwargs):
    """
    Find the linear sigma_8 that corresponds to a given non-linear sigma_8
    using `symbolic_pofk`.
    """
    try:
        from symbolic_pofk import linear_new  # noqa
    except ImportError as e:
        raise ImportError("`linear_new` not found. Please install `symbolic_pofk` package.") from e  # noqa

    def to_opt(As):
        sigma8 = nonlinear_sigma8_from_pk(As, h, Om, Ob, ns, mnu, w0, wa)
        return (sigma8 - target_sigma8_nl)**2

    # Fiducial value to start the optimizer
    As_fid = 2.2
    res = minimize(to_opt, As_fid)
    As = res.x[0]

    # Convert to sigma8
    sigma8 = linear_new.As_to_sigma8(As, Om, Ob, h, ns, mnu, w0, wa)

    sigma8_root = nonlinear_sigma8_from_pk(As, h, Om, Ob, ns, mnu, w0, wa)
    if verbose:
        print(f"sigma8 = {sigma8} -> As = {As}e-9 -> sigma8_nl = {sigma8_root}")  # noqa

    if return_full:
        return sigma8, As, sigma8_root
    else:
        return sigma8


def make_nonlinear_to_linear_sigma8(sigma8_nl_min, sigma8_nl_max, num_points,
                                    h, Om, Ob, ns, mnu, w0, wa, **kwargs):
    """
    Make a function that converts non-linear sigma8 to linear sigma8 using
    `symbolic_pofk`.
    """

    sigma8_nonlinear_range = np.linspace(
        sigma8_nl_min, sigma8_nl_max, num_points)
    sigma8_linear_range = np.full(num_points, np.nan)

    for n in trange(num_points, desc="Computing linear sigma8"):
        x = sigma8_nonlinear_range[n]
        sigma8_linear_range[n] = find_linear_sigma8(
            x, h, Om, Ob, ns, mnu, w0, wa, verbose=False, return_full=False)

    return interp1d(sigma8_nonlinear_range, sigma8_linear_range, kind='cubic')


def sigma8_nonlinear_to_linear_juszkiewicz(sigma8_nl, beta=0.216):
    """
    Convert non-linear sigma8 to linear sigma8 using the Juszkiewicz formula
    (https://arxiv.org/pdf/0901.0697).

    By default, `beta` is 0.216 to match the value used in the paper.
    """
    return ((np.sqrt(1 + 4 * beta * sigma8_nl**2) - 1) / (2 * beta))**0.5
