# Copyright (C) 2025 Richard Stiskalek
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
Numerically stable implementations of the log-pdf of the normal distribution
and its truncated versions.

NOTE: these are currently not used in the inference because of some odd
behaviour when interfaced with a sampler.
"""

import jax.numpy as jnp
from jax.scipy.special import erf, erfc, log_ndtr


###############################################################################
#                           Auxiliary functions                               #
###############################################################################

def log1perf(x):
    """
    Computes `log(1 + erf(x))` but in a numerically stable way when
    `erf(x) -> -1`
    """
    safe_x = jnp.where(x == 0, jnp.nan, x)
    asymptotic_approx = (
        -safe_x**2 - jnp.log(jnp.abs(safe_x)) - 0.5 * jnp.log(jnp.pi)
        + jnp.log1p(-0.5 / safe_x**2 + 0.75 / safe_x**4))

    return jnp.where(
        x >= 0, jnp.log1p(erf(x)),
        jnp.where(
            x > -3,
            jnp.log(erfc(jnp.abs(x))),
            asymptotic_approx
            )
        )


def logerfc(x):
    """
    Computes `log(erfc(x))` but in a numerically stable way when `x >> 1`.
    """
    safe_x = jnp.where(x == 0, jnp.nan, x)
    asymptotic_approx = (
        -safe_x**2 - jnp.log(jnp.abs(safe_x)) - 0.5 * jnp.log(jnp.pi)
        + jnp.log1p(-0.5 / safe_x**2 + 0.75 / safe_x**4))

    return jnp.where(
        x >= 5, asymptotic_approx,
        jnp.log(erfc(x))
        )


def log1m_exp(x):
    """
    Numerically stable calculation of `log(1 - exp(x))`, following the
    algorithm of Machler [1].

    Currently returns NaN for x > 0, but may be modified in the future
    to throw a ValueError

    [1] https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf  # noqa
    """
    # return 0. rather than -0. if
    # we get a negative exponent that exceeds
    # the floating point representation
    arr_x = 1.0 * jnp.array(x)
    oob = arr_x < jnp.log(jnp.finfo(arr_x.dtype).smallest_normal)
    mask = arr_x > -0.6931472  # approx -log(2)
    more_val = jnp.log(-jnp.expm1(arr_x))
    less_val = jnp.log1p(-jnp.exp(arr_x))

    return jnp.where(
        oob,
        0.,
        jnp.where(
            mask,
            more_val,
            less_val))


def log_diff_exp(a, b):
    """
    Computes `log(exp(a) - exp(b))` in a numerically stable way.
    """
    mask = a > b
    masktwo = (a == b) & (a < jnp.inf)
    return jnp.where(mask,
                     1.0 * a + log1m_exp(
                         1.0 * b - 1.0 * a),
                     jnp.where(masktwo,
                               -jnp.inf,
                               jnp.nan))


###############################################################################
#                           Log-pdf functions                                 #
###############################################################################


def normal_logpdf(x, loc, scale):
    """Log of the normal PDF."""
    return (-0.5 * ((x - loc) / scale)**2
            - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi))


def lower_truncated_normal_logpdf(x, loc, scale, xmin):
    """
    Log of the normal PDF lower-truncated at `xmin`.
    """
    return (
        + normal_logpdf(x, loc, scale)
        + jnp.log(2)
        - logerfc((xmin - loc) / (jnp.sqrt(2) * scale))
    )


def upper_truncated_normal_logpdf(x, loc, scale, xmax):
    """Log of the normal PDF upper-truncated at `xmax`."""
    return (+ normal_logpdf(x, loc, scale)
            + jnp.log(2)
            - log1perf((xmax - loc) / (jnp.sqrt(2) * scale))
            )


def truncated_normal_logpdf(x, loc, scale, xmin, xmax):
    """Log of the normal PDF truncated between `xmin` and `xmax`."""
    a = (xmin - loc) / scale
    b = (xmax - loc) / scale

    # Compute log of normalization constant
    log_norm = log_diff_exp(log_ndtr(b), log_ndtr(a))

    return normal_logpdf(x, loc, scale) - log_norm
