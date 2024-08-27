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
"""Selection functions for peculiar velocities."""
import numpy as np
from jax import numpy as jnp
from scipy.integrate import quad
from scipy.optimize import minimize


class ToyMagnitudeSelection:
    """
    Toy magnitude selection according to Boubel+2024 [1].

    References
    ----------
    [1] https://www.arxiv.org/abs/2408.03660
    """
    def log_true_pdf(self, m, m1):
        """Unnormalized `true' PDF."""
        return 0.6 * (m - m1)

    def log_selection_function(self, m, m1, m2, a):
        """Logarithm of the Boubel+2024 selection function."""
        return np.where(m <= m1,
                        0,
                        a * (m - m2)**2 - a * (m1 - m2)**2 - 0.6 * (m - m1))

    def log_observed_pdf(self, m, m1, m2, a):
        """
        Logarithm of the unnormalized observed PDF, which is the product
        of the true PDF and the selection function.
        """
        # Calculate the normalization constant
        f = lambda m: 10**(self.log_true_pdf(m, m1)                             # noqa
                           + self.log_selection_function(m, m1, m2, a))
        mmin, mmax = 0, 25
        norm = quad(f, mmin, mmax)[0]

        return (self.log_true_pdf(m, m1)
                + self.log_selection_function(m, m1, m2, a)
                - np.log10(norm))

    def fit(self, mag):
        """Fit magnitude selection parameters to the observed magnitudes."""

        def loss(x):
            m1, m2, a = x

            if a >= 0:
                return np.inf

            return -np.sum(self.log_observed_pdf(mag, m1, m2, a))

        x0 = [12.0, 12.5, -0.1]
        return minimize(loss, x0, method="Nelder-Mead")


def toy_log_magnitude_selection(mag, m1, m2, a):
    """
    JAX implementation of `ToyMagnitudeSelection` but natural logarithm,
    whereas the one in `ToyMagnitudeSelection` is base 10.
    """
    return jnp.log(10) * jnp.where(
        mag <= m1,
        0,
        a * (mag - m2)**2 - a * (m1 - m2)**2 - 0.6 * (mag - m1))
