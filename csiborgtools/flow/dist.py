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
"""Various probability distributions required in the flow model."""
import jax.numpy as jnp
import jax.random as random
import numpyro.distributions as dist
from numpyro.distributions.util import validate_sample


###############################################################################
#                           x^2 distribution                                  #
###############################################################################


class SquaredLikeDistribution(dist.Distribution):
    """A distribution where the PDF is proportional to `x^2`."""
    reparametrized_params = ["xmin", "xmax"]
    support = dist.constraints.positive

    def __init__(self, xmin, xmax, validate_args=None):
        batch_shape, event_shape = (), ()
        self.xmin, self.xmax = xmin, xmax

        self.log_norm_const = jnp.log(3) - jnp.log(self.xmax**3 - self.xmin**3)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        u = random.uniform(key, shape=sample_shape)

        return jnp.cbrt(self.xmin**3 + u * (self.xmax**3 - self.xmin**3))

    @validate_sample
    def log_prob(self, value):
        """Compute log-probability, ensuring truncation."""
        return jnp.where(
            (value >= self.xmin) & (value <= self.xmax),
            2 * jnp.log(value) + self.log_norm_const,
            -jnp.inf)


###############################################################################
#                           10^(0.6x) distribution                            #
###############################################################################


class MagnitudeDistribution(dist.Distribution):
    """
    A distribution for apparent magnitudes where the PDF is proportional
    to `10^(0.6x)`.

    Careful because this sets the sampled values to be close to the observed
    magnitudes (which is why they must be provided).
    """
    reparametrized_params = ["xmin", "xmax"]
    support = dist.constraints.positive

    def __init__(self, xmin, xmax, mag_sample, e_mag_sample,
                 validate_args=None):
        batch_shape, event_shape = (), ()
        self.xmin, self.xmax = xmin, xmax

        self.mag_sample = mag_sample
        self.e_mag_sample = e_mag_sample

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def sample(self, key, sample_shape=()):
        u = random.normal(key, shape=sample_shape)
        return self.mag_sample + self.e_mag_sample * u

    @validate_sample
    def log_prob(self, value):
        """Compute log-probability, ensuring truncation."""
        return 0.6 * value * jnp.log(10)
