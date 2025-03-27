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
from jax import lax
from jax import scipy as jsy
from jax._src.numpy.util import promote_args_inexact
from numpyro.distributions.util import validate_sample
from numpyro.util import is_prng_key

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


###############################################################################
#                       Truncated Gaussian distribution                       #
###############################################################################


def erfcx(x):
    (x,) = promote_args_inexact("erfcx", x)
    a = lax.abs(x)
    p = a + 2.0
    r = 1.0 / p
    q = (a - 2.0) * r
    t = (q + 1.0) * (-2.0) + a
    e = q * (-a) + t
    q = r * e + q
    p = float.fromhex("0x1.f10000p-15")  # 5.92470169e-5
    p = p * q + (float.fromhex("0x1.521cc6p-13"))   #  1.61224554e-4
    p = p * q + (-float.fromhex("0x1.6b4ffep-12"))  # -3.46481771e-4
    p = p * q + (-float.fromhex("0x1.6e2a7cp-10"))  # -1.39681227e-3
    p = p * q + (float.fromhex("0x1.3c1d7ep-10"))   #  1.20588380e-3
    p = p * q + (float.fromhex("0x1.1cc236p-07"))   #  8.69014394e-3
    p = p * q + (-float.fromhex("0x1.069940p-07"))  # -8.01387429e-3
    p = p * q + (-float.fromhex("0x1.bc1b6cp-05"))  # -5.42122945e-2
    p = p * q + (float.fromhex("0x1.4ff8acp-03"))   #  1.64048523e-1
    p = p * q + (-float.fromhex("0x1.54081ap-03"))  # -1.66031078e-1
    p = p * q + (-float.fromhex("0x1.7bf5cep-04"))  # -9.27637145e-2
    p = p * q + (float.fromhex("0x1.1ba03ap-02"))   #  2.76978403e-1
    d = a + 0.5
    r = 1.0 / d
    r = r * 0.5
    q = p * r + r
    e = (p - q) - (q + q) * a + 1.0
    r = e * r + q
    r = jnp.where(a > float.fromhex("0x1.fffffep127"), 0.0, r)
    s = x * x
    d = x * x - s
    e = lax.exp(s)
    r = jnp.where(
        x < 0,
        jnp.where(e > float.fromhex("0x1.fffffep127"), e, e - r + e * (d + d) + e),  # noqa
        r,
    )
    return r


class UpperTruncatedGaussian(dist.Distribution):
    """An upper truncated Gaussian distribution written by Guilhem Lavaux."""

    arg_constraints = {
        "max_val": dist.constraints.real,
        "loc": dist.constraints.real,
        "std": dist.constraints.positive,
    }

    pytree_data_fields = ["loc", "std", "max_val", "_support"]
    reparametrized_params = ["max_val"]

    def __init__(self, max_val, loc=0.0, std=1.0, validate_args=None):
        batch_shape, event_shape = (), ()
        self.loc, self.std, self.max_val = loc, std, max_val
        self._support = dist.constraints.less_than(max_val)

        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    @dist.constraints.dependent_property(is_discrete=False, event_dim=0)
    def support(self):
        return self._support

    def sample(self, key, sample_shape=()):
        # This does not work in a JITTed environment.
        import jax

        assert is_prng_key(key)
        u = random.normal(key, shape=sample_shape) * self.std + self.loc

        def _test_u(u):
            bad_count = (u > self.max_val).sum()
            return bad_count != 0

        def _upd_u(key, u):
            cond = jnp.where(u > self.max_val)
            new_vals = (
                random.normal(key, shape=sample_shape)[cond] * self.std + self.loc  # noqa
            )
            return u.at[cond].set(new_vals)

        while _test_u(u):
            key = random.split(key)[0]
            u = _upd_u(key, u)

        return u

    @validate_sample
    def log_prob(self, value):
        """Compute log-probability, ensuring truncation."""

        cond = value < self.max_val
        norm1 = jnp.log(self.std)
        delta = (self.loc - self.max_val) / self.std
        q = jnp.where(delta > 5, 5.0, delta)
        # jax.debug.print("delta={delta}", delta=delta)
        norm2 = jnp.where(
            delta < 5,
            jnp.log(jsy.special.erfc(q / jnp.sqrt(2.0))),
            # jnp.where(delta > 4,
            # This is an asymptotic expansion of erfcx for large arguments.
            -jnp.log(jnp.abs(delta) * jnp.sqrt(jnp.pi / 2)) - delta**2 / 2,
            #   jnp.log(erfcx(q / jnp.sqrt(2.0))) - q**2 / 2),
        )
        value = -0.5 * ((value - self.loc) / self.std) ** 2 - jnp.log(
            2 * jnp.sqrt(jnp.pi)
        )
        norm = norm1 + norm2
        return jnp.where(cond, value - norm, -jnp.inf)

    def cdf(self, value):
        delta = (self.loc - self.max_val) / self.std
        norm2 = jnp.where(
            delta < 3,
            jnp.log(jsy.special.erfc(delta / jnp.sqrt(2.0))),
            jnp.log(erfcx(delta / jnp.sqrt(2.0))) - delta**2 / 2,
        )
        delta_value = (self.loc - value) / self.std
        I = jnp.where(  # noqa
            value < self.max_val,
            jsy.special.erfc(delta_value / jnp.sqrt(2.0)),
            jsy.special.erfc(delta / jnp.sqrt(2.0)),
        )

        return I / jnp.exp(norm2)

    @property
    def mean(self):
        delta = (self.loc - self.max_val) / self.std
        norm2 = jnp.where(
            delta < 3,
            jnp.log(jsy.special.erfc(delta / jnp.sqrt(2.0))),
            jnp.log(erfcx(delta / jnp.sqrt(2.0))) - delta**2 / 2,
        )

        return (
            self.loc
            - jnp.sqrt(2 / jnp.pi) * jnp.exp(-0.5 * delta**2 - norm2) * self.std  # noqa
        )

    @property
    def var(self):
        delta = (self.loc - self.max_val) / self.std
        norm2 = jnp.where(
            delta < 3,
            jnp.log(jsy.special.erfc(delta / jnp.sqrt(2.0))),
            jnp.log(erfcx(delta / jnp.sqrt(2.0))) - delta**2 / 2,
        )
        return self.std**2 * (
            1 + delta * jnp.exp(-(delta**2) / 2 - norm2) * 2 / jnp.sqrt(2 * jnp.pi)  # noqa
        )
