from argparse import ArgumentParser, ArgumentTypeError


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use.")
    return parser.parse_args()


ARGS = parse_args()
# This must be done before we import JAX etc.
from numpyro import set_host_device_count, set_platform                         # noqa

set_platform(ARGS.device)                                                       # noqa

from jax import numpy as jnp                                                    # noqa
import numpy as np                                                              # noqa
import csiborgtools                                                             # noqa
from scipy.stats import multivariate_normal                                     # noqa


def get_harmonic_evidence(samples, log_posterior, nchains_harmonic, epoch_num):
    """Compute evidence using the `harmonic` package."""
    data, names = csiborgtools.dict_samples_to_array(samples)
    data = data.reshape(nchains_harmonic, -1, len(names))
    log_posterior = log_posterior.reshape(nchains_harmonic, -1)

    return csiborgtools.harmonic_evidence(
        data, log_posterior, return_flow_samples=False, epochs_num=epoch_num)


ndim = 250
nsamples = 100_000
nchains_split = 10
loc = jnp.zeros(ndim)
cov = jnp.eye(ndim)


gen = np.random.default_rng()
X = gen.multivariate_normal(loc, cov, size=nsamples)
samples = {f"x_{i}": X[:, i] for i in range(ndim)}
logprob = multivariate_normal(loc, cov).logpdf(X)

neg_lnZ_laplace, neg_lnZ_laplace_error = csiborgtools.laplace_evidence(
    samples, logprob, nchains_split)
print(f"neg_lnZ_laplace:  {neg_lnZ_laplace} +/- {neg_lnZ_laplace_error}")


neg_lnZ_harmonic, neg_lnZ_harmonic_error = get_harmonic_evidence(
    samples, logprob, nchains_split, epoch_num=30)
print(f"neg_lnZ_harmonic: {neg_lnZ_harmonic} +/- {neg_lnZ_harmonic_error}")








