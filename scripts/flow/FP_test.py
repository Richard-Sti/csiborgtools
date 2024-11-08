"""
Script to reproduce a toy version of the flow model in
`https://arxiv.org/abs/2007.04993`.
"""
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from h5py import File
from jax import numpy as jnp
from jax import random
from numpyro import factor, sample, deterministic, plate
from numpyro.handlers import scale
from numpyro.distributions import Normal, Uniform
from numpyro.infer import MCMC, NUTS
from quadax import simpson
from jax.debug import print as jprint

SPEED_OF_LIGHT = 299792.458  # km / s
ARCSEC2RAD = 4.84813681109536e-06


subtract_mean_independent = False
sample_sigma_v = False
with_malmquist = True
maglim = 17.0
zmin = 0.003
zmax = 0.03
cosmo = FlatLambdaCDM(H0=100, Om0=0.3)

Vmin = cosmo.comoving_distance(zmin).value**3
Vmax = cosmo.comoving_distance(zmax).value**3

# Minimum and maximum of the radial range for sampling the comoving distance.
rmin = 0.01
rmax = 500

# Range of redshifts over which will integrate the comoving distance
z_range = np.linspace(0.001, 0.065, 200)
r_range = cosmo.comoving_distance(z_range).value
mu_range = cosmo.distmod(z_range).value
da_range_kpc = cosmo.angular_diameter_distance(z_range).value * 1000            # noqa

# Convert to JAX arrays
z_range = jnp.asarray(z_range)
r_range = jnp.asarray(r_range)
da_range_kpc = jnp.asarray(da_range_kpc)


def distmod2redshift(mu, Om0):
    """
    Convert distance modulus to redshift, assuming `h = 1`. The expression is
    valid for a flat universe over the range of 0.00001 < z < 0.1.
    """
    return jnp.exp(((0.461108 * mu) - ((0.022187 * Om0) + (((0.022347 * mu)** (12.631788 - ((-6.708757) * Om0))) + 19.529852))))  # noqa


def distmod2dist(mu, Om0):
    """
    Convert distance modulus to distance in `Mpc / h`. The expression is valid
    for a flat universe over the range of 0.00001 < z < 0.1.
    """
    term1 = jnp.exp((0.443288 * mu) + (-14.286531))
    term2 = (0.506973 * mu) + 12.954633
    term3 = ((0.028134 * mu) ** (
        ((0.684713 * mu)
         + ((0.151020 * mu) + (1.235158 * Om0))) - jnp.exp(0.072229 * mu)))
    term4 = (-0.045160) * mu
    return (-0.000301) + (term1 * (term2 - (term3 - term4)))


def log_dA_to_distmod(log_dA, Om0):
    """
    Convert log10 of the angular diameter distance in Mpc / h to distance
    modulus. The expression is valid for a flat universe over the range of
    0.00001 < z < 0.1.
    """
    return jnp.exp((-6.542245 + 2.307573 * log_dA) +  jnp.exp(2.708684 * log_dA + (-9.708873 + 1.270249 * Om0))) + 5 * log_dA + 25  # noqa


def r2da(r):
    return 0.019863 + (0.999807 * ((0.999084 * r) * jnp.exp(-0.000322 * r)))


def r2distmod(r, Om0):
    return (-0.000009) + (1.0 * (
        ((2.171392 - (0.000005 * r)) *
         ((jnp.log(0.556252 * r)) - ((-0.007872) * r) - (9.911776 * Om0)) -
         (-26.274343)) +
        ((-0.016363 * r) + (21.519941 * Om0))
    ))


def Sn_from_mag_distance(mag, r, Om0):
    """Calculate the weights."""
    mu = r2distmod(r, Om0)

    mu_lim = mu + maglim - mag

    return jnp.clip((distmod2dist(mu_lim, Om0)**3 - Vmin) / (Vmax - Vmin), 0, 1)


###############################################################################
#                             Load the data                                   #
###############################################################################

print("Loading the data...")

with File("/mnt/extraspace/rstiskalek/catalogs/PV/CF4/SDSS-FP.hdf5", 'r') as f:
    czcmb = jnp.asarray(f["gczcmb"][...])

    # Load only the data with z < 0.05
    m = czcmb < zmax * SPEED_OF_LIGHT
    czcmb = czcmb[m]

    log_sigma = jnp.asarray(f["s"][...][m])
    e_log_sigma = jnp.asarray(f["es"][...][m])
    log_Ie = jnp.asarray(f["i"][...][m])
    e_log_Ie = jnp.asarray(f["ei"][...][m])
    theta_eff = jnp.asarray((f["rad"][...] * np.sqrt(f["boa"][...]))[m])
    mag = jnp.asarray(f["rmag"][...][m])

    log_theta_eff_rad = jnp.log10(theta_eff * ARCSEC2RAD)

    theta = jnp.asarray(np.deg2rad(90 - f["Dec"][...][m]))
    phi = jnp.asarray(np.deg2rad(f["Ra"][...][m]))

    # print("Replacing with random sky")
    # theta = np.arccos(np.random.uniform(-1, 1, len(czcmb)))
    # phi = np.random.uniform(0, 2 * np.pi, len(czcmb))

    if subtract_mean_independent:
        print("Subtracting mean of the independent variables.")
        log_sigma -= jnp.mean(log_sigma)
        log_Ie -= jnp.mean(log_Ie)

print(f"Loaded {len(log_sigma)} galaxies.")


###############################################################################
#                          Define the MCMC model                              #
###############################################################################


def normal_logpdf(x, loc, scale):
    """Log of the normal probability density function."""
    return (-0.5 * ((x - loc) / scale)**2
            - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi))


def model():
    a = sample("aFP", Normal(0, 5))
    b = sample("bFP", Normal(0, 5))
    c = sample("cFP", Normal(0, 5))

    # A simple error model, without any error propagation from the other
    # FP parameters.
    scatter = sample("e_r", Uniform(0, 5))
    # scatter = 1.
    # scatter = jnp.ones_like(log_sigma) * scatter
    scatter = jnp.sqrt(scatter**2 + a**2 * e_log_sigma**2 + b**2 * e_log_Ie**2)

    Vext = sample("Vext", Normal(0, 2500).expand([3]))

    if sample_sigma_v:
        sigma_v = sample("sigma_v", Uniform(0, 2500))
        factor("ll_sigma_v", -jnp.log(sigma_v))
    else:
        sigma_v = 250

    # Predict the log effective radius from the FP in kpc / h
    log_Reff_pred = a * log_sigma + b * log_Ie + c


    # Sample radial distance
    with plate("data", len(log_sigma)):
        r = sample("r", Uniform(rmin, rmax))


    # Convert to angular diameter distance in kpc / h
    dA = r2da(r) * 1000
    # Convert to redsfhit, using just Hubble
    zcosmo = 100 * r / SPEED_OF_LIGHT

    Sn = Sn_from_mag_distance(mag, r, 0.3)

    # jprint("mean(Sn) = {x}", x=jnp.mean(Sn))

    # Convert the log effective radius to arcsec, the shape will be
    # (n_data, n_r_range). We will integrate over the radial range.
    # theta_eff_pred = 10**log_Reff_pred[:, None] / da_range_kpc[None, :]  # rad
    theta_eff_pred = 10**log_Reff_pred / dA  # rad
    theta_eff_pred /= ARCSEC2RAD  # arcsec
    # log_theta_eff_pred = jnp.log10(theta_eff_pred)

    deterministic("theta_eff_pred", theta_eff_pred)

    # Likelihood of the angular effective radius
    # ll_theta = normal_logpdf(theta_eff_pred, theta_eff, scatter)
    # with plate("ll_theta", len(theta_eff)):
    #     sample("theta_eff", Normal(theta_eff_pred, scatter), obs=theta_eff)

    # ll = normal_logpdf(log_theta_eff_pred, log_theta_eff_rad[:, None], scatter[:, None])

    # Project V_ext to the line of sight
    Vext_projected = (+ Vext[0] * jnp.sin(theta) * jnp.cos(phi)
                      + Vext[1] * jnp.sin(theta) * jnp.sin(phi)
                      + Vext[2] * jnp.cos(theta))

    zpec = Vext_projected / SPEED_OF_LIGHT

    czpred = ((1 + zcosmo) * (1 + zpec) - 1) * SPEED_OF_LIGHT

    deterministic("czpred", czpred)

    with plate("ll_cz", len(czcmb)), scale(scale=1 / Sn):
        sample("ll_czpred", Normal(czpred, sigma_v), obs=czcmb)
        sample("theta_eff", Normal(theta_eff_pred, scatter), obs=theta_eff)

        if with_malmquist:
            ll_r = 2 * jnp.log(r)
            factor("ll_r", ll_r)

    # factor("ll", jnp.sum(ll))


###############################################################################
#                              Run the MCMC                                   #
###############################################################################


kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
mcmc.run(random.PRNGKey(0))

samples = mcmc.get_samples()
print(f"Samples are: {samples.keys()}")
mcmc.print_summary()


fname = "./test_samples.hdf5"
print(f"Writing to .. `{fname}`")
with File(fname, 'w') as f:
    grp = f.create_group("samples")
    for k, v in samples.items():
        grp.create_dataset(k, data=v)
