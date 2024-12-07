"""
Script to reproduce a toy version of the flow model in
`https://arxiv.org/abs/2007.04993`.
"""
import sys
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from h5py import File
from jax import numpy as jnp
from jax import random
from numpyro import factor, sample, plate
from numpyro.infer import init_to_median
from numpyro.distributions import Normal, Uniform
from numpyro.infer import MCMC, NUTS

SPEED_OF_LIGHT = 299792.458  # km / s
ARCSEC2RAD = 4.84813681109536e-06


subtract_mean_independent = False
sample_sigma_v = False
with_malmquist = False
make_full_sky = False
maglim = 17.0
zmin = 0.003
zmax = 0.03

cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
Vmin = cosmo.comoving_distance(zmin).value**3
Vmax = cosmo.comoving_distance(zmax).value**3

if subtract_mean_independent:
    print("Subtracting mean of the independent variables.")

if sample_sigma_v:
    print("Sampling sigma_v.")

if with_malmquist:
    print("Using homogeneous Malmquist bias.")

if make_full_sky:
    print("Resampling sky coordinates to be full sky.")

# Minimum and maximum of the radial range for sampling the comoving distance.
rmin = 0.01
rmax = cosmo.comoving_distance(zmax + 1000 / SPEED_OF_LIGHT).value
print(f"Setting rmin = {rmin} Mpc / h, rmax = {rmax} Mpc / h.")


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


def dist2redshift(dist, Omega_m, h=1.):
    """
    Convert comoving distance to cosmological redshift if the Universe is
    flat and z << 1.
    """
    eta = 3 * Omega_m / 2
    return 1 / eta * (1 - (1 - 2 * 100 * h * dist / SPEED_OF_LIGHT * eta)**0.5)


def r2da(r):
    return 0.019863 + (0.999807 * ((0.999084 * r) * jnp.exp(-0.000322 * r)))


def r2distmod(r, Om0):
    return (-0.000009) + (1.0 * (
        ((2.171392 - (0.000005 * r)) *
         ((jnp.log(0.556252 * r)) - ((-0.007872) * r) - (9.911776 * Om0)) -
         (-26.274343)) +
        ((-0.016363 * r) + (21.519941 * Om0))
    ))


###############################################################################
#                             Load the data                                   #
###############################################################################

print("Loading the data...")

with File("/mnt/extraspace/rstiskalek/catalogs/PV/CF4/SDSS-FP.hdf5", 'r') as f:
    czcmb = jnp.asarray(f["gczcmb"][...])

    m = czcmb < zmax * SPEED_OF_LIGHT
    czcmb = czcmb[m]

    log_Reff = jnp.asarray(f["r"][...][m])
    e_log_Reff = jnp.asarray(f["er"][...][m])

    log_sigma = jnp.asarray(f["s"][...][m])
    e_log_sigma = jnp.asarray(f["es"][...][m])

    log_Ie = jnp.asarray(f["i"][...][m])
    e_log_Ie = jnp.asarray(f["ei"][...][m])

    theta_eff = jnp.asarray((f["rad"][...] * np.sqrt(f["boa"][...]))[m])
    log_theta_eff_rad = jnp.log10(theta_eff * ARCSEC2RAD)

    # Computed from the *observed* redshift..
    log_da_kpc = jnp.log10(
        cosmo.angular_diameter_distance(czcmb / SPEED_OF_LIGHT).value * 1000)

    rmag = jnp.asarray(f["rmag"][...][m])
    kr = jnp.asarray(f["kcr"][...][m])
    Ar = jnp.asarray(f["Exr"][...][m])

    theta = jnp.asarray(np.deg2rad(90 - f["Dec"][...][m]))
    phi = jnp.asarray(np.deg2rad(f["Ra"][...][m]))

    if make_full_sky:
        print("Replacing with random sky")
        theta = np.arccos(np.random.uniform(-1, 1, len(czcmb)))
        phi = np.random.uniform(0, 2 * np.pi, len(czcmb))

    if subtract_mean_independent:
        print("Subtracting mean of the independent variables.")
        log_sigma -= jnp.mean(log_sigma)
        log_Ie -= jnp.mean(log_Ie)

print(f"Loaded {len(log_sigma)} galaxies.")


###############################################################################
#                          Define the MCMC model                              #
###############################################################################


def model():
    a = sample("aFP", Normal(0, 1))
    b = sample("bFP", Normal(0, 1))
    c = sample("cFP", Normal(0, 1))

    scatter = sample("scatter", Uniform(0, 1))

    Vext = sample("Vext", Normal(0, 500).expand([3]))

    if sample_sigma_v:
        sigma_v = sample("sigma_v", Uniform(0, 2500))
        factor("ll_sigma_v", -jnp.log(sigma_v))
    else:
        sigma_v = 250

    # Sample radial distance
    with plate("data", len(log_sigma)):
        r = sample("r", Uniform(rmin, rmax))

    # Convert to angular diameter distance in kpc / h
    dA = r2da(r) * 1000
    # Convert to redsfhit, using just Hubble
    zcosmo = dist2redshift(r, 0.3)

    # Project V_ext to the line of sight
    Vext_projected = (+ Vext[0] * jnp.sin(theta) * jnp.cos(phi)
                      + Vext[1] * jnp.sin(theta) * jnp.sin(phi)
                      + Vext[2] * jnp.cos(theta))

    zpec = Vext_projected / SPEED_OF_LIGHT
    czpred = ((1 + zcosmo) * (1 + zpec) - 1) * SPEED_OF_LIGHT

    # Predict the log effective radius from the FP in kpc / h
    log_Reff_pred = a * log_sigma + b * log_Ie + c

    # Log effective radius from the angular size and the ang diameter distance
    log_Reff_composed = log_theta_eff_rad + jnp.log10(dA)

    with plate("ll_plate", len(czcmb)):
        sample("ll_log_Reff", Normal(log_Reff_pred, scatter),
               obs=log_Reff_composed)
        sample("ll_czpred", Normal(czpred, sigma_v), obs=czcmb)

        if with_malmquist:
            factor("ll_r", 2 * jnp.log(r))


###############################################################################
#                              Run the MCMC                                   #
###############################################################################


kernel = NUTS(model, init_strategy=init_to_median(num_samples=1000))
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000)
mcmc.run(random.PRNGKey(2),)

samples = mcmc.get_samples()
print(f"Samples are: {samples.keys()}")
mcmc.print_summary()


fname = "./test_samples.hdf5"
print(f"Writing to .. `{fname}`")
with File(fname, 'w') as f:
    grp = f.create_group("samples")
    for k, v in samples.items():
        grp.create_dataset(k, data=v)

    # Write some of the data to check the results
    grp_data = f.create_group("data")
    grp_data.create_dataset("i", data=log_Ie)
    grp_data.create_dataset("s", data=log_sigma)
    grp_data.create_dataset("r", data=log_Reff)

    grp_data.create_dataset("theta", data=theta)
    grp_data.create_dataset("phi", data=phi)

    grp_data.create_dataset("rmag", data=rmag)


fname_summary = fname.replace(".hdf5", "_summary.txt")
print(f"Saving summary: `{fname_summary}`.")
with open(fname_summary, 'w') as f:
    original_stdout = sys.stdout
    sys.stdout = f

    mcmc.print_summary(exclude_deterministic=False)
    sys.stdout = original_stdout
