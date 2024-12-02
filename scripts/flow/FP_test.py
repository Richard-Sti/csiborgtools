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
from numpyro import factor, sample, deterministic, plate
from numpyro.infer import init_to_median
from numpyro.handlers import scale
from numpyro.distributions import Normal, Uniform
from numpyro.infer import MCMC, NUTS
from quadax import simpson
from jax.debug import print as jprint

SPEED_OF_LIGHT = 299792.458  # km / s
ARCSEC2RAD = 4.84813681109536e-06


subtract_mean_independent = False
sample_sigma_v = False
with_malmquist = False
make_full_sky = False
maglim = 17.0
zmin = 0.003
zmax = 0.05
cosmo = FlatLambdaCDM(H0=100, Om0=0.3)

Vmin = cosmo.comoving_distance(zmin).value**3
Vmax = cosmo.comoving_distance(zmax).value**3

if with_malmquist:
    print("Using homogeneous Malmquist bias.")

# Minimum and maximum of the radial range for sampling the comoving distance.
rmin = 0.01
rmax = cosmo.comoving_distance(zmax + 450 / SPEED_OF_LIGHT).value


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


def Sn_from_mag_distance(mag, r, Om0):
    """Calculate the weights."""
    mu = r2distmod(r, Om0)

    mu_lim = mu + maglim - mag

    return jnp.clip((distmod2dist(mu_lim, Om0)**3 - Vmin) / (Vmax - Vmin), 0, 1)


def get_effective_surface_brightness(mr, zcosmo, zpec, theta_eff, kr, Ar):
    return mr + 0.85 * zcosmo + 2.5 * jnp.log10(2 * np.pi * theta_eff**2) - 2.5 * jnp.log10((1 + zcosmo)**4 * (1 + zpec)**2) - kr - Ar


def get_log_Ie(mu_e):
    Msun = 4.65
    return 0.4 * Msun - 0.4 * mu_e + 2 * np.log10(206265 / 10)


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
    log_da_kpc = jnp.log10(cosmo.angular_diameter_distance(czcmb / SPEED_OF_LIGHT).value * 1000)

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

from jax.debug import print as jprint


def model():
    a = sample("aFP", Normal(0, 5))
    b = sample("bFP", Normal(0, 5))
    c = sample("cFP", Normal(-1, 1))
    # c = -0.11

    scatter = sample("scatter", Uniform(0, 1))
    # scatter = 0.1
    # scatter = jnp.sqrt(scatter**2 + a**2 * e_log_sigma**2 + b**2 * e_log_Ie**2)

    Vext = sample("Vext", Normal(0, 500).expand([3]))
    # Vext = [0, 0, 0]

    if sample_sigma_v:
        sigma_v = sample("sigma_v", Uniform(0, 2500))
        factor("ll_sigma_v", -jnp.log(sigma_v))
    else:
        sigma_v = 250


    # Sample radial distance
    with plate("data", len(log_sigma)):
        r = sample("r", Uniform(rmin, rmax))

    # # Convert to angular diameter distance in kpc / h
    dA = r2da(r) * 1000
    # Convert to redsfhit, using just Hubble
    # zcosmo = 100 * r / SPEED_OF_LIGHT
    zcosmo = dist2redshift(r, 0.3)

    # Project V_ext to the line of sight
    Vext_projected = (+ Vext[0] * jnp.sin(theta) * jnp.cos(phi)
                      + Vext[1] * jnp.sin(theta) * jnp.sin(phi)
                      + Vext[2] * jnp.cos(theta))

    zpec = Vext_projected / SPEED_OF_LIGHT
    czpred = ((1 + zcosmo) * (1 + zpec) - 1) * SPEED_OF_LIGHT


    # mue = get_effective_surface_brightness(rmag, zcosmo, zpec, theta_eff, kr, Ar)
    # log_Ie_pred = get_log_Ie(mue)

    # Predict the log effective radius from the FP in kpc / h
    log_Reff_pred = a * log_sigma + b * log_Ie + c
    # Convert to angular diameter distance in Mpc / h
    # log_da_FP = (log_Reff_pred - log_theta_eff_rad) - 3
    # log_da_FP = log_Reff_pred - log_theta_eff_rad - 3
    # print(10**log_da_FP)
    # jprint("DA = {x}", x=10**log_da_FP)
    # Convert to a distance modulus
    # mu_FP = log_dA_to_distmod(log_da_FP, 0.3)
    # mu_FP = 5 * log_da_FP + 25
    # jprint("mu_FP = {x}", x=mu_FP)
    #  print(f"mu_FP = {mu_FP}")

    # with plate("plate_mu", len(log_sigma)):
        # mu = sample("mu", Normal(mu_FP, scatter))

    # zcosmo = distmod2redshift(mu_FP, 0.3)



    # Sn = Sn_from_mag_distance(mag, r, 0.3)
    # # jprint("mean(Sn) = {x}", x=jnp.mean(Sn))

    # deterministic("czpred", czpred)
    log_Reff_composed = log_theta_eff_rad + jnp.log10(dA)

    with plate("ll_plate", len(czcmb)):
        sample("ll_log_Reff", Normal(log_Reff_pred, scatter), obs=log_Reff_composed)

        sample("ll_czpred", Normal(czpred, sigma_v), obs=czcmb)

        if with_malmquist:
            ll_r = 2 * jnp.log(r)
            factor("ll_r", ll_r)


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
