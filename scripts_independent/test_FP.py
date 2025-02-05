import sys

import corner
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from jax import random
from numpyro import factor, plate, sample
from numpyro.distributions import Normal, Uniform
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_median
from quadax import simpson

run_num = int(sys.argv[1])
gen = np.random.default_rng(1 + run_num)
rng_key = random.PRNGKey(run_num)

SPEED_OF_LIGHT = 299_792.458
nwarm, nsamp = 1500, 3000

###############################################################################
#                             Injected values                                 #
###############################################################################

ngal = 300
add_malmquist = False
numerical_distance_marginalisation = True

a_FP_true = 0.5
b_FP_true = 0.1
c_FP_true = 2.0
sigma_FP_gt = 0.05

Vmono_true = 0.1
D_mag_true = 300
D_ra_true = 5 / 4 * np.pi
cos_D_theta_true = 0.3
sigmav_true = 250

sig_mean_true, sig_std_true = 1.5, 0.2
e_sig = 0.02

I_mean_true, I_std_true = 1.5, 0.2
e_I = 0.02

print(f"We have {ngal} galaxies.")

if add_malmquist:
    print("We are using the Malmquist bias.")
else:
    print("We are not using Malmquist bias")

if numerical_distance_marginalisation:
    print("We are using numerical distance marginalisation.")
else:
    print("We are not using numerical distance marginalisation.")


###############################################################################
#                           Mock data generation                              #
###############################################################################


D_theta_true = np.arccos(cos_D_theta_true)

phi = gen.uniform(0, 2 * np.pi, ngal)
theta = np.pi / 2 - np.arcsin(gen.uniform(-1, 1, ngal))

sig_gt = gen.normal(sig_mean_true, sig_std_true, ngal)
sig = gen.normal(sig_gt, e_sig)

sig_mean = np.mean(sig)
print(f"Subtracting `sig_mean` of {sig_mean} from `sig`, `sig_gt`, and "
      "`sig_mean_true`")
sig -= sig_mean
sig_gt -= sig_mean
sig_mean_true -= sig_mean

I_gt = gen.normal(I_mean_true, I_std_true, ngal)
I = gen.normal(I_gt, e_I)  # noqa

I_mean = np.mean(I)
print(f"Subtracting `I_mean` of {I_mean} from `I`, `I_gt`, and `I_mean_true`")
I -= I_mean
I_gt -= I_mean
I_mean_true -= I_mean

logd_FP = a_FP_true * sig_gt + b_FP_true * I_gt + c_FP_true
logd_FP_gt = gen.normal(logd_FP, sigma_FP_gt)
zcosmo_gt = 100 * 10**logd_FP_gt / SPEED_OF_LIGHT
print(f"Mean and std of zcosmo: {np.mean(zcosmo_gt)}, {np.std(zcosmo_gt)}")


d_range = np.linspace(1, 3 * 10**np.max(logd_FP_gt), 800)
log_d_range = np.log10(d_range)
zcosmo_range = 100 * d_range / SPEED_OF_LIGHT
print(f"The distance range goes {d_range.min()} to {d_range.max()} in "
      f"{len(d_range)} steps.")

Vrad = D_mag_true * (
    + np.sin(D_theta_true) * np.sin(theta) * np.cos(D_ra_true - phi)
    + np.cos(D_theta_true) * np.cos(theta))
Vrad += Vmono_true


ztrue = (1 + zcosmo_gt) * (1 + Vrad / SPEED_OF_LIGHT) - 1
czobs = gen.normal(SPEED_OF_LIGHT * ztrue, sigmav_true)

plt.figure()
plt.hist(czobs / SPEED_OF_LIGHT, bins="auto", histtype="step", label="czobs")
plt.hist(zcosmo_gt, bins="auto", histtype="step", label="ztrue")
plt.legend()
plt.xlabel("Redshift")
plt.tight_layout()

fname = "Plots_FP/czobs_hist.png"
print(f"Saving the redshift distribution to `{fname}`.")
plt.savefig("Plots_FP/czobs_hist.png", dpi=450)
plt.close()


###############################################################################
#                              Forward model                                  #
###############################################################################


def model():
    # Sample the velocity dispersion parameters.
    a_FP = sample("a_FP", Uniform(a_FP_true - 0.4, a_FP_true + 0.4))
    sig_mean = sample(
        "sig_mean", Uniform(sig_mean_true - 2, sig_mean_true + 2))
    sig_std = sample("sig_std", Uniform(0, sig_std_true + 0.2))
    with plate("plate_ll_sig", ngal):
        sig_true = sample("sig_true", Normal(sig_mean, sig_std))

    factor('ll_sig', Normal(sig_true, e_sig).log_prob(sig))

    # Sample the surface brightness parameters if non-zero.
    if b_FP_true != 0:
        b_FP = sample("b_FP", Uniform(b_FP_true - 0.4, b_FP_true + 0.4))
        I_mean = sample("I_mean", Uniform(I_mean_true - 2, I_mean_true + 2))
        I_std = sample("I_std", Uniform(0, I_std_true + 0.2))

        with plate("plate_ll_I", ngal):
            I_true = sample("I_true", Normal(I_mean, I_std))
        factor('ll_I', Normal(I_true, e_I).log_prob(I))
    else:
        b_FP = 0.
        I_true = 0.

    # Sample the intercept if non-zero.
    if c_FP_true != 0:
        c_FP = sample("c_FP", Uniform(c_FP_true - 0.4, c_FP_true + 0.4))
        # c_FP = c_FP_true
    else:
        c_FP = 0.

    # Sample the intrinsic scatter and the velocity scatter
    sigma_FP = sample("sigma_FP", Uniform(0, 0.3))
    sigmav = sample("sigmav", Uniform(0, 5000))

    # Sample the (radially constant) peculiar velocity.
    Vpec = 0

    # Sample the monopole velocity if it is non-zero.
    if Vmono_true != 0:
        Vmono = sample("Vmono", Uniform(-3500, 3500))
        Vpec += Vmono

    # Sample the dipole velocity if it is non-zero.
    if D_mag_true > 0:
        D_mag = sample("D_mag", Uniform(0, 10 * D_mag_true))
        D_ra = sample("D_ra", Uniform(0, 2 * np.pi))
        cos_D_theta = sample("cos_D_theta", Uniform(-1, 1))
        D_theta = jnp.arccos(cos_D_theta)
        Vpec += D_mag * (
            + jnp.sin(D_theta) * jnp.sin(theta) * jnp.cos(D_ra - phi)
            + jnp.cos(D_theta) * jnp.cos(theta))

    # Calculate the esimated log-distance and sample the true distance.
    log_d_FP_estimate = a_FP * sig_true + b_FP * I_true + c_FP

    if numerical_distance_marginalisation:
        # Calculate the log-probability of shape `(ngal, len(d_range))`
        log_prob_dist = -0.5 * (log_d_FP_estimate[:, None] - log_d_range[None, :])**2 / sigma_FP**2 - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigma_FP)  # noqa

        if add_malmquist:
            log_prob_dist += 2 * log_d_range[None, :]

            # Calculate the normalisation and finally turn to a log
            norm = jnp.log(simpson(jnp.exp(log_prob_dist), x=d_range, axis=-1))
            log_prob_dist -= norm[:, None]

        # Redshift log-likelihood still of shape `(ngal, len(d_range))`
        czpred = SPEED_OF_LIGHT * ((1 + zcosmo_range[None, :]) * (1 + Vpec[:, None] / SPEED_OF_LIGHT) - 1)                 # noqa
        log_prob_redshift = -0.5 * (czpred - czobs[:, None])**2 / sigmav**2 - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(sigmav)  # noqa

        # Add the log-likelihoods and integrate over the distance
        ll = log_prob_dist + log_prob_redshift
        factor("ll", jnp.log(simpson(jnp.exp(ll), axis=-1)))
    else:
        with plate("plate_log_d", ngal):
            log_d_FP_true = sample(
                "log_d_FP_true", Normal(log_d_FP_estimate, sigma_FP))

        # If adding the Malmquist bias term, then must add the r^2 term and
        # then the normalisation term. There is a Jacobian to convert the
        # Malmquist bias p(r) to p(log r).
        if add_malmquist:
            d_FP_true = 10**log_d_FP_true

            jac = np.log(10) * d_FP_true
            norm = d_range**2 * jnp.exp(-0.5 * (log_d_FP_estimate[:, None] - log_d_range[None, :])**2 / sigma_FP**2) / (jnp.sqrt(2 * np.pi) * sigma_FP)  # noqa
            norm = simpson(norm, x=d_range, axis=-1)
            norm = jnp.log(jac) + 2 * jnp.log(d_FP_true) - jnp.log(norm)

            factor("ll_Malmquist", norm)

        # Assume Hubble law to convert the log-distance to redshift.
        zcosmo = 100 * 10**log_d_FP_true / SPEED_OF_LIGHT

        # Calculate the predicted redshift and the redshift likelihood.
        czpred = SPEED_OF_LIGHT * ((1 + zcosmo) * (1 + Vpec / SPEED_OF_LIGHT) - 1)  # noqa

        with plate("plate_ll_cz", ngal):
            sample("czobs", Normal(czpred, sigmav), obs=czobs)


###############################################################################
#                              MCMC Inference                                 #
###############################################################################

kernel = NUTS(model, init_strategy=init_to_median(num_samples=1000))
mcmc = MCMC(kernel, num_warmup=nwarm, num_samples=nsamp)
mcmc.run(rng_key)

mcmc.print_summary()
samples = mcmc.get_samples()


###############################################################################
#                           Diagnostic plots                                  #
###############################################################################

labels = []
keys = list(samples.keys())
nparam = np.zeros(len(keys), dtype=int)
for i in range(len(keys)):
    if len(samples[keys[i]].shape) == 1:
        labels += [keys[i]]
        nparam[i] = 1
    else:
        nparam[i] = samples[keys[i]].shape[1]
        labels += [keys[i] + '_%i' % j for j in range(nparam[i])]

nparam = [0] + list(np.cumsum(nparam))

# Flatten the samples array so it is (# samples, # parameters)
all_samples = np.empty((samples[keys[0]].shape[0], len(labels)))
for i in range(len(keys)):
    if len(samples[keys[i]].shape) == 1:
        all_samples[:, nparam[i]] = samples[keys[i]][:]
    else:
        for j in range(nparam[i+1]-nparam[i]):
            all_samples[:, nparam[i]+j] = samples[keys[i]][:, j]

labels = np.array(labels)
all_samples = np.array(all_samples)

# All samples and then loglike in last column
samps_like = np.transpose(np.vstack([np.transpose(all_samples)]))
# As listed in the summary table and corner plot; see below
labels_like = np.array(list(labels))

labels_keep = ["a_FP", "sig_mean", "sig_std"]
truths = [a_FP_true, sig_mean_true, sig_std_true]

if "sigmav" in samples.keys():
    labels_keep += ["sigmav"]
    truths += [sigmav_true]

if b_FP_true != 0:
    labels_keep += ["b_FP", "I_mean", "I_std"]
    truths += [b_FP_true, I_mean_true, I_std_true]

if c_FP_true != 0 and "c_FP" in samples.keys():
    labels_keep += ["c_FP"]
    truths += [c_FP_true]

if D_mag_true > 0:
    labels_keep += ["D_mag", "D_ra", "cos_D_theta"]
    truths += [D_mag_true, D_ra_true, cos_D_theta_true]

if sigma_FP_gt > 0 and "sigma_FP" in samples.keys():
    labels_keep += ["sigma_FP"]
    truths += [sigma_FP_gt]

if Vmono_true != 0:
    labels_keep += ["Vmono"]
    truths += [Vmono_true]

labels_keep = np.array(labels_keep)
truths = np.array(truths)
labels = np.array(labels)

mask = np.zeros(len(labels))
for i in range(len(labels_keep)):
    # Bool array of where this label is in the labels array
    mask_i = labels == labels_keep[i]
    # 1 wherever the labels match, 0 elsewhere
    mask += mask_i.astype(int)

# Have to include at least 1 of them
assert np.max(mask) == 1
# True wherever labels match, False elsewhere
mask = mask.astype(bool)

labels_mask = labels[mask]
samples_mask = all_samples[:, mask]

# Samples reordered to be in the order of labels_keep
samples_mask_1 = np.copy(samples_mask)
for i in range(len(labels_mask)):
    ind2 = np.where(labels_keep == labels_mask[i])[0][0]
    samples_mask_1[:, ind2] = samples_mask[:, i]

corner.corner(
    samples_mask_1, labels=labels_keep, truths=truths,
    show_titles=True, title_fmt='.3f', smooth=1)

fname = f"Plots_FP/corner_{run_num}.png"
if add_malmquist:
    fname = f"Plots_FP/corner_{run_num}_Malmquist.png"

print(f"Saving the corner plot to `{fname}`.")
plt.savefig(fname)
