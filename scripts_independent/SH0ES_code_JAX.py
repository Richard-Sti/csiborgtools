from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from h5py import File
from jax import numpy as jnp
from jax import random
from numpyro import factor, sample
from numpyro.distributions import Uniform
from numpyro.infer import MCMC, NUTS
from scipy import linalg


def load_data(fdir):
    lstsq_results_path = join(fdir, 'lstsq_results.txt')
    Y_fits_path = join(fdir, 'ally_shoes_ceph_topantheonwt6.0_112221.fits')
    L_fits_path = join(fdir, 'alll_shoes_ceph_topantheonwt6.0_112221.fits')
    C_fits_path = join(fdir, 'allc_shoes_ceph_topantheonwt6.0_112221.fits')

    Y = fits.open(Y_fits_path)[0].data
    L = fits.open(L_fits_path)[0].data
    C = fits.open(C_fits_path)[0].data

    C_inv_cho = linalg.cho_solve(linalg.cho_factor(C), np.identity(C.shape[0]))
    q_lstsq, sigma_lstsq = np.loadtxt(lstsq_results_path, unpack=True)
    mu_list = q_lstsq
    width_list = sigma_lstsq * 10

    # width_list[width_list == 0] = jnp.min(width_list[width_list > 0])
    ks = np.where(width_list == 0)[0]
    if len(ks) > 0:
        print("Warning: zero width found in the priors. Setting it to 1e-5.")
        print(f"Indices of zero width: {ks}")
    width_list[ks] = 1e-5

    mu_list = jnp.asarray(mu_list)
    width_list = jnp.asarray(width_list)
    theta_min, theta_max = mu_list - width_list / 2, mu_list + width_list / 2

    data = {
        "Y": Y,
        "L": L,
        "C_inv_cho": C_inv_cho,
        "theta_min": theta_min,
        "theta_max": theta_max
        }

    for key in data:
        data[key] = jnp.asarray(data[key], dtype=jnp.float32)

    return data


def model(Y, L, C_inv_cho, theta_min, theta_max):
    theta = sample('theta', Uniform(theta_min, theta_max))
    res = Y - jnp.dot(theta, L)
    factor("ll", -0.5 * jnp.dot(res, jnp.dot(C_inv_cho, res)))


def plot_H0(samples):
    log_H0_samples = np.array(samples['theta'])[:, -1]
    H0_samples = 10**(log_H0_samples / 5.)
    H0_mean = np.mean(H0_samples)
    H0_low, H0_up = np.percentile(H0_samples, [16, 84])

    print(f"Mean H0: {H0_mean:.4f} km/s/Mpc")
    print(f"68% Confidence Interval: ({H0_low:.4f}, {H0_up:.4f}) km/s/Mpc")

    plt.figure()
    plt.hist(H0_samples, bins="auto", histtype='step', color='blue')
    plt.axvline(H0_mean, color='red', linestyle='--', label='Mean')
    plt.axvline(H0_low, color='green', linestyle='--', label='Lower C.L.')
    plt.axvline(H0_up, color='green', linestyle='--', label='Upper C.L.')
    plt.xlabel(r"$H_0 ~ [\mathrm{km} / \mathrm{s} / \mathrm{Mpc}]$")
    plt.ylabel(r'Probability Density')
    plt.legend()
    plt.savefig('H0_posterior_distribution_flat.png', dpi=500)
    plt.close()


def save_samples(samples, fname_out):
    names = ['mu_M101', 'mu_M1337', 'mu_N0691', 'mu_N1015', 'mu_N0105',
             'mu_N1309', 'mu_N1365', 'mu_N1448', 'mu_N1559', 'mu_N2442',
             'mu_N2525', 'mu_N2608', 'mu_N3021', 'mu_N3147', 'mu_N3254',
             'mu_N3370', 'mu_N3447', 'mu_N3583', 'mu_N3972', 'mu_N3982',
             'mu_N4038', 'mu_N4424', 'mu_N4536', 'mu_N4639', 'mu_N4680',
             'mu_N5468', 'mu_N5584', 'mu_N5643', 'mu_N5728', 'mu_N5861',
             'mu_N5917', 'mu_N7250', 'mu_N7329', 'mu_N7541', 'mu_N7678',
             'mu_N0976', 'mu_U9391', 'Delta_mu_N4258', 'M_H1_W',
             'Delta_mu_LMC', 'mu_M31', 'b_W', 'MB0', 'Z_W', 'undefined',
             'Delta_zp', 'log10_H0']
    H0 = 10**(samples['theta'][:, -1] / 5.)
    print(f"Writing H0 samples to `{fname_out}`.")
    with File(fname_out, 'w') as f:
        for i, name in enumerate(names):
            f.create_dataset(name, data=samples['theta'][:, i])
        f.create_dataset('H0', data=H0)
    print(f"Samples saved to `{fname_out}`.")


if __name__ == "__main__":
    fdir = "/mnt/extraspace/rstiskalek/catalogs/SH0ES"
    fname_out = "SH0ES_samples.hdf5"
    num_warmup = 5000
    num_samples = 150_000
    rng_key = random.PRNGKey(0)

    kwargs = load_data(fdir)
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,)
    mcmc.run(rng_key, **kwargs)
    mcmc.print_summary()
    samples = mcmc.get_samples()

    plot_H0(samples)
    save_samples(samples, fname_out)

    # flat_samples = np.array(samples['theta'])
    # param_names = [f"theta_{i}" for i in range(flat_samples.shape[1])]
    # samples_for_getdist = MCSamples(samples=flat_samples, names=param_names,
    # labels=param_names)
    # g = plots.get_subplot_plotter()
    # g.triangle_plot(samples_for_getdist, filled=True)
    # g.export("triangle_plot.pdf")
