import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from tqdm import trange

gen = np.random.default_rng(42)


def select_void_h(kind):
    """Select 'little h' for void profile `kind`."""
    hs = {"mb": 0.7615, "gauss": 0.7724, "exp": 0.7725}
    try:
        return hs[kind]
    except KeyError:
        raise ValueError(f"Unknown void kind: `{kind}`.")

vr_grid = np.genfromtxt("/mnt/extraspace/rstiskalek/catalogs/IndranilVoid/SizeVariation_new/sizenumber010/vr_data/EXPprofile/v_pec_EXPprofile_rLG_40.dat")  # noqa
print(f"Read in a vr_grid of shape `{vr_grid.shape}`.")
r_grid = np.arange(0, 251).astype(float)
phi_grid = np.arange(0, 181).astype(float)
h = select_void_h("exp")

interpolation_method = "fast"   # "fast" or "slow"
nrepeat = 100                   # How many times to resample random points
npoints = 64**3                 # How many random points to sample

r_eval = [100, 150, 200, 250]   # Radii at which to evaluate the bulk flow

# Void constant velocity pointing towards (l, b)= (297, -4) in degrees.
vvoid = 500 * np.asarray([-0.4035093, 0.01363162, -0.91487399])

rgrid_min, rgrid_max = r_grid.min(), r_grid.max()
phi_grid_min, phi_grid_max = phi_grid.min(), phi_grid.max()


def rand_points_in_sphere(rmax, N):
    """Draw `N` random points in a sphere of radius `rmax`."""
    r = rmax * gen.random(N)**(1./3.)
    theta = np.arccos(2 * gen.random(N)-1)
    phi = 2 * np.pi * gen.random(N)
    return r[:, None] * np.vstack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)]).T


def interpolate_vrad(X, which="fast"):
    # Unit vector pointing towards (l, b) = (117, 4) in degrees.
    n_hat = np.asarray([0.4035093, -0.01363162, 0.91487399])

    # Unit vector pointing towards each galaxy.
    r = np.linalg.norm(X, axis=1)
    r_hat = X / r[:, None]

    # Angular separation of each point from the void axis.
    cos_phi = np.sum(r_hat * n_hat[None, :], axis=1)

    if which == "fast":
        nphi, nrad = vr_grid.shape
        r_normalized = (r - rgrid_min) / (rgrid_max - rgrid_min) * (nrad - 1)
        phi_normalized = np.arccos(cos_phi) * 180 / np.pi / (phi_grid_max - phi_grid_min) * (nphi - 1)  # noqa
        vrad = map_coordinates(
            vr_grid, np.vstack([phi_normalized, r_normalized]),
            order=1, mode='constant', cval=np.nan)
    elif which == "slow":
        vrad = RegularGridInterpolator((phi_grid, r_grid), vr_grid)(
            np.vstack([np.arccos(cos_phi) * 180 / np.pi, r]).T)
    else:
        raise ValueError(f"Unknown interpolation method `{which}`.")

    vrad += np.sum(r_hat * vvoid[None, :], axis=1)

    return vrad


def get_bulf_flow_magnitude_from_vrad(rmax):
    """
    Get the BF magnitude using Eq. 14 of  https://arxiv.org/abs/1808.07772
    """
    bulk_flow_estimate = np.full(nrepeat, np.nan, dtype=float)

    for n in trange(nrepeat):
        X = rand_points_in_sphere(rmax, npoints)
        vrad = interpolate_vrad(X, interpolation_method)
        bulk_flow_estimate[n] = np.linalg.norm(rmax**2 * np.mean(X * vrad[:, None] / np.linalg.norm(X, axis=1)[:, None]**3, axis=0))  # noqa

    return np.mean(bulk_flow_estimate), np.std(bulk_flow_estimate)


bf_mean = np.full_like(r_eval, np.nan, dtype=float)
bf_std = np.full_like(r_eval, np.nan, dtype=float)

for i, rmax in enumerate(r_eval):
    bf_mean[i], bf_std[i] = get_bulf_flow_magnitude_from_vrad(rmax)


print(bf_mean)
for i, rmax in enumerate(r_eval):
    print(f"rmax = {rmax}: {bf_mean[i]:.4f} +- {bf_std[i]:.4f}")


fname_plot = "void_bulkflow.png"

plt.figure()
plt.errorbar(np.asarray(r_eval) * h, bf_mean, yerr=bf_std,
             label="Void bulk flow")

plt.errorbar([150], [387], yerr=[28], fmt="o", label="Watkins")

plt.xlabel(r"$R ~ [\mathrm{Mpc} / h]$")
plt.ylabel(r"$B ~ [\mathrm{km} / \mathrm{s}]$")
plt.legend()
plt.tight_layout()

print(f"Saving plot to `{fname_plot}`.")
plt.savefig(fname_plot, dpi=300)
plt.close()
