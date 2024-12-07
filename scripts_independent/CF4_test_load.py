from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from tqdm import trange

kind = "density"
Nmax = 20


def load_data_single(nsim, flip_axes):
    """Load a single realisation."""
    folder = "/mnt/extraspace/rstiskalek/catalogs/CF4/CF4gp_23avr24_256-z008_test_100_realizations"  # noqa

    if kind == "density":
        fpath = join(
            folder,
            f"CF4gp_23avr24_256-z008_test_realization{nsim}_delta.fits")
        rho = fits.open(fpath)[0].data

        if flip_axes:
            rho = np.swapaxes(rho, 0, 2)

        return rho
    elif kind == "velocity":
        fpath = join(
            folder,
            f"CF4gp_23avr24_256-z008_test_realization{nsim}_velocity.fits")
        vx, vy, vz = fits.open(fpath)[0].data
        if flip_axes:
            vx, vy, vz = np.swapaxes(vx, 0, 2), np.swapaxes(vy, 0, 2), np.swapaxes(vz, 0, 2)  # noqa
        return 52 * np.stack([vx, vy, vz], axis=0)
    else:
        raise ValueError(f"Unknown field kind: `{kind}`.")


def load_data(flip_axes):
    """Load all realisations and stack."""
    desc = "Loading data (flipped axes)" if flip_axes else "Loading data"
    for i, n in enumerate(trange(1, Nmax + 1, desc=desc)):
        data_ = load_data_single(n, flip_axes)

        if i == 0:
            data = np.zeros((Nmax, *data_.shape), dtype=data_.dtype)

        data[i] = data_

    return data


rho = load_data(flip_axes=False)
rho_flipped = load_data(flip_axes=True)

# Average over realisations
rho = np.mean(rho, axis=0)
rho_flipped = np.mean(rho_flipped, axis=0)


# Make a plot to compare
fig, axs = plt.subplots(1, 2, figsize=(10, 5),)

# Plot the first image
cax0 = axs[0].imshow(rho[..., 128],
                     cmap="RdBu_r", extent=[-500, 500, -500, 500],)
axs[0].set_title("Without flipping")

# Plot the second image
cax1 = axs[1].imshow(rho_flipped[..., 128], origin="lower",
                     cmap="RdBu_r", extent=[-500, 500, -500, 500])
axs[1].set_title("With flipping")

# Set axis labels
for n in range(2):
    axs[n].set_xlabel(r"$\mathrm{SGX} ~ [\mathrm{Mpc} / h]$")
    axs[n].scatter(0, 0, c="k", s=30, marker="x")
axs[0].set_ylabel(r"$\mathrm{SGY} ~ [\mathrm{Mpc} / h]$")
axs[1].set_xlabel(r"$\mathrm{SGX} ~ [\mathrm{Mpc} / h]$")

# Add a colorbar to the first subplot
cbar0 = fig.colorbar(cax0, ax=axs[0], orientation='vertical', fraction=0.046,
                     pad=0.04)
cbar0.set_label(r'$\delta$')

# Add a colorbar to the second subplot
cbar1 = fig.colorbar(cax1, ax=axs[1], orientation='vertical', fraction=0.046,
                     pad=0.04)
cbar1.set_label(r'$\delta$')

# Adjust layout to avoid overlapping elements
fig.tight_layout()

# Save the figure
fname = f"CF4_mean_{kind}.png"
print(f"Saving to {fname}")
fig.savefig(fname, dpi=450)
plt.close()
