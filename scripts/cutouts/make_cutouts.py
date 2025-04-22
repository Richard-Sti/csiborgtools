"""
Make plots of projected density fields from CSiBORG3 simulation data.
"""
import os
from os.path import join, exists
import numpy as np
import matplotlib.pyplot as plt
import h5py
import csiborgtools
import scienceplots  # noqa

os.environ["PATH"] = os.path.expanduser("~/texlive/bin/x86_64-linux") + ":" + os.environ["PATH"]  # noqa


def find_particles(step, x0, mask_width, boxsize, which_simulation, paths):
    print(f"[step {step}] Finding particles in mask...")
    if which_simulation == "csiborg3":
        reader = csiborgtools.read.CSiBORG3Snapshot(step, 130, paths)
    else:
        raise ValueError("Invalid simulation name.")

    pos = reader.coordinates()
    mass = reader.masses()

    mask_box = csiborgtools.read.find_boxed(pos, x0, mask_width, boxsize)
    pos = pos[mask_box]
    mass = mass[mask_box]

    pos -= x0
    pos += mask_width / 2

    print(f"[step {step}] Found {len(pos)} particles in region.")
    print(np.min(pos, axis=0), np.max(pos, axis=0))

    return pos, mass


def plot_density_projection(rho, mask_width, step, fdir_out, name):
    extent = (-mask_width / 2, mask_width / 2, -mask_width / 2, mask_width / 2)
    img = np.log10(1 + rho)

    vmin = 1.35
    vmax = np.percentile(img.flatten(), 99.9)
    fpath = join(fdir_out, f"{name}_density_step{step:03d}.png")

    print(f"[step {step}] Saving density projection plot to: {fpath}")
    with plt.style.context("science"):
        plt.figure(figsize=(6, 5))
        plt.imshow(img, extent=extent, origin="lower", aspect="auto",
                   cmap="inferno", vmin=vmin, vmax=vmax)
        plt.colorbar(
            label=r"$\log \rho_{\rm DM} ~ [h^2 M_\odot / \mathrm{kpc}^3]$",
            pad=0)
        plt.xlabel(r"$x ~ [\mathrm{Mpc} / h]$")
        plt.ylabel(r"$y ~ [\mathrm{Mpc} / h]$")
        plt.tight_layout()
        plt.savefig(fpath, dpi=500)
        plt.close()


def save_particles_to_hdf5(filepath, step, pos, mass, rho):
    print(f"[step {step}] Writing particle data and density field to: {filepath}")  # noqa
    with h5py.File(filepath, "a") as f:
        g = f.create_group(f"step_{step}")
        g.create_dataset("pos", data=pos)
        g.create_dataset("mass", data=mass)
        g.create_dataset("rho", data=rho)


def load_all_rho_from_hdf5(filepath, steps):
    print(f"Loading all density fields from: {filepath}")
    rhos = []
    with h5py.File(filepath, "r") as f:
        for step in steps:
            print(f"  - loading step_{step}")
            rhos.append(f[f"step_{step}"]["rho"][:])
    return np.stack(rhos)


if __name__ == "__main__":
    mask_width = 100
    boxsize = 681.
    x0 = [boxsize / 2, boxsize / 2, boxsize / 2]
    # x0 = csiborgtools.clusters["Virgo"].cartesian_pos(boxsize)
    # name = "Virgo_15hMpc"
    name = "inner_100"
    grid = 512
    which_simulation = "csiborg3"
    paths = csiborgtools.read.Paths(**csiborgtools.paths_rusty)
    steps = list(range(50))
    fdir_out = "/mnt/home/rstiskalek/ceph/CSiBORG/postprocessing/cutouts"
    os.makedirs(fdir_out, exist_ok=True)

    h5_path = join(fdir_out, f"{name}_cutouts.hdf5")
    if exists(h5_path):
        print(f"Removing existing file: {h5_path}")
        os.remove(h5_path)

    for step in steps:
        print(f"\n=== Processing step {step} ===")
        pos_box, mass_box = find_particles(
            step, x0, mask_width, boxsize, which_simulation, paths)

        density_obj = csiborgtools.field.DensityField(mask_width, "TSC")
        print(f"[step {step}] Computing projected density field...")
        rho = density_obj.density_2d(
            pos_box, mass_box, 2, grid=grid, verbose=True)

        plot_density_projection(rho, mask_width, step, fdir_out, name)
        save_particles_to_hdf5(h5_path, step, pos_box, mass_box, rho)

    # Now load all density fields and compute mean and std
    print("\n=== Computing summary statistics ===")
    all_rho = load_all_rho_from_hdf5(h5_path, steps)
    mean_rho = np.mean(all_rho, axis=0)
    std_rho = np.std(all_rho, axis=0)

    for arr, title, suffix in zip([mean_rho, std_rho],
                                  ["Mean", "Std"],
                                  ["mean", "std"]):
        img = np.log10(1 + arr)
        extent = (
            -mask_width / 2, mask_width / 2, -mask_width / 2, mask_width / 2)
        fname = f"{name}_{suffix}_rho.png"
        print(f"Saving {title.lower()} density field plot to: {join(fdir_out, fname)}")  # noqa

        if suffix == "mean":
            vmin = 1.35
            vmax = np.percentile(img.flatten(), 99.9)
        else:
            vmin, vmax = np.percentile(img.flatten(), [0.5, 99.5])

        with plt.style.context("science"):
            plt.figure(figsize=(6, 5))
            plt.imshow(img, extent=extent, origin="lower", aspect="auto",
                       cmap="inferno", vmin=vmin, vmax=vmax)
            plt.colorbar(label=r"$\log \rho_{\rm DM} ~ [h^2 M_\odot / \mathrm{kpc}^3]$")  # noqa
            plt.xlabel(r"$x ~ [\mathrm{Mpc} / h]$")
            plt.ylabel(r"$y ~ [\mathrm{Mpc} / h]$")
            plt.tight_layout()
            plt.savefig(join(fdir_out, fname), dpi=500)
            plt.close()
