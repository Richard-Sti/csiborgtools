# Copyright (C) 2024 Richard Stiskalek
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
"""Mock data generators."""
import numpy as np
from jax import numpy as jnp
from tqdm import trange

from ..field.interp import evaluate_cartesian_regular
from ..params import SPEED_OF_LIGHT
from ..utils import (fprint, galactic_to_radec_cartesian, radec_to_cartesian,
                     radec_to_galactic)
from .cosmography import ComovingDistance2Redshift, Distmod2Distance
from .selection import log_magnitude_selection


def reject_sample_TFR(gen, mean_eta, std_eta, mean_e_eta, a_TF, b_TF, c_TF,
                      sigma_TF, mean_e_mag, m1, m2, a, mag_min, mag_max):
    """Rejection sampling of the TFR linewidth and distance."""
    num_attempts = 0

    # First sample the linewidth, from which to compute the absolute magnitude.
    eta_true = gen.normal(mean_eta, std_eta)
    eta_obs = gen.normal(eta_true, mean_e_eta)

    absmag = a_TF + b_TF * eta_true
    if eta_true > 0:
        absmag += c_TF * eta_true**2

    # Now rejection sample to obtain a reasonable apparent magnitude.
    mag_xrange = jnp.linspace(mag_min, mag_max, 5000)
    Sm = np.exp(log_magnitude_selection(mag_xrange, m1, m2, a))

    zmin = 10**(0.6 * mag_min)
    zmax = 10**(0.6 * mag_max)

    while True:
        num_attempts += 1

        mag_true = 5 / 3 * np.log10(zmin + gen.uniform(0, 1) * (zmax - zmin))
        mag_obs = gen.normal(mag_true, mean_e_mag)

        if np.interp(mag_obs, mag_xrange, Sm) > gen.uniform(0, 1):
            break

    mu_TFR = mag_true - absmag

    # TODO: here add Malmquist
    mu = gen.normal(mu_TFR, sigma_TF)

    return eta_true, eta_obs, mag_true, mag_obs, mu, num_attempts


def mock_Carrick2MTF(velocity_field, boxsize, RA_2MTF, DEC_2MTF,
                     a_TF=-22.8, b_TF=-7.2, c_TF=0, sigma_TF=0.35, sigma_v=100,
                     Vext_mag=150, Vext_l=300, Vext_b=-4, h=1.0, beta=0.4,
                     mean_eta=0.069, std_eta=0.078, mean_e_eta=0.012,
                     mean_e_mag=0.044, m1_selection=11.206,
                     m2_selection=13.203, a_selection=-0.152,
                     a_TF_dipole_mag=0, a_TF_dipole_l=140, a_TF_dipole_b=30,
                     seed=42, Om0=0.3, Rmax_mask=150, mag_min=8, mag_max=18,
                     **kwargs):
    """
    Mock TFR catalogue build against the Carrick velocity field and the
    2MTF sky distribution to avoid recomputing the LOS velocities.
    """
    nsamples = len(RA_2MTF)
    distmod2distance = Distmod2Distance(Om0=Om0)
    dist2redshift = ComovingDistance2Redshift(Om0=Om0)

    # Convert Vext from ICRS to Galactic coordinates.
    Vext = Vext_mag * galactic_to_radec_cartesian(Vext_l, Vext_b)
    a_TF_dipole = a_TF_dipole_mag * galactic_to_radec_cartesian(
        a_TF_dipole_l, a_TF_dipole_b)

    truths = {"a": a_TF, "b": b_TF, "c": c_TF, "e_mu": sigma_TF,
              "sigma_v": sigma_v, "Vext": Vext, "a_TF_dipole": a_TF_dipole,
              "mean_eta": mean_eta, "std_eta": std_eta,
              "mean_e_eta": mean_e_eta, "mean_e_mag": mean_e_mag,
              "h": h, "beta": beta,
              "Vmag": Vext_mag, "Vl": Vext_l, "Vb": Vext_b,
              "a_TF_dipole_mag": a_TF_dipole_mag,
              "a_TF_dipole_l": a_TF_dipole_l, "a_TF_dipole_b": a_TF_dipole_b
              }

    gen = np.random.default_rng(seed)

    # The Carrick box is in the Galactic coordinates.
    l, b = radec_to_galactic(RA_2MTF, DEC_2MTF)
    gal_phi = np.deg2rad(l)
    gal_theta = np.pi / 2 - np.deg2rad(b)

    # Adjust the TFR zero-point if there is a dipole in it.
    if a_TF_dipole_mag > 0:
        rhat = radec_to_cartesian(
            np.vstack([np.ones_like(RA_2MTF), RA_2MTF, DEC_2MTF]).T)

        a_TF_dipole = np.asarray(a_TF_dipole)
        a_TF = a_TF + np.sum(rhat * a_TF_dipole[None, :], axis=1)
    else:
        a_TF = np.full(nsamples, a_TF)

    eta_true = np.zeros_like(RA_2MTF)
    eta_obs = np.zeros_like(RA_2MTF)
    mag_true = np.zeros_like(RA_2MTF)
    mag_obs = np.zeros_like(RA_2MTF)
    mu = np.zeros_like(RA_2MTF)
    num_attempts = np.zeros_like(RA_2MTF, dtype=int)

    fprint("starting rejection sampling.")
    for n in trange(nsamples, desc="Rejection sampling"):
        eta_true_, eta_obs_, mag_true_, mag_obs_, mu_, num_attempts_ = reject_sample_TFR(  # noqa
            gen, mean_eta, std_eta, mean_e_eta, a_TF[n], b_TF, c_TF, sigma_TF,
            mean_e_mag, m1_selection, m2_selection, a_selection, mag_min,
            mag_max)

        eta_true[n] = eta_true_
        eta_obs[n] = eta_obs_
        mag_true[n] = mag_true_
        mag_obs[n] = mag_obs_
        mu[n] = mu_
        num_attempts[n] = num_attempts_

    fprint("average number of attempts per draw is "
           f"{num_attempts.mean():.2f}.")

    if h != 1:
        raise RuntimeError("Currently only h = 1 is supported.")

    r = distmod2distance(mu)
    zcosmo = dist2redshift(r)

    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(zcosmo)):
        raise ValueError("Some distance moduli are outside the interpolation "
                         "range.")

    # Calculate the Cartesian coordinates of each galaxy. This is initially
    # centered at (0, 0, 0).
    pos = r * np.asarray([
        np.sin(gal_theta) * np.cos(gal_phi),
        np.sin(gal_theta) * np.sin(gal_phi),
        np.cos(gal_theta)])
    pos = pos.T
    pos_box = pos / boxsize + 0.5

    vel = evaluate_cartesian_regular(
        velocity_field[0], velocity_field[1], velocity_field[2],
        pos=pos_box, smooth_scales=None, method="linear")
    vel = beta * np.vstack(vel).T

    for i in range(3):
        vel[:, i] += Vext[i]

    # Compute the radial velocity.
    Vr = np.sum(vel * pos, axis=1) / np.linalg.norm(pos, axis=1)

    # The true redshift of the source.
    zCMB_true = (1 + zcosmo) * (1 + Vr / SPEED_OF_LIGHT) - 1
    zCMB_obs = gen.normal(zCMB_true, sigma_v / SPEED_OF_LIGHT)

    # These galaxies will be masked out when LOS is read it because they are
    # too far away.
    distance_mask = r < Rmax_mask
    truths["distance_mask"] = distance_mask
    print(f"{np.sum(~distance_mask)} galaxies are above {Rmax_mask} Mpc/h.")

    sample = {"RA": RA_2MTF,
              "DEC": DEC_2MTF,
              "z_CMB": zCMB_obs,
              "eta": eta_obs,
              "mag": mag_obs,
              "e_eta": np.ones(nsamples) * mean_e_eta,
              "e_mag": np.ones(nsamples) * mean_e_mag,
              "r": r,
              }

    return sample, truths
