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
from astropy.cosmology import FlatLambdaCDM

from ..field.interp import evaluate_cartesian_regular
from ..params import SPEED_OF_LIGHT
from ..utils import (galactic_to_radec_cartesian, radec_to_cartesian,
                     radec_to_galactic)

###############################################################################
#                        Mock Carrick observations                            #
###############################################################################


def interp_distmod2redshift(distmod, Om0=0.3, zmin_interp=1e-4,
                            zmax_interp=0.5, npoints_interp=1000):
    """
    Convert distance modulus to redshift. Calls `astropy` to generate a grid
    of redshifts and distance moduli and then interpolates between distance
    modulus and log(redshift). Assumes `h = 1`.

    With the default settings, the mapping is accurate to better than 10 m / s
    over the entire range.
    """
    cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
    z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
    distmod_grid = cosmo.distmod(z_grid).value

    return np.exp(np.interp(distmod, distmod_grid, np.log(z_grid),
                            left=np.nan, right=np.nan))


def interp_distmod2dist(distmod, Om0=0.3, zmin_interp=1e-4,
                        zmax_interp=0.5, npoints_interp=1000):
    """
    Convert distance modulus to redshift. Calls `astropy` to generate a grid
    of redshifts and distance moduli and then interpolates between distance
    modulus and log(redshift). Assumes `h = 1`.

    With the default settings, the mapping is accurate to better than 1 kpc / h
    over the entire range.
    """
    cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
    z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)

    distmod_grid = cosmo.distmod(z_grid).value
    dist_grid = cosmo.comoving_distance(z_grid).value

    return np.exp(np.interp(distmod, distmod_grid, np.log(dist_grid)))


def mock_Carrick2MTF(velocity_field, boxsize, RA_2MTF, DEC_2MTF,
                     a_TF=-22.8, b_TF=-7.2, c_TF=0, sigma_TF=0.35, sigma_v=100,
                     Vext_mag=150, Vext_l=300, Vext_b=-4, h=1.0, beta=0.4,
                     mean_eta=0.069, std_eta=0.078, mean_e_eta=0.012,
                     mean_mag=10.31, std_mag=0.83, mean_e_mag=0.044,
                     add_calibration=False, sigma_calibration=0.05,
                     a_TF_dipole_mag=0, a_TF_dipole_l=140, a_TF_dipole_b=30,
                     calibration_max_percentile=10,
                     calibration_rand_fraction=0.5, nrepeat_calibration=1,
                     seed=42, Om0=0.3, verbose=True, **kwargs):
    """
    Mock TFR catalogue build against the Carrick velocity field and the
    2MTF sky distribution to avoid recomputing the LOS velocities.
    """
    nsamples = len(RA_2MTF)

    # Convert Vext from ICRS to Galactic coordinates.
    Vext = Vext_mag * galactic_to_radec_cartesian(Vext_l, Vext_b)
    a_TF_dipole = a_TF_dipole_mag * galactic_to_radec_cartesian(
        a_TF_dipole_l, a_TF_dipole_b)

    truths = {"a": a_TF, "b": b_TF, "c": c_TF, "e_mu": sigma_TF,
              "sigma_v": sigma_v, "Vext": Vext, "a_TF_dipole": a_TF_dipole,
              "mean_eta": mean_eta, "std_eta": std_eta,
              "mean_mag": mean_mag, "std_mag": std_mag,
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

    # Sample the linewidth of each galaxy from a Gaussian distribution to mimic
    # the MNR procedure.
    eta_true = gen.normal(mean_eta, std_eta, nsamples)
    eta_obs = gen.normal(eta_true, mean_e_eta)

    # Subtract the mean of the observed linewidths, so that they are
    # centered around zero. For consistency subtract from both observed
    # and true values.
    eta_mean_sampled = np.mean(eta_obs)
    eta_true -= eta_mean_sampled
    eta_obs -= eta_mean_sampled

    # Sample the magnitude from some Gaussian distribution to replicate MNR.
    mag_true = gen.normal(mean_mag, std_mag, nsamples)
    mag_obs = gen.normal(mag_true, mean_e_mag)

    # Calculate the 'true' distance modulus and redshift from the TFR distance.
    if a_TF_dipole_mag > 0:
        rhat = radec_to_cartesian(
            np.vstack([np.ones_like(RA_2MTF), RA_2MTF, DEC_2MTF]).T)

        a_TF_dipole = np.asarray(a_TF_dipole)
        a_TF = a_TF + np.sum(rhat * a_TF_dipole[None, :], axis=1)

    # If h != 1, then these distance modulii are in physical units.
    mu_TFR = mag_true - (a_TF + b_TF * eta_true + c_TF * eta_true**2)
    mu_true = gen.normal(mu_TFR, sigma_TF)
    # This is the distance modulus in units of little h.
    mu_true_h = mu_true + 5 * np.log10(h)

    # Convert the true distance modulus to true distance and cosmological
    # redshift. The distance is in Mpc/h because the box is in Mpc / h.
    zcosmo = interp_distmod2redshift(mu_true_h, Om0)
    r = interp_distmod2dist(mu_true_h, Om0)

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

    # We make the cut in observed redshift, cutting in the true distance yields
    # biases.
    if add_calibration:
        p = np.percentile(zCMB_obs, calibration_max_percentile)
        ks = np.where(zCMB_obs < p)[0]
        nsel = int(calibration_rand_fraction * len(ks))
        if verbose:
            print(f"Assigning calibration to {nsel}/{nsamples} galaxies.")

        mu_calibration = np.full((nrepeat_calibration, nsamples), np.nan)
        e_mu_calibration = np.full((nrepeat_calibration, nsamples), np.nan)

        for n in range(nrepeat_calibration):
            ks_n = gen.choice(ks, nsel, replace=False)

            mu_calibration[n, ks_n] = gen.normal(mu_true[ks_n], sigma_calibration)  # noqa
            e_mu_calibration[n, ks_n] = np.ones(len(ks_n)) * sigma_calibration
    else:
        mu_calibration = np.full((nrepeat_calibration, nsamples), np.nan)
        e_mu_calibration = np.full((nrepeat_calibration, nsamples), np.nan)

    # These galaxies will be masked out when LOS is read it because they are
    # too far away.
    distance_mask = r < 150
    truths["distance_mask"] = distance_mask
    print(f"{np.sum(~distance_mask)} galaxies are above 150 Mpc/h.")

    sample = {"RA": RA_2MTF,
              "DEC": DEC_2MTF,
              "z_CMB": zCMB_obs,
              "eta": eta_obs,
              "mag": mag_obs,
              "e_eta": np.ones(nsamples) * mean_e_eta,
              "e_mag": np.ones(nsamples) * mean_e_mag,
              "mu_true": mu_true,
              "mu_TFR": mu_TFR,
              "mu_calibration": mu_calibration,
              "e_mu_calibration": e_mu_calibration,
              "r": r,
              }

    return sample, truths
