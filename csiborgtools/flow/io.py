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

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from h5py import File

from ..params import SPEED_OF_LIGHT, simname2Omega_m
from ..utils import fprint, radec_to_galactic, radec_to_supergalactic
from .flow_model import PV_LogLikelihood
from .void_model import load_void_data, mock_void, select_void_h

H0 = 100  # km / s / Mpc


##############################################################################
#                             Data loader                                     #
###############################################################################


class DataLoader:
    """
    Data loader for the line of sight (LOS) interpolated fields and the
    corresponding catalogues.

    Parameters
    ----------
    simname : str
        Simulation name.
    ksim : int or list of int
        Index of the simulation to read in (not the IC index).
    catalogue : str
        Name of the catalogue with LOS objects.
    catalogue_fpath : str
        Path to the LOS catalogue file.
    paths : csiborgtools.read.Paths
        Paths object.
    ksmooth : int, optional
        Smoothing index.
    store_full_velocity : bool, optional
        Whether to store the full 3D velocity field. Otherwise stores only
        the radial velocity.
    verbose : bool, optional
        Verbose flag.
    """
    def __init__(self, simname, ksim, catalogue, catalogue_fpath, paths,
                 ksmooth=None, store_full_velocity=False, verbose=True):
        fprint("reading the catalogue,", verbose)
        self._cat, self._absmag_calibration = self._read_catalogue(
            catalogue, catalogue_fpath)
        self._catname = catalogue

        fprint("reading the interpolated field.", verbose)
        self._field_rdist, self._los_density, self._los_velocity = self._read_field(  # noqa
            simname, ksim, catalogue, ksmooth, paths)

        if "IndranilVoid" not in simname:
            if len(self._cat) != self._los_density.shape[1]:
                raise ValueError(
                    "The number of objects in the catalogue does not match "
                    "the number of objects in the field.")

            fprint("calculating the radial velocity.", verbose)
            nobject = self._los_density.shape[1]
            dtype = self._los_density.dtype
            num_sims = len(self._los_density)

        if simname in ["Carrick2015", "Lilow2024"]:
            # Carrick+2015 and Lilow+2024 are in galactic coordinates
            d1, d2 = radec_to_galactic(self._cat["RA"], self._cat["DEC"])
        elif simname in ["CF4", "CLONES"]:
            # CF4 is in supergalactic coordinates
            d1, d2 = radec_to_supergalactic(self._cat["RA"], self._cat["DEC"])
        else:
            d1, d2 = self._cat["RA"], self._cat["DEC"]

        if "IndranilVoid" in simname:
            self._los_radial_velocity = None
            self._los_velocity = None
        else:
            radvel = np.empty(
                (num_sims, nobject, len(self._field_rdist)), dtype)
            for k in range(num_sims):
                for i in range(nobject):
                    radvel[k, i, :] = radial_velocity_los(
                        self._los_velocity[k, :, i, ...], d1[i], d2[i])
            self._los_radial_velocity = radvel

        if not store_full_velocity:
            self._los_velocity = None

        self._Omega_m = simname2Omega_m(simname)

        # Normalize the CSiBORG & CLONES density by the mean matter density
        if "csiborg" in simname or simname == "CLONES":
            cosmo = FlatLambdaCDM(H0=H0, Om0=self._Omega_m)
            mean_rho_matter = cosmo.critical_density0.to("Msun/kpc^3").value
            mean_rho_matter *= self._Omega_m
            self._los_density /= mean_rho_matter

        # Since Carrick+2015 and CF4 provide `rho / <rho> - 1`
        if simname in ["Carrick2015", "CF4", "CF4gp"]:
            self._los_density += 1

        # But some CF4 delta values are < -1. Check that CF4 really reports
        # this.
        if simname in ["CF4", "CF4gp"]:
            self._los_density = np.clip(self._los_density, 1e-2, None,)

        # Lilow+2024 outside of the range data is NaN. Replace it with some
        # finite values. This is OK because the PV tracers are not so far.
        if simname == "Lilow2024":
            self._los_density[np.isnan(self._los_density)] = 1.
            self._los_radial_velocity[np.isnan(self._los_radial_velocity)] = 0.

        self._mask = np.ones(len(self._cat), dtype=bool)
        self._catname = catalogue

    @property
    def cat(self):
        """The distance indicators catalogue (structured array)."""
        return self._cat[self._mask]

    @property
    def absmag_calibration(self):
        """Returns the absolute magnitude calibration with masking applied."""
        if self._absmag_calibration is None:
            return None

        return {key: val[:, self._mask]
                for key, val in self._absmag_calibration.items()}

    @property
    def catname(self):
        """Catalogue name."""
        return self._catname

    @property
    def rdist(self):
        """Radial distances at which the field was interpolated."""
        return self._field_rdist

    @property
    def los_density(self):
        """
        Density field along the line of sight `(n_sims, n_objects, n_steps)`
        """
        return self._los_density[:, self._mask, ...]

    @property
    def los_velocity(self):
        """
        Velocity field along the line of sight `(n_sims, 3, n_objects,
        n_steps)`.
        """
        if self._los_velocity is None:
            raise ValueError("The 3D velocities were not stored.")

        return self._los_velocity[:, :, self._mask, ...]

    @property
    def los_radial_velocity(self):
        """
        Radial velocity along the line of sight `(n_sims, n_objects, n_steps)`.
        """
        return self._los_radial_velocity[:, self._mask, ...]

    def _read_field(self, simname, ksims, catalogue, ksmooth, paths):
        if "IndranilVoid" in simname:
            return None, None, None

        nsims = paths.get_ics(simname)
        if isinstance(ksims, int):
            ksims = [ksims]

        # For no-field read in Carrick+2015 but then zero it.
        if simname == "no_field":
            simname = "Carrick2015"
        to_wipe = simname == "no_field"

        if not all(0 <= ksim < len(nsims) for ksim in ksims):
            raise ValueError(f"Invalid simulation index: `{ksims}`")

        if "Pantheon+" in catalogue:
            fpath = paths.field_los(simname, "Pantheon+")
        elif "CF4_TFR" in catalogue:
            fpath = paths.field_los(simname, "CF4_TFR")
        elif "Carrick2MTFmock" in catalogue:
            fpath = paths.field_los(simname, "2MTF")
        else:
            fpath = paths.field_los(simname, catalogue)

        los_density = [None] * len(ksims)
        los_velocity = [None] * len(ksims)

        for n, ksim in enumerate(ksims):
            nsim = nsims[ksim]

            with File(fpath, 'r') as f:
                has_smoothed = True if f[f"density_{nsim}"].ndim > 2 else False
                if has_smoothed and (ksmooth is None or not isinstance(ksmooth, int)):  # noqa
                    raise ValueError("The output contains smoothed field but "
                                     "`ksmooth` is None. It must be provided.")

                indx = (..., ksmooth) if has_smoothed else (...)
                los_density[n] = f[f"density_{nsim}"][indx]
                los_velocity[n] = f[f"velocity_{nsim}"][indx]
                rdist = f[f"rdist_{nsim}"][...]

        los_density = np.stack(los_density)
        los_velocity = np.stack(los_velocity)

        if to_wipe:
            los_density = np.ones_like(los_density)
            los_velocity = np.zeros_like(los_velocity)

        return rdist, los_density, los_velocity

    def _read_catalogue(self, catalogue, catalogue_fpath):
        absmag_calibration = None

        if catalogue == "A2":
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                arr = np.empty(len(f["RA"]), dtype=dtype)
                for key in f.keys():
                    arr[key] = f[key][:]
        elif catalogue in ["LOSS", "Foundation", "SFI_gals", "2MTF",
                           "Pantheon+", "SFI_gals_masked", "SFI_groups",
                           "Pantheon+_groups", "Pantheon+_groups_zSN",
                           "Pantheon+_zSN"]:
            with File(catalogue_fpath, 'r') as f:
                if "Pantheon+" in catalogue:
                    grp = f["Pantheon+"]
                else:
                    grp = f[catalogue]

                dtype = [(key, np.float32) for key in grp.keys()]
                arr = np.empty(len(grp["RA"]), dtype=dtype)
                for key in grp.keys():
                    arr[key] = grp[key][:]
        elif "CB2_" in catalogue:
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                arr = np.empty(len(f["RA"]), dtype=dtype)
                for key in f.keys():
                    arr[key] = f[key][:]
        elif "IndranilVoidTFRMock" in catalogue:
            # The name can be e.g. "IndranilVoidTFRMock_exp_34_0", where the
            # first and second number are the LG observer index and random
            # seed.
            profile, rLG_index, seed = catalogue.split("_")[1:]
            rLG_index = int(rLG_index)
            seed = int(seed)
            rLG, vrad_data = load_void_data(profile, "vrad")
            h = select_void_h(profile)
            print(f"Mock observed galaxies for LG observer with index "
                  f"{rLG_index} at {rLG[rLG_index] * h} Mpc / h and "
                  f"seed {seed}.")
            mock_data = mock_void(vrad_data, rLG_index, profile, seed=seed)[0]

            # Convert the dictionary to a structured array
            dtype = [(key, np.float32) for key in mock_data.keys()]
            arr = np.empty(len(mock_data["RA"]), dtype=dtype)
            for key in mock_data.keys():
                arr[key] = mock_data[key]
        elif "Carrick2MTFmock" in catalogue:
            with File(catalogue_fpath, 'r') as f:
                keys_skip = ["mu_calibration", "e_mu_calibration"]
                dtype = [(key, np.float32) for key in f.keys()
                         if key not in keys_skip]
                arr = np.empty(len(f["RA"]), dtype=dtype)
                for key in f.keys():
                    if key not in keys_skip:
                        arr[key] = f[key][:]

                absmag_calibration = {
                    "mu_calibration": f["mu_calibration"][...],
                    "e_mu_calibration": f["e_mu_calibration"][...]}

        elif "UPGLADE" in catalogue:
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                arr = np.empty(len(f["RA"]), dtype=dtype)
                for key in f.keys():
                    if key == "mask":
                        continue

                    arr[key] = f[key][:]
        elif catalogue in ["CF4_GroupAll"] or "CF4_TFR" in catalogue:
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                dtype += [("DEC", np.float32)]
                arr = np.empty(len(f["RA"]), dtype=dtype)

                for key in f.keys():
                    arr[key] = f[key][:]
                arr["DEC"] = arr["DE"]

                if "CF4_TFR" in catalogue:
                    arr["RA"] *= 360 / 24
        else:
            raise ValueError(f"Unknown catalogue: `{catalogue}`.")

        return arr, absmag_calibration


###############################################################################
#                       Supplementary flow functions                          #
###############################################################################


def radial_velocity_los(los_velocity, ra, dec):
    """
    Calculate the radial velocity along the LOS from the 3D velocity
    along the LOS `(3, n_steps)`.
    """
    types = (float, np.float32, np.float64)
    if not isinstance(ra, types) and not isinstance(dec, types):
        raise ValueError("RA and dec must be floats.")

    if los_velocity.ndim != 2 and los_velocity.shape[0] != 3:
        raise ValueError("The shape of `los_velocity` must be (3, n_steps).")

    ra_rad, dec_rad = np.deg2rad(ra), np.deg2rad(dec)

    vx, vy, vz = los_velocity
    return (vx * np.cos(ra_rad) * np.cos(dec_rad)
            + vy * np.sin(ra_rad) * np.cos(dec_rad)
            + vz * np.sin(dec_rad))


##############################################################################
#                       Shortcut to create a model                           #
###############################################################################


def read_absolute_calibration(kind, data_length, calibration_fpath):
    """
    Read the absolute calibration for the CF4 TFR sample from LEDA but
    preprocessed by me. Missing values are replaced with NaN.

    Parameters
    ----------
    kind : str
        Calibration kind: `Cepheids`, `TRGB`, `SBF`, ...
    data_length : int
        Number of samples in CF4 TFR (should be 9,788).
    calibration_fpath : str
        Path to the preprocessed calibration file.

    Returns
    -------
    mu : 2-dimensional array of shape `(ncalib, ngalaxies)`
        Absolute calibration data.
    e_mu : 2-dimensional array of shape `(ncalib, ngalaxies)`
        Uncertainties of the absolute calibration.
    """
    raise RuntimeError("The read-in functions are not guaranteed to work "
                       "properly.")
    data = {}
    with File(calibration_fpath, 'r') as f:
        for key in f[kind].keys():
            x = f[kind][key][:]

            # Get rid of points without uncertainties
            x = x[~np.isnan(x[:, 1])]

            data[key] = x

    max_calib = max(len(val) for val in data.values())

    out = np.full((data_length, max_calib, 2), np.nan)
    for i in data.keys():
        out[int(i), :len(data[i]), :] = data[i]

    # Unpack from this the distsance modulus and its uncertainty.
    mu = out[:, :, 0].T
    e_mu = out[:, :, 1].T

    return mu, e_mu


def mask_fields(density, velocity, mask, return_none):
    """Shortcut to mask fields, unless they are `None`"""
    if return_none:
        return None, None

    return density[:, mask], velocity[:, mask]


def get_model(loader, zcmb_min=None, zcmb_max=None, mag_selection=None,
              wo_num_dist_marginalisation=False, absolute_calibration=None,
              calibration_fpath=None, void_kwargs=None):
    """
    Get a model and extract the relevant data from the loader.

    Parameters
    ----------
    loader : DataLoader
        DataLoader instance.
    zcmb_min : float, optional
        Minimum observed redshift in the CMB frame to include.
    zcmb_max : float, optional
        Maximum observed redshift in the CMB frame to include.
    mag_selection : dict, optional
        Magnitude selection parameters.
    wo_num_dist_marginalisation : bool, optional
        Whether to directly sample the distance without numerical
        marginalisation. in which case the tracers can be coupled by a
        covariance matrix. By default `False`.
    add_absolute_calibration : bool, optional
        Whether to add an absolute calibration for CF4 TFRs.
    calibration_fpath : str, optional
        Path to the file containing the absolute calibration of CF4 TFR.

    Returns
    -------
    model : NumPyro model
    """
    zcmb_min = 0.0 if zcmb_min is None else zcmb_min
    zcmb_max = np.infty if zcmb_max is None else zcmb_max

    if void_kwargs is None:
        los_overdensity = loader.los_density
        los_velocity = loader.los_radial_velocity
    else:
        los_overdensity = None
        los_velocity = None

    kind = loader._catname

    if void_kwargs is not None:
        rdist = void_kwargs.pop("rdist", None)
        if rdist is None:
            raise ValueError(
                "The radial distances must be provided for the void.")

        loader._field_rdist = rdist

    if absolute_calibration is not None and not ("CF4_TFR_" in kind or "Carrick2MTFmock" in kind):  # noqa
        raise ValueError("Absolute calibration supported only for either "
                         "the CF4 TFR sample or Carrick 2MTF mocks.")

    if kind in ["LOSS", "Foundation"]:
        keys = ["RA", "DEC", "z_CMB", "mB", "x1", "c", "e_mB", "e_x1", "e_c"]
        RA, dec, zCMB, mag, x1, c, e_mag, e_x1, e_c = (
            loader.cat[k] for k in keys)
        e_zCMB = None

        mask = (zCMB < zcmb_max) & (zCMB > zcmb_min)
        calibration_params = {"mag": mag[mask], "x1": x1[mask], "c": c[mask],
                              "e_mag": e_mag[mask], "e_x1": e_x1[mask],
                              "e_c": e_c[mask]}

        los_overdensity, los_velocity = mask_fields(
            los_overdensity, los_velocity, mask, void_kwargs is not None)

        model = PV_LogLikelihood(
            los_overdensity, los_velocity,
            RA[mask], dec[mask], zCMB[mask], e_zCMB, calibration_params,
            mag_selection, loader.rdist, loader._Omega_m, "SN",
            name=kind, void_kwargs=void_kwargs,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation)
    elif "Pantheon+" in kind:
        keys = ["RA", "DEC", "zCMB", "mB", "x1", "c", "biasCor_m_b", "mBERR",
                "x1ERR", "cERR", "biasCorErr_m_b", "zCMB_SN", "zCMB_Group",
                "zCMBERR"]

        RA, dec, zCMB, mB, x1, c, bias_corr_mB, e_mB, e_x1, e_c, e_bias_corr_mB, zCMB_SN, zCMB_Group, e_zCMB = (loader.cat[k] for k in keys)  # noqa
        mB -= bias_corr_mB
        e_mB = np.sqrt(e_mB**2 + e_bias_corr_mB**2)

        mask = (zCMB < zcmb_max) & (zCMB > zcmb_min)

        if kind == "Pantheon+_groups":
            mask &= np.isfinite(zCMB_Group)

        if kind == "Pantheon+_groups_zSN":
            mask &= np.isfinite(zCMB_Group)
            zCMB = zCMB_SN

        if kind == "Pantheon+_zSN":
            zCMB = zCMB_SN

        calibration_params = {"mag": mB[mask], "x1": x1[mask], "c": c[mask],
                              "e_mag": e_mB[mask], "e_x1": e_x1[mask],
                              "e_c": e_c[mask]}

        los_overdensity, los_velocity = mask_fields(
            los_overdensity, los_velocity, mask, void_kwargs is not None)

        model = PV_LogLikelihood(
            los_overdensity, los_velocity,
            RA[mask], dec[mask], zCMB[mask], e_zCMB[mask], calibration_params,
            mag_selection, loader.rdist, loader._Omega_m, "SN",
            name=kind, void_kwargs=void_kwargs,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation)
    elif kind in ["SFI_gals", "2MTF", "SFI_gals_masked"] or "IndranilVoidTFRMock" in kind or "Carrick2MTFmock" in kind:  # noqa
        keys = ["RA", "DEC", "z_CMB", "mag", "eta", "e_mag", "e_eta"]
        RA, dec, zCMB, mag, eta, e_mag, e_eta = (loader.cat[k] for k in keys)

        mask = (zCMB < zcmb_max) & (zCMB > zcmb_min)
        if "Carrick2MTFmock" in kind:
            # For the mock we only want to select objects with the '2M++'
            # volume.
            mask &= loader.cat["r"] < 130
            # The mocks are generated without Malmquist.
            fprint("disabling homogeneous and inhomogeneous Malmquist bias for the mock.")  # noqa
            with_homogeneous_malmquist = False
            with_inhomogeneous_malmquist = False
        else:
            with_homogeneous_malmquist = True
            with_inhomogeneous_malmquist = True

        calibration_params = {"mag": mag[mask], "eta": eta[mask],
                              "e_mag": e_mag[mask], "e_eta": e_eta[mask]}

        # Append the calibration data
        if "Carrick2MTFmock" in kind:
            absmag_calibration = loader.absmag_calibration

            # The shape of these is (`ncalibrators, nobjects`).
            mu_calibration = absmag_calibration["mu_calibration"][:, mask]
            e_mu_calibration = absmag_calibration["e_mu_calibration"][:, mask]
            # Auxiliary parameters.
            m = np.isfinite(mu_calibration)

            # NumPyro refuses to start if any inputs are not finite, so we
            # replace with some ficutive mean and very large standard
            # deviation.
            mu_calibration[~m] = 0.0
            e_mu_calibration[~m] = 1000.0

            calibration_params["mu_calibration"] = mu_calibration
            calibration_params["e_mu_calibration"] = e_mu_calibration
            calibration_params["is_finite_calibrator"] = m
            calibration_params["counts_calibrators"] = np.sum(m, axis=0)
            calibration_params["any_calibrator"] = np.any(m, axis=0)

        los_overdensity, los_velocity = mask_fields(
            los_overdensity, los_velocity, mask, void_kwargs is not None)

        model = PV_LogLikelihood(
            los_overdensity, los_velocity,
            RA[mask], dec[mask], zCMB[mask], None, calibration_params,
            mag_selection, loader.rdist, loader._Omega_m, "TFR", name=kind,
            void_kwargs=void_kwargs,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation,
            with_homogeneous_malmquist=with_homogeneous_malmquist,
            with_inhomogeneous_malmquist=with_inhomogeneous_malmquist)
    elif "CF4_TFR_" in kind:
        # The full name can be e.g. "CF4_TFR_not2MTForSFI_i" or "CF4_TFR_i".
        band = kind.split("_")[-1]
        if band not in ['g', 'r', 'i', 'z', 'w1', 'w2']:
            raise ValueError(f"Band `{band}` not recognized.")

        keys = ["RA", "DEC", "Vcmb", f"{band}", "lgWmxi", "elgWi",
                "not_matched_to_2MTF_or_SFI", "Qs", "Qw"]
        RA, dec, z_obs, mag, eta, e_eta, not_matched_to_2MTF_or_SFI, Qs, Qw = (
            loader.cat[k] for k in keys)
        l, b = radec_to_galactic(RA, dec)

        not_matched_to_2MTF_or_SFI = not_matched_to_2MTF_or_SFI.astype(bool)
        # NOTE: fiducial uncertainty until we can get the actual values.
        e_mag = 0.05 * np.ones_like(mag)

        z_obs /= SPEED_OF_LIGHT
        eta -= 2.5

        fprint("selecting only galaxies with mag > 5 and eta > -0.3.")
        mask = (mag > 5) & (eta > -0.3)
        fprint("selecting only galaxies with |b| > 7.5.")
        mask &= np.abs(b) > 7.5
        mask &= (z_obs < zcmb_max) & (z_obs > zcmb_min)

        if "not2MTForSFI" in kind:
            mask &= not_matched_to_2MTF_or_SFI
        elif "2MTForSFI" in kind:
            mask &= ~not_matched_to_2MTF_or_SFI

        fprint("employing a quality cut on the galaxies.")
        if "w" in band:
            mask &= Qw == 5
        else:
            mask &= Qs == 5

        calibration_params = {"mag": mag[mask], "eta": eta[mask],
                              "e_mag": e_mag[mask], "e_eta": e_eta[mask]}

        # Read the absolute calibration
        mu_calibration, e_mu_calibration = read_absolute_calibration(
            absolute_calibration, len(RA), calibration_fpath)

        # The shape of these is (`ncalibrators, nobjects`).
        mu_calibration = mu_calibration[:, mask]
        e_mu_calibration = e_mu_calibration[:, mask]
        # Auxiliary parameters.
        m = np.isfinite(mu_calibration)

        # NumPyro refuses to start if any inputs are not finite, so we
        # replace with some ficutive mean and very large standard
        # deviation.
        mu_calibration[~m] = 0.0
        e_mu_calibration[~m] = 1000.0

        calibration_params["mu_calibration"] = mu_calibration
        calibration_params["e_mu_calibration"] = e_mu_calibration
        calibration_params["is_finite_calibrator"] = m
        calibration_params["counts_calibrators"] = np.sum(m, axis=0)
        calibration_params["any_calibrator"] = np.any(m, axis=0)

        los_overdensity, los_velocity = mask_fields(
            los_overdensity, los_velocity, mask, void_kwargs is not None)

        model = PV_LogLikelihood(
            los_overdensity, los_velocity,
            RA[mask], dec[mask], z_obs[mask], None, calibration_params,
            mag_selection, loader.rdist, loader._Omega_m, "TFR", name=kind,
            void_kwargs=void_kwargs,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation)
    elif kind in ["CF4_GroupAll"]:
        # Note, this for some reason works terribly.
        keys = ["RA", "DE", "Vcmb", "DMzp", "eDM"]
        RA, dec, zCMB, mu, e_mu = (loader.cat[k] for k in keys)

        zCMB /= SPEED_OF_LIGHT
        mask = (zCMB < zcmb_max) & (zCMB > zcmb_min) & np.isfinite(mu)

        # The distance moduli in CF4 are most likely given assuming h = 0.75
        mu += 5 * np.log10(0.75)

        calibration_params = {"mu": mu[mask], "e_mu": e_mu[mask]}

        los_overdensity, los_velocity = mask_fields(
            los_overdensity, los_velocity, mask, void_kwargs is not None)

        model = PV_LogLikelihood(
            los_overdensity, los_velocity,
            RA[mask], dec[mask], zCMB[mask], None, calibration_params,
            mag_selection,  loader.rdist, loader._Omega_m, "simple",
            name=kind, void_kwargs=void_kwargs,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation)
    else:
        raise ValueError(f"Catalogue `{kind}` not recognized.")

    fprint(f"selected {np.sum(mask)}/{len(mask)} galaxies in catalogue `{kind}`")  # noqa

    return model
