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
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
from h5py import File
from warnings import warn

from ..params import SPEED_OF_LIGHT, simname2Omega_m
from ..utils import fprint, radec_to_galactic, radec_to_supergalactic
from .flow_model import PV_LogLikelihood
# from .void_model import load_void_size_variation, mock_void, select_void_h
from ..read import read_pantheonplus_data

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
        self._is_no_field = "no_field" in simname

        fprint("reading the catalogue,", verbose=verbose)
        self._cat, self._absmag_calibration = self._read_catalogue(
            catalogue, catalogue_fpath)
        self._catname = catalogue

        fprint("reading the interpolated field.", verbose=verbose)
        self._field_rdist, self._los_density, self._los_velocity = self._read_field(  # noqa
            simname, ksim, catalogue, ksmooth, paths)

        if "IndranilVoid" not in simname:
            if len(self._cat) != self._los_density.shape[1]:
                raise ValueError(
                    "The number of objects in the catalogue does not match "
                    "the number of objects in the field.")

            fprint("calculating the radial velocity.", verbose=verbose)
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

        if "no_field" in simname:
            ngal = len(self._cat)
            dr = 0.25
            if len(simname.split("_")) != 3:
                raise ValueError(f"Invalid simulation name: `{simname}`.")
            rmax = float(simname.split("_")[-1])
            rdist = np.arange(1, rmax, dr)
            fprint(f"setting the `no_field` radial distances from {rdist.min()} to {rmax} Mpc/h in {len(rdist)} steps.")  # noqa

            los_density = np.ones((1, ngal, len(rdist)))
            los_velocity = np.zeros((1, 3, ngal, len(rdist)))
            return rdist, los_density, los_velocity

        nsims = paths.get_ics(simname, subsample=True)
        if isinstance(ksims, int):
            ksims = [ksims]

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
                           "SFI_gals_masked", "SFI_groups"]:
            with File(catalogue_fpath, 'r') as f:
                grp = f[catalogue]

                dtype = [(key, np.float32) for key in grp.keys()]
                arr = np.empty(len(grp["RA"]), dtype=dtype)
                for key in grp.keys():
                    arr[key] = grp[key][:]
        elif "Pantheon+" in catalogue:
            fname_covmat = catalogue_fpath.replace(".dat", "_STAT+SYS.cov")
            fname_pecvel_covmat = catalogue_fpath.replace(".dat", "_122221_VPEC.cov")  # noqa

            arr, C, Csysvpec = read_pantheonplus_data(
                catalogue_fpath, fname_covmat, fname_pecvel_covmat)

            self._covmat = C
            self._covmat_sysvpec = Csysvpec

        elif "CB2_" in catalogue:
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                arr = np.empty(len(f["RA"]), dtype=dtype)
                for key in f.keys():
                    arr[key] = f[key][:]
        elif "IndranilVoidTFRMock" in catalogue:
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                arr = np.empty(len(f["RA"]), dtype=dtype)
                for key in f.keys():
                    arr[key] = f[key][:]
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
        elif catalogue == "SDSS-FP":
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                dtype += [("DEC", np.float32), ("RA", np.float32)]
                arr = np.empty(len(f["Ra"]), dtype=dtype)
                for key in f.keys():
                    arr[key] = f[key][:]

                arr["DEC"] = arr["Dec"]
                arr["RA"] = arr["Ra"]
        elif catalogue == "CF4_test_points":
            with File(catalogue_fpath, 'r') as f:
                dtype = [(key, np.float32) for key in f.keys()]
                dtype += [("DEC", np.float32), ]
                arr = np.empty(len(f["RA"]), dtype=dtype)
                for key in f.keys():
                    arr[key] = f[key][:]

                arr["DEC"] = arr["dec"]
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


def get_model(loader, zcmb_min=None, zcmb_max=None, selection=None,
              wo_num_dist_marginalisation=False, absolute_calibration=None,
              calibration_fpath=None, void_kwargs=None, dust_model=None,
              remove_CF4_outliers=None):
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
    selection : dict, optional
        Magnitude selection parameters.
    wo_num_dist_marginalisation : bool, optional
        Whether to directly sample the distance without numerical
        marginalisation. in which case the tracers can be coupled by a
        covariance matrix. By default `False`.
    add_absolute_calibration : bool, optional
        Whether to add an absolute calibration for CF4 TFRs.
    calibration_fpath : str, optional
        Path to the file containing the absolute calibration of CF4 TFR.
    void_kwargs : dict, optional
        Keyword arguments for the void model.
    dust_model : str, optional
        Choice of a dust model, currently only supported for CF4 TFR WISE
        bands. Can provide comma-separeted dust maps, in which case they dust
        map choise is marginalised over. Overwrites the default dust model
    remove_CF4_outliers : bool, optional
        Whether to remove the CF4 outlier.

    Returns
    -------
    model : NumPyro model
    """
    zcmb_min = 0.0 if zcmb_min is None else zcmb_min
    zcmb_max = np.infty if zcmb_max is None else zcmb_max

    with_inhomogeneous_malmquist = True
    if loader._is_no_field:
        with_inhomogeneous_malmquist = False

    if void_kwargs is None:
        los_overdensity = loader.los_density
        los_velocity = loader.los_radial_velocity
    else:
        los_overdensity = None
        los_velocity = None

    kind = loader._catname

    if void_kwargs is not None:
        try:
            rdist = void_kwargs["rdist"]
        except KeyError as e:
            raise ValueError(
                "The radial distances must be provided for the void.") from e

        loader._field_rdist = rdist

    if absolute_calibration is not None and not ("CF4_TFR_" in kind or "Carrick2MTFmock" in kind):  # noqa
        raise ValueError("Absolute calibration supported only for either "
                         "the CF4 TFR sample or Carrick 2MTF mocks.")

    if "CF4_TFR_w" not in kind and dust_model is not None:
        raise ValueError("Changes to the dust model are supported only for "
                         "CF4 TFR WISE samples.")

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
            selection, loader.rdist, loader._Omega_m, "SN",
            name=kind, void_kwargs=void_kwargs,
            with_inhomogeneous_malmquist=with_inhomogeneous_malmquist,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation)
    elif kind == "Pantheon+":
        keys = ["RA", "DEC", "zCMB", "zCMBERR", "m_b_corr"]

        RA, dec, zCMB, e_zCMB, m_b = (loader.cat[k] for k in keys)

        covmat = loader._covmat - loader._covmat_sysvpec

        mask = np.ones(len(RA), dtype=bool)
        mask &= (zCMB < zcmb_max) & (zCMB > zcmb_min)
        covmat = covmat[mask][:, mask]

        dmu = find_covmat_regul(covmat)
        fprint(f"regularising the covariance matrix with `{dmu}`.")
        covmat += dmu * np.eye(covmat.shape[0])

        if not np.all(np.linalg.eigvals(covmat) > 0):
            raise ValueError("The covariance matrix is not positive definite.")

        calibration_params = {"mag": m_b[mask], "mag_covmat": covmat}
        los_overdensity, los_velocity = mask_fields(
            los_overdensity, los_velocity, mask, void_kwargs is not None)

        model = PV_LogLikelihood(
            los_overdensity, los_velocity,
            RA[mask], dec[mask], zCMB[mask], e_zCMB[mask], calibration_params,
            selection, loader.rdist, loader._Omega_m, "SN_calibrated",
            name=kind, void_kwargs=void_kwargs,
            with_inhomogeneous_malmquist=with_inhomogeneous_malmquist,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation)
    elif kind in ["SFI_gals", "2MTF", "SFI_gals_masked"] or "IndranilVoidTFRMock" in kind or "Carrick2MTFmock" in kind:  # noqa
        keys = ["RA", "DEC", "z_CMB", "mag", "eta", "e_mag", "e_eta"]
        RA, dec, zCMB, mag, eta, e_mag, e_eta = (loader.cat[k] for k in keys)

        mask = (zCMB < zcmb_max) & (zCMB > zcmb_min)
        if "Carrick2MTFmock" in kind:
            # For the mock we only want to select objects with the '2M++'
            # volume.
            if not loader._is_no_field:
                mask &= loader.cat["r"] < 150
            # The mocks are generated without Malmquist.
            fprint("disabling homogeneous and inhomogeneous Malmquist bias for the mock.")  # noqa
            with_homogeneous_malmquist = False
            with_inhomogeneous_malmquist &= False
        elif "IndranilVoidTFRMock" in kind:
            fprint("disabling homogeneous bias for the mock.")  # noqa
            with_homogeneous_malmquist = True
        else:
            with_homogeneous_malmquist = True
            with_inhomogeneous_malmquist &= True

        calibration_params = {"mag": mag[mask], "eta": eta[mask],
                              "e_mag": e_mag[mask], "e_eta": e_eta[mask]}

        # Append the calibration data
        if "Carrick2MTFmock" in kind:
            absmag_calibration = loader.absmag_calibration

            # The shape of these is (`ncalibrators, nobjects`).
            mu_calibration = absmag_calibration["mu_calibration"][:, mask]
            e_mu_calibration = absmag_calibration["e_mu_calibration"][:, mask]

            m = np.any(np.isfinite(mu_calibration), axis=0)
            print(f"Only {m.sum()} out of {len(m)} galaxies have at least "
                  "one calibrator.")

            # print(f"Selecting only {m.sum()} out of {len(m)} calibrators.")
            calibration_indxs = np.hstack(
                [np.where(np.isfinite(mu_calibration[i]))[0]
                 for i in range(len(mu_calibration))])

            mu_calibration = np.hstack(
                [mu_calibration[i][np.isfinite(mu_calibration[i])]
                 for i in range(len(mu_calibration))])
            e_mu_calibration = np.hstack(
                [e_mu_calibration[i][np.isfinite(e_mu_calibration[i])]
                 for i in range(len(e_mu_calibration))])

            calibration_params["mu_calibration"] = mu_calibration
            calibration_params["e_mu_calibration"] = e_mu_calibration
            calibration_params["calibration_indxs"] = calibration_indxs

        los_overdensity, los_velocity = mask_fields(
            los_overdensity, los_velocity, mask, void_kwargs is not None)

        model = PV_LogLikelihood(
            los_overdensity, los_velocity,
            RA[mask], dec[mask], zCMB[mask], None, calibration_params,
            selection, loader.rdist, loader._Omega_m, "TFR", name=kind,
            void_kwargs=void_kwargs,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation,
            with_homogeneous_malmquist=with_homogeneous_malmquist,
            with_inhomogeneous_malmquist=with_inhomogeneous_malmquist)
    elif "CF4_TFR_" in kind:
        # The full name can be e.g. "CF4_TFR_not2MTForSFI_i" or "CF4_TFR_i".
        band = kind.split("_")[-1]
        if band not in ['g', 'r', 'i', 'z', 'w1', 'w2']:
            raise ValueError(f"Band `{band}` not recognized.")

        keys = ["RA", "DEC", "Vcmb", f"{band}", "lgWmxi", "elgWi", "Qs", "Qw",
                "inc_e", "pgc"]
        RA, dec, z_obs, mag, eta, e_eta, Qs, Qw, e_inc, pgc = (
            loader.cat[k] for k in keys)
        l, b = radec_to_galactic(RA, dec)

        # Fiducial values set after asking Rhsan Kourkchi.
        e_mag = 0.05 * np.ones_like(mag)

        z_obs /= SPEED_OF_LIGHT
        eta -= 2.5

        fprint("selecting only galaxies with mag > 5 and eta > -0.3.")
        mask = (mag > 5) & (eta > -0.3)
        fprint("selecting only galaxies with |b| > 7.5.")
        mask &= np.abs(b) > 7.5
        mask &= (z_obs < zcmb_max) & (z_obs > zcmb_min)

        if remove_CF4_outliers:
            warn("Using local paths to retrieve the outlier files.",
                 RuntimeWarning)
            fprint("removing the CF4 outliers.")
            i_outliers = np.genfromtxt(
                "/mnt/extraspace/rstiskalek/catalogs/PV/CF4_i_outliers.csv",
                skip_header=1, delimiter=",", usecols=[0], dtype=int)
            w1_outliers = np.genfromtxt(
                "/mnt/extraspace/rstiskalek/catalogs/PV/CF4_W1_outliers.csv",
                skip_header=1, delimiter=",", usecols=[0], dtype=int)
            outliers = np.concatenate([i_outliers, w1_outliers])
            is_outlier = np.isin(pgc, outliers)
            mask &= ~is_outlier

        if band in ["w1", "w2"] and dust_model is not None:
            fprint(f"switching the dust model to `{dust_model}`.")

            # Read off the correction that was applied to the magnitudes.
            Ab_default = loader.cat[f"A_{band}"]

            # Check if we have multiple dust maps to marginalise over.
            dust_model = dust_model.split(",")
            fprint(f"adding the following dust models: `{dust_model}`.")
            ebv = np.full((len(dust_model), len(mag)), np.nan)

            if len(dust_model) > 1:
                raise RuntimeError(
                    "Multiple dust models are not supported. NumPyro raises "
                    "error when sampling a discrete variable, the "
                    "log-likelihood will need to be rewritten to numerically "
                    "marginalise instead.")

            for i, dust_model_i in enumerate(dust_model):
                if dust_model_i == "default":
                    ebv[i] = Ab_default / (0.186 if band == "w1" else 0.123)
                else:
                    ebv[i] = read_dustmap(RA, dec, dust_model_i)

                if not np.all(np.isfinite(ebv[i])):
                    raise ValueError("Found non-finite E(B-V) values for the "
                                     f"dust map `{dust_model_i}`.")

            # Remove the original dust correction, the new one is applied on
            # the fly.
            mag += Ab_default
        else:
            ebv = np.full_like(mag, np.nan)

        if "not2MTForSFI" in kind or "2MTForSFI" in kind:
            raise NotImplementedError("Unmatching the 2MTF and SFI samples "
                                      "is not supported.")

        if "notSDSS" in kind:
            mask &= Qs < 5

        fprint("employing a quality cut on the galaxies.")
        if "w" in band:
            mask &= Qw == 5
        else:
            mask &= Qs == 5

        m = np.isfinite(mag[mask])
        if not np.all(m):
            raise ValueError(f"Some magnitudes are not finite, {np.sum(~m)}.")

        calibration_params = {"mag": mag[mask], "eta": eta[mask],
                              "e_mag": e_mag[mask], "e_eta": e_eta[mask],
                              "ebv": ebv[..., mask]}

        # Read the absolute calibration
        if absolute_calibration is not None:
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
            selection, loader.rdist, loader._Omega_m, "TFR", name=kind,
            void_kwargs=void_kwargs,
            with_inhomogeneous_malmquist=with_inhomogeneous_malmquist,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation,
            dust_model=dust_model)
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
            selection,  loader.rdist, loader._Omega_m, "simple",
            name=kind, void_kwargs=void_kwargs,
            with_inhomogeneous_malmquist=with_inhomogeneous_malmquist,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation)
    elif kind in ["SDSS-FP"]:
        Msun = 4.65

        # We want to read in the group redshifts, instead of the galaxy
        # redshifts to suppress the noise due to small-scale velocities.
        keys = ["Ra", "Dec", "gczcmb", "rad", "erad", "boa", "eboa",
                "sig", "esig", "plate", "rmag", "ermag", "Exr", "kcr",
                "gczcmb", "czh", "r", "er", "s", "es", "i", "ei", "Sn"]

        (RA, dec, zCMB, rdev, e_rdev, boa, e_boa, sig, e_sig, SDSS_plate,
         rmag, e_rmag, Ar, kcr, gzCMB, zhel, r, er, s, es, i,
         ei, Sn) = (loader.cat[k] for k in keys)

        # Convert from velocity to redshift.
        zCMB = zCMB.astype(float) / SPEED_OF_LIGHT
        gzCMB = gzCMB.astype(float) / SPEED_OF_LIGHT
        zhel = zhel.astype(float) / SPEED_OF_LIGHT

        # Aperture size in arcseconds, depending on SDSS plate number.
        theta_aperture = np.ones_like(RA) * 1.5
        theta_aperture[SDSS_plate >= 3510] = 1.0

        # Precompute the effective size along with its propagated error.
        theta_eff = rdev * np.sqrt(boa)
        e_theta_eff = theta_eff * np.sqrt(
            (e_rdev / rdev)**2 + (e_boa / boa)**2 / 4)

        e_log_theta_eff = e_theta_eff / theta_eff
        e_log_sig = e_sig / sig

        # Constant composed of several terms that enter the effective
        # brightness calculation.
        K = 0.4 * (Msun - 0.85 * gzCMB + kcr + Ar)

        mask = (zCMB < zcmb_max) & (zCMB > zcmb_min)
        calibration_params = {
            "theta_eff": theta_eff[mask], "e_theta_eff": e_theta_eff[mask],
            "sig": sig[mask], "e_sig": e_sig[mask],
            "log_theta_aperture": np.log10(theta_aperture[mask]),
            "rmag": rmag[mask],
            "e_rmag": e_rmag[mask], "K": K[mask],
            "e_log_theta_eff": e_log_theta_eff[mask],
            "e_log_sig": e_log_sig[mask],
            "zhel": zhel[mask],
            "r": r[mask], "e_r": er[mask],
            "s": s[mask], "e_s": es[mask],
            "i": i[mask], "e_i": ei[mask],
            "Sn": Sn[mask],
            }

        los_overdensity, los_velocity = mask_fields(
            los_overdensity, los_velocity, mask, void_kwargs is not None)

        model = PV_LogLikelihood(
            los_overdensity, los_velocity,
            RA[mask], dec[mask], zCMB[mask], None, calibration_params,
            selection, loader.rdist, loader._Omega_m, "FP", name=kind,
            void_kwargs=void_kwargs,
            with_inhomogeneous_malmquist=with_inhomogeneous_malmquist,
            wo_num_dist_marginalisation=wo_num_dist_marginalisation)
    else:
        raise ValueError(f"Catalogue `{kind}` not recognized.")

    fprint(f"selected {np.sum(mask)}/{len(mask)} galaxies in catalogue `{kind}`")  # noqa

    return model


def read_dustmap(RA, dec, model):
    """Read off `E(B-V)` at `RA` and `dec` for a given `model`."""
    coords = SkyCoord(RA, dec, unit="deg", frame="icrs")

    if model == "SFD":
        try:
            from dustmaps.sfd import SFDQuery
        except ImportError:
            raise ImportError("Cannot import `dustmaps`. Please install it.")
        query = SFDQuery()
    elif model == "CSFD":
        try:
            from dustmaps.csfd import CSFDQuery
        except ImportError:
            raise ImportError("Cannot import `dustmaps`. Please install it.")
        query = CSFDQuery()
    elif model == "Planck2013":
        try:
            from dustmaps.planck import PlanckQuery
        except ImportError:
            raise ImportError("Cannot import `dustmaps`. Please install it.")
        query = PlanckQuery()
    elif model == "Planck2016":
        try:
            from dustmaps.planck import PlanckGNILCQuery
        except ImportError:
            raise ImportError("Cannot import `dustmaps`. Please install it.")
        query = PlanckGNILCQuery()
    else:
        raise ValueError(f"Unsupported model: `{model}`.")

    return np.asarray(query(coords), dtype=np.float32)


###############################################################################
#                         Supplementary functions                             #
###############################################################################


def find_covmat_regul(C, dx_init=0, dx_step=0.001, dx_max=0.15, verbose=True):
    """
    Find a regularisation term for a covariance matrix `C` (so that all
    eigenvalues are positive) by adding a constant diagonal term.
    """
    dx = dx_init

    if verbose:
        print(f"Finding a regularisation term for C with shape {C.shape}...")

    while True:
        eigval = np.linalg.eigvals(C + np.diag([dx] * C.shape[0]))

        if np.all(eigval.real > 0):
            break
        else:
            dx += dx_step

        if dx > dx_max:
            raise ValueError("No valid solution found.")

    return dx
