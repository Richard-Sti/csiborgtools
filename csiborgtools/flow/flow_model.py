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
"""
Validation of the CSiBORG velocity field against PV measurements. Based on [1].

References
----------
[1] https://arxiv.org/abs/1912.09383.
"""
from datetime import datetime

import numpy as np
import numpyro
import numpyro.distributions as dist
from h5py import File
from jax import numpy as jnp
from tqdm import tqdm, trange

SPEED_OF_LIGHT = 299792.458  # km / s


def t():
    """Shortcut to get the current time."""
    return datetime.now().strftime("%H:%M:%S")


###############################################################################
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
    catalogue : str
        Name of the catalogue with LOS objects.
    los_catalogue_fpath : str
        Path to the LOS catalogue file.
    paths : csiborgtools.read.Paths
        Paths object.
    store_full_velocity : bool, optional
        Whether to store the full 3D velocity field. Otherwise stores only
        the radial velocity.
    """
    def __init__(self, simname, catalogue, catalogue_fpath, paths,
                 store_full_velocity=False):
        print(f"{t()}: reading the catalogue.")
        self._cat = self._read_catalogue(catalogue, catalogue_fpath)
        self._catname = catalogue

        print(f"{t()}: reading the interpolated field.")
        self._field_rdist, self._los_density, los_velocity = self._read_field(
            simname, catalogue, paths)

        if store_full_velocity:
            self._los_velocity = los_velocity
        else:
            self._los_velocity = None

        if len(self._cat) != len(self._los_density):
            raise ValueError("The number of objects in the catalogue does not "
                             "match the number of objects in the field.")

        print(f"{t()}: calculating the radial velocity.")
        nobject, nsim = self._los_density.shape[:2]

        radvel = np.empty((nobject, nsim, len(self._field_rdist)),
                          los_velocity.dtype)
        for i in trange(nobject):
            RA, dec = self._cat[i]["RA"], self._cat[i]["DEC"]
            for j in range(nsim):
                radvel[i, j, :] = radial_velocity_los(los_velocity[i, j, ...],
                                                      RA, dec)
        self._los_radial_velocity = radvel

    @property
    def cat(self):
        """
        The distance indicators catalogue.

        Returns
        -------
        structured array
        """
        return self._cat

    @property
    def catname(self):
        """
        Name of the catalogue.

        Returns
        -------
        str
        """
        return self._catname

    @property
    def rdist(self):
        """
        Radial distances where the field was interpolated for each object.

        Returns
        -------
        1-dimensional array
        """
        return self._field_rdist

    @property
    def los_density(self):
        """
        Density field along the line of sight.

        Returns
        ----------
        3-dimensional array of shape (n_objects, n_simulations, n_steps)
        """
        return self._los_density

    @property
    def los_velocity(self):
        """
        Velocity field along the line of sight.

        Returns
        -------
        4-dimensional array of shape (n_objects, n_simulations, 3, n_steps)
        """
        if self._los_velocity is None:
            raise ValueError("The 3D velocities were not stored.")
        return self._los_velocity

    @property
    def los_radial_velocity(self):
        """
        Radial velocity along the line of sight.

        Returns
        -------
        3-dimensional array of shape (n_objects, n_simulations, n_steps)
        """
        return self._los_radial_velocity

    def _read_field(self, simname, catalogue, paths):
        """Read in the interpolated field."""
        out_density = None
        out_velocity = None
        nsims = paths.get_ics(simname)
        with File(paths.field_los(simname, catalogue), 'r') as f:
            for i, nsim in enumerate(tqdm(nsims)):
                if out_density is None:
                    nobject, nstep = f[f"density_{nsim}"].shape
                    out_density = np.empty(
                        (nobject, len(nsims), nstep), dtype=np.float32)
                    out_velocity = np.empty(
                        (nobject, len(nsims), 3, nstep), dtype=np.float32)

                out_density[:, i, :] = f[f"density_{nsim}"][:]
                out_velocity[:, i, :, :] = f[f"velocity_{nsim}"][:].reshape(nobject, 3, nstep)  # noqa

            rdist = f[f"rdist_{nsims[0]}"][:]

        return rdist, out_density, out_velocity

    def _read_catalogue(self, catalogue, catalogue_fpath):
        """
        Read in the distance indicator catalogue.
        """
        with File(catalogue_fpath, 'r') as f:
            if catalogue == "LOSS" or catalogue == "Foundation":
                grp = f[catalogue]

                dtype = [(key, np.float32) for key in grp.keys()]
                arr = np.empty(len(grp["RA"]), dtype=dtype)
                for key in grp.keys():
                    arr[key] = grp[key][:]
            else:
                raise ValueError(f"Unknown catalogue: `{catalogue}`.")

        return arr


###############################################################################
#                          Supplementary functions                            #
###############################################################################


def radial_velocity_los(los_velocity, ra, dec):
    """
    Calculate the radial velocity along the line of sight.

    Parameters
    ----------
    los_velocity : 2-dimensional array of shape (3, n_steps)
        Line of sight velocity field.
    ra, dec : floats
        Right ascension and declination of the line of sight in degrees.

    Returns
    -------
    1-dimensional array of shape (n_steps)
    """
    types = (float, np.float32, np.float64)
    if not isinstance(ra, types) and not isinstance(dec, types):
        raise ValueError("RA and dec must be floats.")

    if los_velocity.ndim != 2 and los_velocity.shape[0] != 3:
        raise ValueError("The shape of `los_velocity` must be (3, n_steps).")

    ra_rad = ra / 180 * np.pi
    dec_rad = dec / 180 * np.pi
    vx, vy, vz = los_velocity

    return (vx * np.cos(ra_rad) * np.cos(dec_rad)
            + vy * np.sin(ra_rad) * np.cos(dec_rad)
            + vz * np.sin(dec_rad))


###############################################################################
#                           JAX Flow model                                    #
###############################################################################


def simps(y, dx):
    """
    Simpson's rule 1D integration, assuming that the number of steps is even
    and that the step size is constant.

    Parameters
    ----------
    y : 1-dimensional array
        Function values.
    dx : float
        Step size.

    Returns
    -------
    float
    """
    if len(y) % 2 != 0:
        raise ValueError("The number of steps must be even.")

    return dx / 3 * jnp.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2])


def dist2redshift(dist, Omega_m):
    """
    Convert comoving distance to cosmological redshift if the Universe is
    flat and z << 1.

    Parameters
    ----------
    dist : float or 1-dimensional array
        Comoving distance in `Mpc / h`.
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    float or 1-dimensional array
    """
    H0 = 100
    eta = 3 * Omega_m / 2
    return 1 / eta * (1 - (1 - 2 * H0 * dist / SPEED_OF_LIGHT * eta)**0.5)


def dist2distmodulus(dist, Omega_m):
    """
    Convert comoving distance to distance modulus, assuming z << 1.

    Parameters
    ----------
    dist : float or 1-dimensional array
        Comoving distance in `Mpc / h`.
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    float or 1-dimensional array
    """
    zcosmo = dist2redshift(dist, Omega_m)
    luminosity_distance = dist * (1 + zcosmo)
    return 5 * jnp.log10(luminosity_distance) + 25


def project_Vext(Vext_x, Vext_y, Vext_z, RA, dec):
    """
    Project the external velocity onto the line of sight along direction
    specified by RA/dec.

    Parameters
    ----------
    Vext_x, Vext_y, Vext_z : floats
        Components of the external velocity.
    RA, dec : floats
        Right ascension and declination in degrees.

    Returns
    -------
    float
    """
    RA_rad = RA / 180 * np.pi
    dec_rad = dec / 180 * np.pi

    return (Vext_x * np.cos(RA_rad) * np.cos(dec_rad)
            + Vext_y * np.sin(RA_rad) * np.cos(dec_rad)
            + Vext_z * np.sin(dec_rad))


def predict_zobs(dist, beta, Vext_radial, vpec_radial, Omega_m):
    """
    Predict the observed redshift at a given comoving distance given some
    velocity field.

    Parameters
    ----------
    dist : float
        Comoving distance in `Mpc / h`.
    beta : float
        Velocity bias parameter.
    Vext_radial : float
        Radial component of the external velocity along the LOS.
    vpec_radial : float
        Radial component of the peculiar velocity along the LOS.
    Omega_m : float
        Matter density parameter.

    Returns
    -------
    float
    """
    zcosmo = dist2redshift(dist, Omega_m)
    return ((1 + zcosmo) * (1 + (beta * vpec_radial + Vext_radial) / SPEED_OF_LIGHT) - 1)  # noqa


class FlowModel:
    """
    ?


    """

    def __init__(self, loader,
                 Omega_m,
                 sigma_v_dist=dist.Uniform(low=0, high=1000),
                 beta_dist=dist.Uniform(low=0, high=5),
                 Vext_dist=dist.Uniform(low=-1000, high=1000)):

        self._beta_dist = beta_dist
        self._sigma_v_dist = sigma_v_dist
        self._Vext_dist = Vext_dist

        self._zobs = loader.cat["zobs"]
        self._nobjects = len(loader.cat)

        self._Omega_m = Omega_m
        self._loader = loader

        self._r = loader.rdist
        self._mu = dist2distmodulus(self._r, self._Omega_m)
        # TODO Add a check to make sure these are all equal and there is even
        # number of points
        self._dr = self._r[1] - self._r[0]

        self._nsim = None


        self._correct_malmquist = True


        self._los_density = None
        self._los_velocity = None


    def __call__(self):
        # First off we sample all parameters.
        sigma_v = None
        e_mu_intrinsic = None
        beta = numpyro.sample("beta", self._beta_dist)
        # Sample the velocity uncertainty
        sigma_v = numpyro.sample("sigma_v", self._sigma_v_dist)
        # Sample the external velocity
        Vext_x = numpyro.sample("Vext_x", self._dist_Vext)
        Vext_y = numpyro.sample("Vext_y", self._dist_Vext)
        Vext_z = numpyro.sample("Vext_z", self._dist_Vext)

        # Calculate the distance modulus for the catalogue that we are using.
        if self._loader.catname == "A2":
            mag_cal = None
            alpha_cal = None
            beta_cal = None
            mu = (self._cat["mB"] - mag_cal
                  + alpha_cal * self._cat["x1"]
                  - beta_cal * self._cat["c"])
            squared_e_mu = (self._cat["e_mB"]**2
                            + alpha_cal**2 * self._cat["e_x1"]**2
                            + beta_cal**2 * self._cat["e_c"]**2)
        else:
            raise ValueError(f"Unknown catalogue: `{self._loader.catname}`.")

        squared_e_mu += e_mu_intrinsic**2

        ll_tot = 1
        for i in range(self._nsim):
            ll_sim = 1
            for j in range(self._nobjects):
                # Project the external velocity for this galaxy.
                Vext_rad = project_Vext(
                    Vext_x, Vext_y, Vext_z, self._cat[j]["RA"],
                    self._cat[j]["DEC"])

                # Because of a linear bias we don't need to normalize the
                # density field, and the bias cancels in the likelihood.
                ptilde = jnp.exp(-0.5 * self._mu - mu[j]**2 / squared_e_mu)
                ptilte *= self._los_density[j, i, :]

                if self._correct_malmquist:
                    ptilde *= self._r**2

                zobs_pred = predict_zobs(
                    self._r, beta, Vext_rad, self._los_velocity[j, i],
                    self._Omega_m)

                dczobs = SPEED_OF_LIGHT * (self._cat["z_CMB"][j] - zobs_pred)
                ll_zobs = 1 / jnp.sqrt(2 * jnp.pi * sigma_v**2)
                ll_zobs *= jnp.exp(-0.5 * (dczobs / sigma_v**2))

                ll_sim *= simps(ptilde * ll_zobs, self._dr)
                ll_sim /= simps(ptilde, self._dr)

            ll_tot += ll_sim

        ll_tot /= self._nsim

        log_ll = jnp.log(ll_tot)


