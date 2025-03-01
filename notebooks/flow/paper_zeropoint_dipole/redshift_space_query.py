import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
SPEED_OF_LIGHT = 299_792.458


def interp_dist2redshift(dist, Om0=0.3, zmin_interp=0, zmax_interp=0.1,
                         npoints_interp=5000):
    """Convert comoving distance in `Mpc/h` to redshift."""
    cosmo = FlatLambdaCDM(H0=100, Om0=Om0)
    z_grid = np.linspace(zmin_interp, zmax_interp, npoints_interp)
    dist_grid = cosmo.comoving_distance(z_grid).value

    return np.interp(dist, dist_grid, z_grid, left=np.nan, right=np.nan)


def map_to_zobs(r, Vrad_norm, nhat, beta, Vext):
    """Compute the observed redshift."""
    zcosmo = interp_dist2redshift(r)

    Vrad = beta * Vrad_norm + np.dot(Vext, nhat)
    zobs = (1 + zcosmo) * (1 + Vrad / SPEED_OF_LIGHT) - 1
    return zobs, Vrad


r, Vrad_norm = np.loadtxt("virgo_los.txt").T
nhat = np.asarray([-0.96870904, -0.11465427, 0.22012997])

beta_fiducial = 0.43
Vext_fiducial = np.asarray([150, 300, -4])

beta_target = 0.3
Vext_target = np.asarray([100, 250, -50])
fname_plot = "los_example.png"

print(f"Fiducial: beta = {beta_fiducial}, Vext = {Vext_fiducial}")
print(f"Target: beta = {beta_target}, Vext = {Vext_target}")
print(f"nhat = {nhat}")

zobs_fiducial, Vrad_fiducial = map_to_zobs(
    r, Vrad_norm, nhat, beta_fiducial, Vext_fiducial)
zobs_target, Vrad_target = map_to_zobs(
    r, Vrad_norm, nhat, beta_target, Vext_target)


zobs_test = np.linspace(zobs_fiducial.min(), zobs_fiducial.max(), 1000)
Vrad_scaled = np.interp(zobs_test, zobs_fiducial, Vrad_fiducial)
Vrad_scaled = (Vrad_scaled - np.dot(Vext_fiducial, nhat)) / beta_fiducial * beta_target + np.dot(Vext_target, nhat)  # noqa


fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))

ax1.plot(zobs_fiducial, Vrad_fiducial,
         label=rf"Fiducial: $\beta = {beta_fiducial},\, \vec{{V}}_{{\rm ext}} = ({Vext_fiducial[0]}, {Vext_fiducial[1]}, {Vext_fiducial[2]}) ~ \mathrm{{km}} / \mathrm{{s}}$")  # noqa
ax1.plot(zobs_target, Vrad_target,
         label=rf"Target: $\beta = {beta_target},\, \vec{{V}}_{{\rm ext}} = ({Vext_target[0]}, {Vext_target[1]}, {Vext_target[2]}) ~ \mathrm{{km}} / \mathrm{{s}}$")  # noqa
ax1.plot(zobs_test, Vrad_scaled, label="Fiducial scaled to target")

ax1.set_ylabel(r"$V_{\rm rad} ~ [\mathrm{km} / \mathrm{s}]$")
ax1.legend()


Vrad_target_interp = np.interp(zobs_test, zobs_target, Vrad_target)
diff = Vrad_scaled - Vrad_target_interp

ax2.plot(zobs_test, diff, color="red")
ax2.axhline(0, color="black", linestyle="--", linewidth=0.5)
ax2.set_xlabel(r"$z_{\rm obs}$")
ax2.set_ylabel(r"$V_{\rm scaled} - V_{\rm target} ~ [\mathrm{km} / \mathrm{s}]$")  # noqa
ax2.set_ylim(*np.percentile(diff, [1, 99.]))

fig.tight_layout()

print(f"Saving a plot to `{fname_plot}`.")
plt.savefig(fname_plot, dpi=450)
plt.show()
