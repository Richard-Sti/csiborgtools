# Copyright (C) 2022 Richard Stiskalek, Deaglan Bartlett
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
Tools for splitting the particles and a clump object.
"""


import numpy
from os import remove
from warnings import warn
from os.path import join
from tqdm import trange
from ..io import nparts_to_start_ind


def clump_with_particles(particle_clumps, clumps):
    """
    Count how many particles does each clump have.

    Parameters
    ----------
    particle_clumps : 1-dimensional array
        Array of particles' clump IDs.
    clumps : structured array
        The clumps array.

    Returns
    -------
    with_particles : 1-dimensional array
        Array of whether a clump has any particles.
    """
    return numpy.isin(clumps["index"], particle_clumps)


def distribute_halos(Nsplits, clumps):
    """
    Evenly distribute clump indices to smaller splits. Clumps should only be
    clumps that contain particles.

    Parameters
    ----------
    Nsplits : int
        Number of splits.
    clumps : structured array
        The clumps array.

    Returns
    -------
    splits : 2-dimensional array
        Array of starting and ending indices of each CPU of shape `(Njobs, 2)`.
    """
    # Make sure these are unique IDs
    indxs = clumps["index"]
    if indxs.size > numpy.unique((indxs)).size:
        raise ValueError("`clump_indxs` constains duplicate indices.")
    Ntotal = indxs.size
    Njobs_per_cpu = numpy.ones(Nsplits, dtype=int) * Ntotal // Nsplits
    # Split the remainder Ntotal % Njobs among the CPU
    Njobs_per_cpu[:Ntotal % Nsplits] += 1
    start = nparts_to_start_ind(Njobs_per_cpu)
    return numpy.vstack([start, start + Njobs_per_cpu]).T


def dump_split_particles(particles, particle_clumps, clumps, Nsplits,
                         dumpfolder, Nsim, Nsnap, verbose=True):
    """
    Save the data needed for each split so that a process does not have to load
    everything.

    Parameters
    ----------
    particles : structured array
        The particle array.
    particle_clumps : 1-dimensional array
        Array of particles' clump IDs.
    clumps : structured array
        The clumps array.
    Nsplits : int
        Number of times to split the clumps.
    dumpfolder : str
        Path to the folder where to dump the splits.
    Nsim : int
        CSiBORG simulation index.
    Nsnap : int
        Snapshot index.
    verbose : bool, optional
        Verbosity flag. By default `True`.

    Returns
    -------
    None
    """
    if particles.size != particle_clumps.size:
        raise ValueError("`particles` must correspond to `particle_clumps`.")
    # Calculate which clumps have particles
    with_particles = clump_with_particles(particle_clumps, clumps)
    clumps = numpy.copy(clumps)[with_particles]
    if verbose:
        warn(r"There are {:.4f}% clumps that have identified particles."
             .format(with_particles.sum() / with_particles.size * 100))

    # The starting clump index of each split
    splits = distribute_halos(Nsplits, clumps)
    fname = join(dumpfolder, "out_{}_snap_{}_{}.npz")

    iters = trange(Nsplits) if verbose else range(Nsplits)
    tot = 0
    for n in iters:
        # Lower and upper array index of the clumps array
        i, j = splits[n, :]
        # Clump indices in this split
        indxs = clumps["index"][i:j]
        hmin, hmax = indxs.min(), indxs.max()
        mask = (particle_clumps >= hmin) & (particle_clumps <= hmax)
        # Check number of clumps
        npart_unique = numpy.unique(particle_clumps[mask]).size
        if indxs.size > npart_unique:
            raise RuntimeError(
                "Split `{}` contains more unique clumps (`{}`) than there are "
                "unique particles' clump indices (`{}`)after removing clumps "
                "with no particles.".format(n, indxs.size, npart_unique))
        # Dump it!
        tot += mask.sum()
        fout = fname.format(Nsim, Nsnap, n)
        numpy.savez(fout, particles[mask], particle_clumps[mask], clumps[i:j])

    # There are particles whose clump ID is > 1 and have no counterpart in the
    # clump file. Therefore can save fewer particles, depending on the cut.
    if tot > particle_clumps.size:
        raise RuntimeError(
            "Num. of dumped particles `{}` is greater than the particle file "
            "size `{}`.".format(tot, particle_clumps.size))


def split_jobs(Njobs, Ncpu):
    """
    Split `Njobs` amongst `Ncpu`.

    Parameters
    ----------
    Njobs : int
        Number of jobs.
    Ncpu : int
        Number of CPUs.

    Returns
    -------
    jobs : list of lists of integers
        Outer list of each CPU and inner lists for CPU's jobs.
    """
    njobs_per_cpu, njobs_remainder = divmod(Njobs, Ncpu)
    jobs = numpy.arange(njobs_per_cpu * Ncpu).reshape((njobs_per_cpu, Ncpu)).T
    jobs = jobs.tolist()
    for i in range(njobs_remainder):
        jobs[i].append(njobs_per_cpu * Ncpu + i)

    return jobs


def load_split_particles(Nsplit, dumpfolder, Nsim, Nsnap, remove_split=False):
    """
    Load particles of a split saved by `dump_split_particles`.

    Parameters
    --------
    Nsplit : int
        Split index.
    dumpfolder : str
        Path to the folder where the splits were dumped.
    Nsim : int
        CSiBORG simulation index.
    Nsnap : int
        Snapshot index.
    remove_split : bool, optional
        Whether to remove the split file. By default `False`.

    Returns
    -------
    particles : structured array
        Particle array of this split.
    clumps_indxs : 1-dimensional array
        Array of particles' clump IDs of this split.
    clumps : 1-dimensional array
        Clumps belonging to this split.
    """
    fname = join(
        dumpfolder, "out_{}_snap_{}_{}.npz".format(Nsim, Nsnap, Nsplit))
    file = numpy.load(fname)
    particles, clump_indxs, clumps = (file[f] for f in file.files)
    if remove_split:
        remove(fname)
    return particles, clump_indxs, clumps


def pick_single_clump(n, particles, particle_clumps, clumps):
    """
    Get particles belonging to the `n`th clump in `clumps` arrays.

    Parameters
    ----------
    n : int
        Clump position in `clumps` array. Not its halo finder index!
    particles : structured array
        Particle array.
    particle_clumps : 1-dimensional array
        Array of particles' clump IDs.
    clumps : structured array
        Array of clumps.

    Returns
    -------
    sel_particles : structured array
        Particles belonging to the requested clump.
    sel_clump : array
        A slice of a `clumps` array corresponding to this clump. Must
        contain `["peak_x", "peak_y", "peak_z", "mass_cl"]`.
    """
    # Clump index on the nth position
    k = clumps["index"][n]
    # Mask of which particles belong to this clump
    mask = particle_clumps == k
    return particles[mask], clumps[n]


class Clump:
    """
    A clump (halo) object to handle the particles and their clump's data.

    Parameters
    ----------
    x : 1-dimensional array
        Particle coordinates along the x-axis.
    y : 1-dimensional array
        Particle coordinates along the y-axis.
    z : 1-dimensional array
        Particle coordinates along the z-axis.
    m : 1-dimensional array
        Particle masses.
    x0 : float
        Clump center coordinate along the x-axis.
    y0 : float
        Clump center coordinate along the y-axis.
    z0 : float
        Clump center coordinate along the z-axis.
    clump_mass : float
        Mass of the clump.
    vx : 1-dimensional array
        Particle velocity along the x-axis.
    vy : 1-dimensional array
        Particle velocity along the y-axis.
    vz : 1-dimensional array
        Particle velocity along the z-axis.
    """
    _r = None
    _pos = None
    _clump_pos = None
    _clump_mass = None
    _vel = None

    def __init__(self, x, y, z, m, x0, y0, z0, clump_mass=None,
                 vx=None, vy=None, vz=None):
        self.pos = (x, y, z, x0, y0, z0)
        self.clump_pos = (x0, y0, z0)
        self.clump_mass = clump_mass
        self.vel = (vx, vy, vz)
        self.m = m

    @property
    def pos(self):
        """
        Cartesian particle coordinates centered at the clump.

        Returns
        -------
        pos : 2-dimensional array
            Array of shape `(n_particles, 3)`.
        """
        return self._pos

    @pos.setter
    def pos(self, X):
        """Sets `pos` and calculates radial distance."""
        x, y, z, x0, y0, z0 = X
        self._pos = numpy.vstack([x - x0, y - y0, z - z0]).T
        self.r = numpy.sum(self.pos**2, axis=1)**0.5

    @property
    def Npart(self):
        """
        Number of particles associated with this clump.

        Returns
        -------
        Npart : int
            Number of particles.
        """
        return self.r.size

    @property
    def clump_pos(self):
        """
        Cartesian clump coordinates.

        Returns
        -------
        pos : 1-dimensional array
            Array of shape `(3, )`.
        """
        return self._clump_pos

    @clump_pos.setter
    def clump_pos(self, pos):
        """Sets `clump_pos`. Makes sure it is the correct shape."""
        pos = numpy.asarray(pos)
        if pos.shape != (3,):
            raise TypeError("Invalid clump position `{}`".format(pos.shape))
        self._clump_pos = pos

    @property
    def clump_mass(self):
        """
        Clump mass.

        Returns
        -------
        mass : float
            Clump mass.
        """
        return self._clump_mass

    @clump_mass.setter
    def clump_mass(self, mass):
        """Sets `clump_mass`, making sure it is a float."""
        if not isinstance(mass, float):
            raise ValueError("`clump_mass` must be a float.")
        self._clump_mass = mass

    @property
    def vel(self):
        """
        Cartesian particle velocities. Throws an error if they are not set.

        Returns
        -------
        vel : 2-dimensional array
            Array of shape (`n_particles, 3`).
        """
        if self._vel is None:
            raise ValueError("Velocities `vel` have not been set.")
        return self._vel

    @vel.setter
    def vel(self, V):
        """Sets the particle velocities, making sure the shape is OK."""
        if any(v is None for v in V):
            return
        vx, vy, vz = V
        self._vel = numpy.vstack([vx, vy, vz]).T
        if self.pos.shape != self.vel.shape:
            raise ValueError("Different `pos` and `vel` arrays!")

    @property
    def m(self):
        """
        Particle masses.

        Returns
        -------
        m : 1-dimensional array
            Array of shape `(n_particles, )`.
        """
        return self._m

    @m.setter
    def m(self, m):
        """Sets particle masses `m`, ensuring it is the right size."""
        if not isinstance(m, numpy.ndarray) and m.size != self.r.size:
            raise TypeError("`r` and `m` must be equal size 1-dim arrays.")
        self._m = m

    @property
    def r(self):
        """
        Radial distance of particles from the clump peak.

        Returns
        -------
        r : 1-dimensional array
            Array of shape `(n_particles, )`
        """
        return self._r

    @r.setter
    def r(self, r):
        """Sets `r`. Again checks the shape."""
        if not isinstance(r, numpy.ndarray) and r.ndim == 1:
            raise TypeError("`r` must be a 1-dimensional array.")
        if not numpy.all(r >= 0):
            raise ValueError("`r` larger than zero.")
        self._r = r

    @property
    def total_particle_mass(self):
        """
        Total mass of all particles.

        Returns
        -------
        tot_mass : float
            The summed mass.
        """
        return numpy.sum(self.m)

    @property
    def mean_particle_pos(self):
        """
        Mean Cartesian particle coordinate. Not centered at the halo!

        Returns
        -------
        pos : 1-dimensional array
            Array of shape `(3, )`.
        """
        return numpy.mean(self.pos + self.clump_pos, axis=0)

    @classmethod
    def from_arrays(cls, particles, clump):
        """
        Initialises `Halo` from `particles` containing the relevant particle
        information and its `clump` information.

        Paramaters
        ----------
        particles : structured array
            Array of particles belonging to this clump. Must contain
            `["x", "y", "z", "M"]` and optionally also `["vx", "vy", "vz"]`.
        clump : array
            A slice of a `clumps` array corresponding to this clump. Must
            contain `["peak_x", "peak_y", "peak_z", "mass_cl"]`.

        Returns
        -------
        halo : `Halo`
            An initialised halo object.
        """
        x, y, z, m = (particles[p] for p in ["x", "y", "z", "M"])
        x0, y0, z0, cl_mass = (
            clump[p] for p in ["peak_x", "peak_y", "peak_z", "mass_cl"])
        try:
            vx, vy, vz = (particles[p] for p in ["vx", "vy", "vz"])
        except ValueError:
            vx, vy, vz = None, None, None
        return cls(x, y, z, m, x0, y0, z0, cl_mass, vx, vy, vz)
