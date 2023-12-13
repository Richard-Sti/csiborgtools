# Copyright (C) 2023 Richard Stiskalek
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


def add_initial_snapshot(nsim, simname, halo_finder, verbose):
    """
    Sort the initial snapshot particles according to their final snapshot and
    add them to the final snapshot's HDF5 file.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    fname = paths.processed_output(nsim, simname, halo_finder)

    if simname == "csiborg":
        partreader = csiborgtools.read.CSiBORGReader(paths)
    else:
        partreader = csiborgtools.read.QuijoteReader(paths)

    fprint(f"processing simulation `{nsim}`.", verbose)
    if simname == "csiborg":
        nsnap0 = 1
    elif simname == "quijote":
        nsnap0 = -1
    else:
        raise ValueError(f"Unknown simulation `{simname}`.")

    fprint("loading and sorting the initial PID.", verbose)
    sort_indxs = numpy.argsort(partreader.read_snapshot(nsnap0, nsim, "pid"))

    fprint("loading the final particles.", verbose)
    with h5py.File(fname, "r") as f:
        sort_indxs_final = f["snapshot_final/pid"][:]
        f.close()

    fprint("sorting the particles according to the final snapshot.", verbose)
    sort_indxs_final = numpy.argsort(numpy.argsort(sort_indxs_final))
    sort_indxs = sort_indxs[sort_indxs_final]

    del sort_indxs_final
    collect()

    fprint("loading and sorting the initial particle position.", verbose)
    pos = partreader.read_snapshot(nsnap0, nsim, "pos")[sort_indxs]

    del sort_indxs
    collect()

    # In Quijote some particles are position precisely at the edge of the
    # box. Move them to be just inside.
    if simname == "quijote":
        mask = pos >= 1
        if numpy.any(mask):
            spacing = numpy.spacing(pos[mask])
            assert numpy.max(spacing) <= 1e-5
            pos[mask] -= spacing

    fprint(f"dumping particles for `{nsim}` to `{fname}`.", verbose)
    with h5py.File(fname, "r+") as f:
        if "snapshot_initial" in f.keys():
            del f["snapshot_initial"]
        group = f.create_group("snapshot_initial")
        group.attrs["header"] = "Initial snapshot data."
        dset = group.create_dataset("pos", data=pos)
        dset.attrs["header"] = "DM particle positions in box units."

        f.close()