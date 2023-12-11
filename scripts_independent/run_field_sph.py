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
"""
Script to write the SLURM submission script and submit it to the queue to
calculate the SPH density & velocity field.
"""
from os import system


def write_submit(chain_index, kind, resolution, nthreads):
    if kind not in ["main", "random", "varysmall"]:
        raise RuntimeError(f"Unknown kind `{kind}`.")

    txt = f"""#!/bin/sh

#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task={nthreads}
#SBATCH --mem-per-cpu=7000
#SBATCH -J SPH_{chain_index}
#SBATCH -o output_{chain_index}_%J.out
#SBATCH -e error_{chain_index}_%J.err
#SBATCH -p cosma8-serial
#SBATCH -A dp016
#SBATCH -t 04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=richard.stiskalek@physics.ox.ac.uk


module purge
module load intel_comp/2019
module load intel_mpi/2019
module load hdf5
module load fftw
module load gsl
module load cmake
module load python/3.10.12
module list

source /cosma/home/dp016/dc-stis1/csiborgtools/venv_csiborgtools/bin/activate
export OMP_NUM_THREADS={nthreads}
export OMP_NESTED=true

snapshot_path="/cosma8/data/dp016/dc-stis1/csiborg2_{kind}/chain_{chain_index}/output/snapshot_099_full.hdf5"
output_path="/cosma8/data/dp016/dc-stis1/csiborg2_{kind}/field/chain_{chain_index}.hdf5"
resolution={resolution}
scratch_space="/cosma8/data/dp016/dc-stis1/csiborg2_main/field"
SPH_executable="/cosma8/data/dp016/dc-stis1/cosmotool/bld2/sample/simple3DFilter"
snapshot_kind="gadget4"

python3 field_sph.py --snapshot_path $snapshot_path --output_path $output_path --resolution $resolution --scratch_space $scratch_space --SPH_executable $SPH_executable --snapshot_kind $snapshot_kind
"""
    fname = f"submit_SPH_{kind}_{chain_index}.sh"
    print(f"Writing file:  `{fname}`.")
    with open(fname, "w") as txtfile:
        txtfile.write(txt)
    # Make the file executable
    system(f"chmod +x {fname}")
    return fname


if __name__ == "__main__":
    kind = "main"
    resolution = 256
    nthreads = 4
    chains = [15517]

    for chain_index in chains:
        fname = write_submit(chain_index, kind, resolution, nthreads)
        system(f"sbatch {fname}")
