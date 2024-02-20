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
calculate the SPH density & velocity field for GADGET.
"""
from os import system
from os.path import join


def write_submit(chain_index, kind, resolution, nthreads, snapshot_kind):
    if kind not in ["main", "random", "varysmall"]:
        raise RuntimeError(f"Unknown kind `{kind}`.")

    basepath = "/cosma8/data/dp016/dc-stis1/"
    if snapshot_kind == "gadget4":
        snapshot_path = join(basepath, f"csiborg2_{kind}/chain_{chain_index}",
                             "output/snapshot_099_full.hdf5")
        output_path = join(
            basepath,
            f"csiborg2_{kind}/field/chain_{chain_index}_{resolution}.hdf5")
    else:
        chain_index = str(chain_index).zfill(5)
        snapshot_path = join(basepath, "csiborg1_sph",
                             f"ramses_{chain_index}.hdf5")
        output_path = join(basepath, "csiborg1_sph",
                           f"sph_ramses_{chain_index}_{resolution}.hdf5")

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
#SBATCH -t 16:00:00
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

snapshot_path={snapshot_path}
output_path={output_path}
resolution={resolution}
scratch_space="/snap8/scratch/dp016/dc-stis1/"
SPH_executable="/cosma8/data/dp016/dc-stis1/cosmotool/bld2/sample/simple3DFilter"
snapshot_kind={snapshot_kind}

python3 field_sph_gadget.py --snapshot_path $snapshot_path --output_path $output_path --resolution $resolution --scratch_space $scratch_space --SPH_executable $SPH_executable --snapshot_kind $snapshot_kind
"""
    fname = f"submit_SPH_{kind}_{chain_index}.sh"
    print(f"Writing file:  `{fname}`.")
    with open(fname, "w") as txtfile:
        txtfile.write(txt)
    # Make the file executable
    system(f"chmod +x {fname}")
    return fname


if __name__ == "__main__":
    snapshot_kind = "gadget4"
    # kind = "main"
    # chains = [15617, 15717, 15817, 15917, 16017, 16117, 16217, 16317, 16417, 16517, 16617, 16717, 16817, 16917, 17017, 17117, 17217, 17317, 17417]

    # kind = "varysmall"
    # chains = ["16417_001", "16417_025", "16417_050", "16417_075", "16417_100", "16417_125", "16417_150", "16417_175", "16417_200", "16417_225", "16417_250", "16417_275", "16417_300", "16417_325", "16417_350", "16417_375", "16417_400", "16417_425", "16417_450", "16417_475"]

    # kind = "random"
    # chains = [1, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475]

    # CSiBORG1 RAMSES
    snapshot_kind = "ramses"
    kind = "main"
    chains = [7444]

    resolution = 1024
    nthreads = 32

    for chain_index in chains:
        fname = write_submit(chain_index, kind, resolution, nthreads,
                             snapshot_kind)
        system(f"sbatch {fname}")
