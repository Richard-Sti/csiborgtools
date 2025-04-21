#!/bin/bash

#SBATCH -p cca -C genoa -N1
#SBATCH --mail-user=rstiskalek@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -J cutout
#SBATCH -o run_cutout.o%j
#SBATCH -e run_cutout.e%j

# Load modules
module --force purge
module load modules/2.3-20240529
module load slurm
module load python/3.11.7


# Threading control
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_MAX_ACTIVE_LEVELS=4

echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"



snapshot_path="/mnt/home/rstiskalek/ceph/CSiBORG/2MPP_MULTIBIN_N256_DES_V2/resimulations/step_0/output/snapdir_130/snapshot_130.hdf5"
output_path="/mnt/home/rstiskalek/ceph/scratch/test_field.hdf5"
resolution=256
scratch_space="/mnt/home/rstiskalek/ceph/scratch/real_scratch"
SPH_executable="./simple3DFilter"
snapshot_kind="gadget4"

# if __name__ == "__main__":
#     parser = ArgumentParser(description="Generate SPH density and velocity field.")  # noqa
#     parser.add_argument("--snapshot_path", type=str, required=True,
#                         help="Path to the simulation snapshot.")
#     parser.add_argument("--output_path", type=str, required=True,
#                         help="Path to the output HDF5 file.")
#     parser.add_argument("--resolution", type=int, required=True,
#                         help="Resolution of the density and velocity field.")
#     parser.add_argument("--scratch_space", type=str, required=True,
#                         help="Path to a folder where temporary files can be stored.")  # noqa
#     parser.add_argument("--SPH_executable", type=str, required=True,
#                         help="Path to the `simple3DFilter` executable.")
#     parser.add_argument("--snapshot_kind", type=str, required=True,
#                         choices=["gadget4", "gadget2", "ramses", "swift"],
#                         help="Kind of the simulation snapshot.")
#     args = parser.parse_args()

python_exec="/mnt/home/rstiskalek/csiborgtools/venv_csiborg/bin/python"
$python_exec field_sph_gadget.py \
    --snapshot_path $snapshot_path \
    --output_path $output_path \
    --resolution $resolution \
    --scratch_space $scratch_space \
    --SPH_executable $SPH_executable \
    --snapshot_kind $snapshot_kind