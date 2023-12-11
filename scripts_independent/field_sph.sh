#!/bin/sh

#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=7000
#SBATCH -J SPH
#SBATCH -o output_%J.out
#SBATCH -e error_%J.err
#SBATCH -p cosma8-serial
#SBATCH -A dp016
#SBATCH -t 01:00:00
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
export OMP_NUM_THREADS=1

# ADD CHAINS HERE

snapshot_path="/cosma8/data/dp016/dc-stis1/csiborg2_main/chain_15517/output/snapshot_099_full.hdf5"
output_path="/cosma8/data/dp016/dc-stis1/csiborg2_main/field/chain_15517.hdf5"
resolution=64
scratch_space="/cosma8/data/dp016/dc-stis1/csiborg2_main/field"
SPH_executable="/cosma8/data/dp016/dc-stis1/cosmotool/bld/sample/simple3DFilter"
snapshot_kind="gadget4"


python3 field_sph.py --snapshot_path $snapshot_path --output_path $output_path --resolution $resolution --scratch_space $scratch_space --SPH_executable $SPH_executable --snapshot_kind $snapshot_kind
