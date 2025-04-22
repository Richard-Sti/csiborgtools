#!/bin/bash

#SBATCH -p cca -C genoa -N1
#SBATCH --mail-user=rstiskalek@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -J get_mah
#SBATCH -o run_get_mah.o%j
#SBATCH -e run_get_mah.e%j
#### #SBATCH --array=40-49%5

# Load modules
module --force purge
module load modules/2.3-20240529
module load slurm
module load python/3.11.7

# Define the current step from the SLURM array task ID
# nstep=$SLURM_ARRAY_TASK_ID
nstep=0

# Define input and output file paths
# input_file="/mnt/home/rstiskalek/ceph/CSiBORG/2MPP_MULTIBIN_N256_DES_V2/resimulations/step_${nstep}/output/treedata/trees.0.hdf5"
# output_file="/mnt/home/rstiskalek/ceph/CSiBORG/postprocessing/MAH/2MPP_MULTIBIN_N256_DES_V2/step_${nstep}.hdf5"

input_file="/mnt/home/rstiskalek/ceph/CSiBORG/RANDOM_N1024_L1000/step_${nstep}/output/treedata/trees.0.hdf5"
output_file="/mnt/home/rstiskalek/ceph/CSiBORG/postprocessing/MAH/RANDOM_N1024_L1000/step_${nstep}.hdf5"



# Threading control
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Python environment and script execution
python_exec="/mnt/home/rstiskalek/csiborgtools/venv_csiborg/bin/python"
$python_exec mah_extract.py --input_file $input_file --output_file $output_file
