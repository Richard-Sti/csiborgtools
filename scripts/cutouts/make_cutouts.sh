#!/bin/bash

#SBATCH -p cca -C genoa -N1
#SBATCH --mail-user=rstiskalek@flatironinstitute.org
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Python environment and script execution
python_exec="/mnt/home/rstiskalek/csiborgtools/venv_csiborg/bin/python"
$python_exec make_cutouts.py
