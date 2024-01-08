#!/bin/bash
nthreads=1
memory=32
on_login=${1}
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_sph_ramses.py"


nsims=(7444)
mode="prepare"
output_folder="/mnt/extraspace/rstiskalek/dump/"
resolution=1024
scratch_space="/mnt/extraspace/rstiskalek/dump/"
SPH_executable="NaN"
snapshot_kind="ramses"


for nsim in "${nsims[@]}"; do
    pythoncm="$env $file --nsim $nsim --mode $mode --output_folder $output_folder --resolution $resolution --scratch_space $scratch_space --SPH_executable $SPH_executable --snapshot_kind $snapshot_kind"
    if [ $on_login -eq 1 ]; then
        echo $pythoncm
        $pythoncm
    else
        cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
        echo "Submitting:"
        echo $cm
        echo
        eval $cm
    fi
done
