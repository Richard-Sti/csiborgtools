#!/bin/bash
nthreads=1  # Keep this at 1!!
memory=32
on_login=${1}
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_sph_ramses.py"


# nsims=(7444)
nsims=(7444 7468 7492 7516 7540 7564 7588 7612 7636 7660 7684 7708 7732 7756 7780 7804 7828 7852 7876 7900 7924 7948 7972 7996 8020 8044 8068 8092 8116 8140 8164 8188 8212 8236 8260 8284 8308 8332 8356 8380 8404 8428 8452 8476 8500 8524 8548 8572 8596 8620 8644 8668 8692 8716 8740 8764 8788 8812 8836 8860 8884 8908 8932 8956 8980 9004 9028 9052 9076 9100 9124 9148 9172 9196 9220 9244 9268 9292 9316 9340 9364 9388 9412 9436 9460 9484 9508 9532 9556 9580 9604 9628 9652 9676 9700 9724 9748 9772 9796 9820 9844)
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
