#!/bin/bash
nthreads=28
memory=7
on_login=${1}
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_sph_gadget.py"

# Guilhem says higher resolution is better
resolution=1024
SPH_executable="/mnt/users/rstiskalek/cosmotool/bld/sample/simple3DFilter"
scratch_space="/mnt/extraspace/rstiskalek/dump/"

snapshot_kind="gadget2"
snapshot_path="/mnt/extraspace/rstiskalek/CLONES/s8/cf2gvpecc1pt5elmo73_sig6distribsbvoldi_RZA3Derrv2_512_500_ss8_zinit60_000"
output_path="/mnt/extraspace/rstiskalek/CLONES/s8/cf2gvpecc1pt5elmo73_sig6distribsbvoldi_RZA3Derrv2_512_500_ss8_zinit60_000.hdf5"


# Check if `on_login` is either 0 or 1
# Check if on_login is not empty and is a valid integer (0 or 1)
if [ -z "$on_login" ] || ! [[ "$on_login" =~ ^[0-1]$ ]]; then
    echo "First argument must be either 0 or 1. Received: $on_login"
    exit 1
fi

export OMP_NUM_THREADS={nthreads}
export OMP_NESTED=true

pythoncm="$env $file --snapshot_path $snapshot_path --output_path $output_path --resolution $resolution --scratch_space $scratch_space --SPH_executable $SPH_executable --snapshot_kind $snapshot_kind"
if [ $on_login -eq 1 ]; then
    echo $pythoncm
    $pythoncm
else
    cm="addqueue -s -q $queue -n 1x$nthreads -m $memory $pythoncm"
    echo "Submitting:"
    echo $cm
    echo
    eval $cm
fi