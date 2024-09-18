#!/bin/bash
nthreads=12
memory=7
on_login=${1}
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_sph_gadget.py"

# Guilhem says higher resolution is better
resolution=1024
SPH_executable="/mnt/users/rstiskalek/cosmotool/bld/sample/simple3DFilter"
scratch_space="/mnt/extraspace/rstiskalek/dump/"

# CLONES settings
# snapshot_kind="gadget2"
# snapshot_path="/mnt/extraspace/rstiskalek/CLONES/s8/cf2gvpecc1pt5elmo73_sig6distribsbvoldi_RZA3Derrv2_512_500_ss8_zinit60_000"
# output_path="/mnt/extraspace/rstiskalek/CLONES/s8/cf2gvpecc1pt5elmo73_sig6distribsbvoldi_RZA3Derrv2_512_500_ss8_zinit60_000.hdf5"


# Check if `on_login` is either 0 or 1
# Check if on_login is not empty and is a valid integer (0 or 1)
if [ -z "$on_login" ] || ! [[ "$on_login" =~ ^[0-1]$ ]]; then
    echo "First argument must be either 0 or 1. Received: $on_login"
    exit 1
fi

export OMP_NUM_THREADS={nthreads}
export OMP_NESTED=true

# pythoncm="$env $file --snapshot_path $snapshot_path --output_path $output_path --resolution $resolution --scratch_space $scratch_space --SPH_executable $SPH_executable --snapshot_kind $snapshot_kind"
# if [ $on_login -eq 1 ]; then
#     echo $pythoncm
#     $pythoncm
# else
#     cm="addqueue -s -q $queue -n 1x$nthreads -m $memory $pythoncm"
#     echo "Submitting:"
#     echo $cm
#     echo
#     eval $cm
# fi


# Manticore SWIFT submission loop
snapshot_kind="swift"
for k in {0..40}; do
    snapshot_path="/mnt/extraspace/rstiskalek/MANTICORE/2MPP_N128_DES_V1/resimulations/R512/mcmc_$k/swift_monofonic/snap_0001/snap_0001.hdf5"
    output_path="/mnt/extraspace/rstiskalek/MANTICORE/2MPP_N128_DES_V1/fields/R512/SPH_$k.hdf5"

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

    sleep 0.05
done
