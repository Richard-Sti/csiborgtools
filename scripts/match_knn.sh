#!/bin/bash
nthreads=50
memory=7
queue="cmb"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="match_finsnap.py"
verbose="true"
nsims="-1"
onlogin=false

# for run in "mass001" "mass003" "mass005" "mass007" "mass009"
for run in "mass002" "mass004" "mass006" "mass008"
do
for simname in "csiborg"
do
pythoncm="$env $file --simname $simname --run $run --nsims $nsims --verbose $verbose"

if $onlogin
then
    $pythoncm
else
    cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
    echo "Submitting:"
    echo $cm
    echo
    $cm
fi
done
done
