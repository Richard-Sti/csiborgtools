#!/bin/bash
nthreads=41
memory=12
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="match_overlap_all.py"

simname=${1}
min_logmass=12.25
sigma=1
kind="overlap"
mult=10         # Only for Max's method
nsim0=0         # Only for Max's method
verbose="false"


pythoncm="$env $file --kind $kind --simname $simname --nsim0 $nsim0 --min_logmass $min_logmass --mult $mult --sigma $sigma --verbose $verbose"
# $pythoncm

cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
echo "Submitting:"
echo $cm
echo
$cm
