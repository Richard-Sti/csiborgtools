#!/bin/bash
nthreads=11
memory=4
queue="cmb"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="match_all.py"

simname="quijote"
min_logmass=13.25
nsim0=0
kind="max"
mult=10
sigma=1
verbose="false"


pythoncm="$env $file --kind $kind --simname $simname --nsim0 $nsim0 --min_logmass $min_logmass --mult $mult --sigma $sigma --verbose $verbose"
# $pythoncm

cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
echo "Submitting:"
echo $cm
echo
$cm
