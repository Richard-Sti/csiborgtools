#!/bin/bash
nthreads=1
memory=16
on_login=${1}
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="quijote_pecvel_covmat.py"


if [ "$on_login" != "1" ] && [ "$on_login" != "0" ]; then
  echo "Invalid input: 'on_login' (1). Please provide 1 or 0."
  exit 1
fi

pythoncm="$env $file"
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
