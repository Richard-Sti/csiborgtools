#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./script.sh <path_to_file>"
    exit 1
fi

env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
queue="berg"
nthreads=1
memory=7

file="$1"

cm="addqueue -q $queue -n $nthreads -m $memory $env $file"

echo "Submitting:"
echo $cm
echo
$cm
