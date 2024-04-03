#!/bin/bash
nthreads=1
memory=7
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
verbose="true"
file="match_overlap_single.py"

simname="csiborg2_main"
kind="overlap"
min_logmass=12
mult=5  # Only for Max's method
sigma=1

# sims=(7444 7468)
sims=(16417 16517)
# sims=(0 1)
# sims=(7468 7588)
nsims=${#sims[@]}

for i in $(seq 0 $((nsims-1)))
do
    for j in $(seq 0 $((nsims-1)))
    do
        if [ $i -eq $j ]
        then
            continue
        elif [ $i -gt $j ]
        then
            continue
        else
            :
        fi

        nsim0=${sims[$i]}
        nsimx=${sims[$j]}

        pythoncm="$env $file --kind $kind --nsim0 $nsim0 --nsimx $nsimx  --simname $simname --min_logmass $min_logmass --sigma $sigma --mult $mult --verbose $verbose"

        $pythoncm

        # cm="addqueue -q $queue -n 1x1 -m $memory $pythoncm"
        # echo "Submitting:"
        # echo $cm
        # echo
        # $cm
        # sleep 0.05

    done
done
