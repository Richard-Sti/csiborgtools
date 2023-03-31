#!/bin/bash
# nthreads=1
memory=16
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="run_singlematch.py"

nmult=1.
sigma=1.

sims=(7468 7588 8020 8452 8836)
nsims=${#sims[@]}

for i in $(seq 0 $((nsims-1))); do
for j in $(seq 0 $((nsims-1))); do
if [ $i -eq $j ]; then
    continue
elif [ $i -gt $j ]; then
    continue
else
    :
fi

nsim0=${sims[$i]}
nsimx=${sims[$j]}

pythoncm="$env $file --nsim0 $nsim0 --nsimx $nsimx --nmult $nmult --sigma $sigma"

cm="addqueue -q $queue -n 1x1 -m $memory $pythoncm"
echo "Submitting:"
echo $cm
echo
$cm
sleep 0.05

done; done
