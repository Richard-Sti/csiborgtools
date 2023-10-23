nthreads=4
memory=4
queue="cmb"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="cluster_knn_auto.py"
Rmax=219.8581560283688
verbose="true"



simname="quijote"
nsims="0 1 2"
# simname="csiborg"
# nsims="7444 7900 9052"

run="mass003"

pythoncm="$env $file --run $run --simname $simname --nsims $nsims --Rmax $Rmax --verbose $verbose"

echo $pythoncm
$pythoncm

# cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
# echo "Submitting:"
# echo $cm
# echo
# $cm
