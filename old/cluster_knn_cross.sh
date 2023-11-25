nthreads=151
memory=4
queue="cmb"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="knn_cross.py"

runs="mass001"

pythoncm="$env $file --runs $runs"

echo $pythoncm
$pythoncm

# cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
# echo "Submitting:"
# echo $cm
# echo
# $cm
