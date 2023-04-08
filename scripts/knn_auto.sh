nthreads=50
memory=4
queue="cmb"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="knn_auto.py"

runs="mass001 mass001_spinlow mass001_spinhigh mass001_spinmedian_perm mass001_spinmedian_cross"

pythoncm="$env $file --runs $runs"

echo $pythoncm
$pythoncm

# cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
# echo "Submitting:"
# echo $cm
# echo
# $cm
