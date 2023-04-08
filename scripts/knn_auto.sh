nthreads=102
memory=4
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="knn_auto.py"

#runs="mass001 mass001_spinlow mass001_spinhigh mass001_spinmedian_perm mass001_spinmedian_cross"
# runs="mass001_spinmedian_cross_perm"

runs="mass001_spinlow_cross_perm mass001_spinhigh_cross_perm"

pythoncm="$env $file --runs $runs"

# echo $pythoncm
# $pythoncm

cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
echo "Submitting:"
echo $cm
echo
$cm
