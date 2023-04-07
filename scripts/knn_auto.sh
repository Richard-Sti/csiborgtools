nthreads=50
memory=4
queue="cmb"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="knn_auto.py"


runs="mass001 mass001_perm mass002 mass002_perm mass003 mass003_perm mass001_spinlow mass001_spinlow_perm mass001_spinhigh mass001_spinhigh_perm mass002_spinlow mass002_spinlow_perm mass002_spinhigh mass002_spinhigh_perm mass003_spinlow mass003_spinlow_perm mass_003_spinhigh mass_003_spinhigh_perm"


pythoncm="$env $file --runs $runs"

# echo $pythoncm
# $pythoncm

cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
echo "Submitting:"
echo $cm
echo
$cm
