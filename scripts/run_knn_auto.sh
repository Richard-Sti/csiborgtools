nthreads=101
memory=4
queue="cmb"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="run_knn_auto.py"



pythoncm="$env $file --runs 001 002 003 004 005 006 007 008 009"

# echo $pythoncm
# $pythoncm

cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
echo "Submitting:"
echo $cm
echo
$cm
