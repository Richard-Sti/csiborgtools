nthreads=140
memory=7
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="run_knn.py"

rmin=0.01
rmax=100
nneighbours=16
nsamples=10000000
neval=10000

pythoncm="$env $file --rmin $rmin --rmax $rmax --nneighbours $nneighbours --nsamples $nsamples --neval $neval"

# echo $pythoncm
# $pythoncm

cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
echo "Submitting:"
echo $cm
echo
$cm
