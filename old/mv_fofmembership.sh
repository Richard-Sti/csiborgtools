nthreads=1
memory=100
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="mv_fofmembership.py"
nsims="5511"

pythoncm="$env $file --nsims $nsims"

# echo $pythoncm
# $pythoncm

cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
echo "Submitting:"
echo $cm
echo
$cm