nthreads=1
memory=40
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="pre_dumppart.py"
simname="csiborg"
nsims="5511"

pythoncm="$env $file --nsims $nsims --simname $simname"

# echo $pythoncm
# $pythoncm

cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
echo "Submitting:"
echo $cm
echo
$cm