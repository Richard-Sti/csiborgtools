nthreads=1
memory=64
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="sort_halomaker.py"

method="FOF"
nsim="7444"

pythoncm="$env $file --method $method --nsim $nsim"

# echo $pythoncm
# $pythoncm

cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
echo "Submitting:"
echo $cm
echo
$cm