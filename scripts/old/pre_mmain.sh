nthreads=102
memory=5
queue="cmb"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="pre_mmain.py"

# pythoncm="$env $file"
# $pythoncm


cm="addqueue -q $queue -n $nthreads -m $memory $env $file"
echo "Submitting:"
echo $cm
$cm
