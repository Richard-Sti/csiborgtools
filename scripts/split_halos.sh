nthreads=1
memory=30
queue="cmb"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="split_halos.py"

cm="addqueue -q $queue -n $nthreads -m $memory $env $file"

echo "Submitting:"
echo $cm
echo
$cm
