nthreads=100
memory=3
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="run_fit_halos.py"

cm="addqueue -q $queue -n $nthreads -m $memory $env $file"

echo "Submitting:"
echo $cm
echo
$cm
