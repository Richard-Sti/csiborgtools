nthreads=10
memory=32
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="fieldprop.py"
# grid=1024
# halfwidth=0.1

cm="addqueue -q $queue -n $nthreads -m $memory $env $file"

echo "Submitting:"
echo $cm
echo
$cm
