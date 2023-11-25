nthreads=20
memory=40
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="cluster_crosspk.py"
grid=1024
halfwidth=0.13

cm="addqueue -q $queue -n $nthreads -m $memory $env $file --grid $grid --halfwidth $halfwidth"

echo "Submitting:"
echo $cm
echo
$cm
