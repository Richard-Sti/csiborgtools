nthreads=15  # There isn't too much benefit going to too many CPUs...
memory=32
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="run_initmatch.py"

dump_clumps="false"

cm="addqueue -q $queue -n $nthreads -m $memory $env $file --dump_clumps $dump_clumps"

echo "Submitting:"
echo $cm
echo
$cm
