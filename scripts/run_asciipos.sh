nthreads=1
memory=75
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="run_asciipos.py"
mode="dump"

cm="addqueue -q $queue -n $nthreads -m $memory $env $file --mode $mode"

echo "Submitting:"
echo $cm
echo
$cm
