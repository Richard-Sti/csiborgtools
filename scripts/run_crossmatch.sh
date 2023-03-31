nthreads=1
memory=32
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_galomatch/bin/python"
file="run_crossmatch.py"

pythoncm="$env $file"
# echo "Submitting:"
# echo $pythoncm
# echo
# $pythoncm

cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
echo "Submitting:"
echo $cm
echo
$cm
