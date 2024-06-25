nthreads=1
memory=12
on_login=1
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_bulk.py"

simname=${1}


pythoncm="$env $file --simname $simname"
if [ $on_login -eq 1 ]; then
    echo $pythoncm
    $pythoncm
else
    cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
    echo "Submitting:"
    echo $cm
    echo
    eval $cm
fi
