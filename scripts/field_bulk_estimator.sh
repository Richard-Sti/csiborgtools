nthreads=20
memory=24
on_login=0
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_bulk_estimator.py"

simname="quijote"


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
