nthreads=6
memory=64
on_login=${1}
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_prop.py"
kind="radvel"
simname="csiborg2_random"
nsims="-1"
MAS="SPH"
grid=1024


pythoncm="$env $file --nsims $nsims --simname $simname --kind $kind --MAS $MAS --grid $grid"
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
