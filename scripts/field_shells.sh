nthreads=1
memory=32
on_login=${1}
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_shells.py"

field="overdensity"
simname="borg2"
MAS="SPH"
grid=1024


pythoncm="$env $file --field $field --simname $simname --MAS $MAS --grid $grid"
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
