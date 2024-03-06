nthreads=6
memory=64
on_login=${1}
queue="berg"
env="/mnt/users/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_los.py"

# catalogue="Foundation"
catalogue="csiborg1_9844"
nsims="-1"
simname="csiborg1"
MAS="SPH"
grid=1024


pythoncm="$env $file --catalogue $catalogue --nsims $nsims --simname $simname --MAS $MAS --grid $grid"
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
