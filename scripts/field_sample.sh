nthreads=11
memory=32
on_login=${1}
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_sample.py"


nsims="-1"
simname="csiborg1"
survey="SDSS"
smooth_scales="0 2 4 8 16"
kind="density"
MAS="SPH"
grid=1024
scatter=0


pythoncm="$env $file --nsims $nsims --simname $simname --survey $survey --smooth_scales $smooth_scales --kind $kind --MAS $MAS --grid $grid --scatter $scatter"
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
