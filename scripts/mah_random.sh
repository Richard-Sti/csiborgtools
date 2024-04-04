memory=4
on_login=0
nthreads=${1}

queue="berg"
env="/mnt/users/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="mah_random.py"

min_logmass=13.0
simname=${2}


pythoncm="$env $file --simname $simname --min_logmass $min_logmass"
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
