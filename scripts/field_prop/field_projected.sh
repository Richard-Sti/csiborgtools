nthreads=1
memory=64
on_login=${1}
simname=${2}
queue="berg"
env="/mnt/users/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_projected.py"


if [ "$on_login" != "1" ] && [ "$on_login" != "0" ]
then
    echo "'on_login' (1) must be either 0 or 1."
    exit 1
fi

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
