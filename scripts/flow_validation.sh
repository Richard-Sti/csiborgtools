memory=4
on_login=${1}
nthreads=${2}
queue="berg"
env="/mnt/users/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="flow_validation.py"

catalogue="A2"
simname="Carrick2015"
ksmooth=0


pythoncm="$env $file --catalogue $catalogue --simname $simname --ksmooth $ksmooth"
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