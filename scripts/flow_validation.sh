memory=4
on_login=1
# nthreads=${1}
nthreads=1


device="gpu"
queue="berg"
env="/mnt/users/rstiskalek/csiborgtools/venv_gpu_csiborgtools/bin/python"
file="flow_validation.py"

#"Pantheon+_zSN"
simname="csiborg1"
catalogue="Pantheon+_groups"
ksmooth=0


pythoncm="$env $file --catalogue $catalogue --simname $simname --ksmooth $ksmooth --ndevice $nthreads --device $device"

if [ $on_login -eq 1 ]; then
    # Add a error if too many devices
    echo $pythoncm
    $pythoncm
else
    cm="addqueue -s -q $queue -n 1x$nthreads -m $memory $pythoncm"
    echo "Submitting:"
    echo $cm
    echo
    eval $cm
fi
