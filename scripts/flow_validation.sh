#!/bin/bash
memory=8
on_login=${1}
ndevice=1

device="gpu"
queue="gpulong"
gputype="rtx2080with12gb"
env="/mnt/users/rstiskalek/csiborgtools/venv_gpu_csiborgtools/bin/python"
file="flow_validation.py"
ksmooth=0


if [ "$on_login" != "1" ] && [ "$on_login" != "0" ]; then
  echo "Invalid input: 'on_login' (1). Please provide 1 or 0."
  exit 1
fi

# Submit a job for each combination of simname, catalogue, ksim
for simname in "csiborg2_main"; do
    for catalogue in "Pantheon+"; do
        # for ksim in 0 1 2; do
        # for ksim in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        for ksim in "none"; do
            pythoncm="$env $file --catalogue $catalogue --simname $simname --ksim $ksim --ksmooth $ksmooth --ndevice $ndevice --device $device"

            if [ $on_login -eq 1 ]; then
                echo $pythoncm
                $pythoncm
            else
                cm="addqueue -q $queue -s -m $memory --gpus 1 --gputype $gputype $pythoncm"
                echo "Submitting:"
                echo $cm
                eval $cm
            fi
            echo
            sleep 0.05
        done
    done
done
