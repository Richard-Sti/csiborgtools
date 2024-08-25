#!/bin/bash
memory=7
on_login=${1}
queue=${2}
ndevice=1
file="flow_validation.py"
ksmooth=0


if [ "$on_login" != "1" ] && [ "$on_login" != "0" ]
then
    echo "'on_login' (1) must be either 0 or 1."
    exit 1
fi


if [ "$queue" != "redwood" ] && [ "$queue" != "berg" ] && [ "$queue" != "cmb" ] && [ "$queue" != "gpulong" ] && [ "$queue" != "cmbgpu" ]; then
  echo "Invalid queue: $queue (2). Please provide one of 'redwood', 'berg', 'cmb', 'gpulong', 'cmbgpu'."
  exit 1
fi


if [ "$queue" == "gpulong" ]
then
    device="gpu"
    gputype="rtx2080with12gb"
    # gputype="rtx3070with8gb"
    env="/mnt/users/rstiskalek/csiborgtools/venv_gpu_csiborgtools/bin/python"
elif [ "$queue" == "cmbgpu" ]
then
    device="gpu"
    gputype="rtx3090with24gb"
    env="/mnt/users/rstiskalek/csiborgtools/venv_gpu_csiborgtools/bin/python"
else
    device="cpu"
    env="/mnt/users/rstiskalek/csiborgtools/venv_csiborg/bin/python"
fi


# for simname in "Lilow2024" "CF4" "CF4gp" "csiborg1" "csiborg2_main" "csiborg2X"; do
for simname in "Carrick2015"; do
    for catalogue in "CF4_GroupAll"; do
    # for catalogue in "CF4_TFR_i"; do
        # for ksim in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        for ksim in "none"; do
            pythoncm="$env $file --catalogue $catalogue --simname $simname --ksim $ksim --ksmooth $ksmooth --ndevice $ndevice --device $device"

            if [ "$on_login" == "1" ]; then
                echo $pythoncm
                eval $pythoncm
            else
                if [ "$device" == "gpu" ]; then
                    cm="addqueue -q $queue -s -m $memory --gpus 1 --gputype $gputype $pythoncm"
                else
                    cm="addqueue -s -q $queue -n 1 -m $memory $pythoncm"
                fi
                echo "Submitting:"
                echo $cm
                eval $cm
            fi

            echo
            sleep 0.001

        done
    done
done
