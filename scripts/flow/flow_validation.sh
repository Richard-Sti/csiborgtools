#!/bin/bash
memory=14
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


for simname in "Carrick2015" "csiborg2_main"; do
# for simname in "csiborg2_main" "csiborg2X" ; do
# for simname in "Carrick2015" "Lilow2024" "csiborg2_main" "csiborg2X" "CF4"; do
# for simname in "Carrick2015" "csiborg2X" "csiborg2_main"; do
# for simname in "Carrick2015"; do
    # for catalogue in "LOSS" "Foundation" "2MTF" "SFI_gals" "CF4_TFR_i" "CF4_TFR_w1"; do
    for catalogue in "2MTF" "SFI_gals" "CF4_TFR_i"; do
    # for catalogue in "2MTF" "SFI" "CF4_TFR_not2MTForSFI_i"; do
    # for catalogue in "2MTF" "SFI_gals" "CF4_TFR_i"; do
    # for catalogue in "CF4_TFR_w1"; do
    # for catalogue in "CF4_GroupAll"; do
        for ksim in "none"; do
        for ksmooth in 1 2 3 4; do
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

done