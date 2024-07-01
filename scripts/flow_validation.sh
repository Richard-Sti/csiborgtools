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


: << 'COMMENT'
Finished running:
    - Carrick2015 with all catalogues

    - Lilow2024 wbeta with all catalogues
    - Lilow2024 wobeta with all catalogues

    - CF4 wbeta with all catalogues
    - CF4 wobeta with all catalogues
    - CF4gp wbeta with all catalogues
    - CF4gp wobeta with all catalogues

    - csiborg1 wbeta with all catalogues
    - csiborg1 wobeta with all catalogues
    - csiborg2_main wbeta with all catalogues
    - csiborg2_main wobeta with all catalogues
    - csiborg2X wbeta with all catalogues
    - csiborg2X wobeta with all catalogues

    - csiborg2_main/csiborg2X 2MTF & Pantheon+ boxes individually.

Remaining to do:
    - Lilow, CF4, and csiborgs with beta fixed.
COMMENT

# Submit a job for each combination of simname, catalogue, ksim
# for simname in "Lilow2024" "CF4" "CF4gp" "csiborg2_main" "csiborg2X"; do
for simname in "Lilow2024"; do
# for simname in "csiborg1" "csiborg2_main" "csiborg2X"; do
    # for catalogue in "LOSS" "Foundation" "2MTF" "Pantheon+" "Pantheon+_groups" "Pantheon+_zSN" "SFI_gals"; do
    for catalogue in "LOSS"; do
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
            sleep 0.001
        done
    done
done
