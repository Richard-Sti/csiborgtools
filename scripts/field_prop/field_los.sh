nthreads=1
memory=64
on_login=${1}
queue="cmb"
env="/mnt/users/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_los.py"

nsims="-1"
# These are only for CB
MAS="SPH"
grid=1024


if [ "$on_login" != "1" ] && [ "$on_login" != "0" ]
then
    echo "'on_login' (1) must be either 0 or 1."
    exit 1
fi


# for simname in "csiborg1" "csiborg2_main" "csiborg2X" "Lilow2024" "Carrick2015" "CF4" "manticore_2MPP_N128_DES_V1"; do
for simname in "manticore_2MPP_MULTIBIN_N128_DES_V1"; do
# for simname in "csiborg1" "csiborg2_main" "Lilow2024" "Carrick2015" "CF4" "CLONES"; do
    for catalogue in "LOSS" "Foundation" "2MTF" "SFI_gals" "CF4_TFR"; do
    # for catalogue in "LOSS"; do
    # for catalogue in "SDSS-FP"; do
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

        sleep 0.05
    done
done
