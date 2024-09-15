nthreads=1
memory=48
on_login=${1}
queue="berg"
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


# for simname in "csiborg1" "csiborg2_main" "csiborg2X" "Lilow2024" "Carrick2015" "CF4"; do
for simname in "Carrick2015"; do
    # for catalogue in "2MTF" "SFI_gals" "CF4_TFR"; do
    # for catalogue in "Foundation" "2MTF" "SFI_gals" "CF4_TFR"; do
    for catalogue in "CF4_TFR"; do
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
