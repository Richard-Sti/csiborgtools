nthreads=1
memory=7
on_login=${1}
queue="berg"
env="/mnt/users/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="field_los_indranil_void.py"


if [ "$on_login" != "1" ] && [ "$on_login" != "0" ]
then
    echo "'on_login' (1) must be either 0 or 1."
    exit 1
fi


for kind in "exp" "gauss" "mb"; do
    for catalogue in "2MTF" "SFI_gals" "CF4_TFR"; do
        pythoncm="$env $file --kind $kind --catalogue $catalogue"

        if [ $on_login -eq 1 ]; then
            echo $pythoncm
            eval $pythoncm
        else
            cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
            echo "Submitting:"
            echo $cm
            echo
            eval $cm
        fi
    done
done

