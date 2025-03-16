nthreads=24
on_login=${1}
memory=3
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="void_stats_for_samples.py"


if [[ "$on_login" != "0" && "$on_login" != "1" ]]
then
    echo "Error: on_login (1) must be either 0 or 1."
    exit 1
fi


for profile in "exp"; do
    # fname="/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity/samples_IndranilVoid_${profile}_CF4_TFR_i_CF4_TFR_notSDSS_w1_bayes_zcmb_max_0.05.hdf5"
    # fname="/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity/samples_IndranilVoidSizeVar_${profile}_CF4_TFR_i_CF4_TFR_notSDSS_w1_bayes_zcmb_max_0.05_which_void_size_run_zoom.hdf5"
    # fname="/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity/void/samples_IndranilVoid_exp_CF4_TFR_w1_mike_zcmb_max_0.065_remove_CF4_outliers.hdf5"
    fname="/mnt/extraspace/rstiskalek/csiborg_postprocessing/peculiar_velocity/void/samples_IndranilVoidSizeVar_exp_CF4_TFR_i_CF4_TFR_notSDSS_w1_mike_zcmb_max_0.065_which_void_size_run_zoom_remove_CF4_outliers.hdf5"

    pythoncm="$env $file $fname --njobs $nthreads"
    if [ $on_login -eq 1 ]; then
        echo $pythoncm
        $pythoncm
    else
        cm="addqueue -s -q $queue -n 1x$nthreads -m $memory $pythoncm"
        echo "Submitting:"
        echo $cm
        echo
        eval $cm
    fi
done
