nthreads=1
on_login=${1}
memory=18
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="void_stats_grid.py"


if [[ "$on_login" != "0" && "$on_login" != "1" ]]
then
    echo "Error: on_login (1) must be either 0 or 1."
    exit 1
fi


for profile in "exp" "gauss" "mb"; do
# for profile in "gauss"; do
    pythoncm="$env $file $profile"
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
done