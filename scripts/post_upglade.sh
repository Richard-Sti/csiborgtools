nthreads=${1}
on_login=${2}
memory=7
queue="redwood"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="post_upglade.py"


if [[ "$on_login" != "0" && "$on_login" != "1" ]]
then
    echo "Error: on_login must be either 0 or 1."
    exit 1
fi

if ! [[ "$nthreads" =~ ^[0-9]+$ ]] || [ "$nthreads" -le 0 ]; then
    echo "Error: nthreads must be an integer larger than 0."
    exit 1
fi


pythoncm="$env $file"
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
