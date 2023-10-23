nthreads=11
memory=2
queue="berg"
env="/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
file="fit_hmf.py"

simname="quijote_full"
nsims="-1"
verbose=True
lower_lim=12.0
upper_lim=16.0
Rmax=155
from_quijote_backup="true"
bw=0.2

pythoncm="$env $file --simname $simname --nsims $nsims --Rmax $Rmax --lims $lower_lim $upper_lim --bw $bw --from_quijote_backup $from_quijote_backup --verbose $verbose"

$pythoncm

# cm="addqueue -q $queue -n $nthreads -m $memory $pythoncm"
# echo "Submitting:"
# echo $cm
# echo
# $cm
