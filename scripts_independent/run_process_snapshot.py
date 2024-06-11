# Copyright (C) 2023 Richard Stiskalek
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
from os import system

import numpy as np

if __name__ == "__main__":
    # chains = [15717, 15817, 15917, 16017, 16117, 16217, 16317, 16417, 16517,
    #           16617, 16717, 16817, 16917, 17017, 17117, 17217, 17317, 17417]
    # simname = "csiborg2_main"
    # mode = 1

    # chains = [1] + [25 + n * 25 for n in range(19)]
    # simname = "csiborg2_varysmall"
    # mode = 1

    # chains = [1] + [25 + n * 25 for n in range(19)]
    # simname = "csiborg2_random"
    # mode = 1

    # chains = [7444 + n * 24 for n in range(101)]
    # simname = "csiborg1"
    # mode = 2
    chains = np.hstack([np.arange(0, 10), np.arange(10000, 10030 + 1)])[1:]
    simname = "quijote"
    mode = 0

    env = "/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
    memory = 32

    for chain in chains:
        out = f"output_{simname}_{chain}_%j.out"
        cmd = f"addqueue -q berg -o {out} -n 1x1 -m {memory} {env} process_snapshot.py --nsim {chain} --simname {simname} --mode {mode}"  # noqa
        print(cmd)
        system(cmd)
        print()
