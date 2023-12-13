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

if __name__ == "__main__":
    # Quijote chains
    chains = [1]
    simname = "quijote"
    mode = 2

    env = "/mnt/zfsusers/rstiskalek/csiborgtools/venv_csiborg/bin/python"
    memory = 64

    for chain in chains:
        out = f"output_{simname}_{chain}_%j.out"
        cmd = f"addqueue -q berg -o {out} -n 1x1 -m {memory} {env} process_snapshot.py --nsim {chain} --simname {simname} --mode {mode}"  # noqa
        print(cmd)
        system(cmd)
        print()
