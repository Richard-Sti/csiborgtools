# Copyright (C) 2022 Richard Stiskalek
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

using Pkg: activate # , build
activate("../jucsiborgtools")
using Revise

using jucsiborgtools
using NPZ: npzread
using PyCall: pyimport
csiborgtools = pyimport("csiborgtools")

verbose = true
paths = csiborgtools.read.CSiBORGPaths()
nsims = paths.ic_ids[:1]

for nsim in nsims
    nsnap_min = convert(Int64, paths.get_minimum_snapshot(nsim))
    nsnap_max = convert(Int64, paths.get_maximum_snapshot(nsim))

    # Get the maximum snapshot properties
    verbose ? println("Loading snapshot $nsnap_max from simulation $nsim") : nothing
    pids, ppos, pmass, clumpids = csiborgtools.read.get_positions(nsim, nsnap_max, get_clumpid=true, verbose=false)

    println("Sizes are: ")
    println(size(pids))
    println(size(ppos))
    println(size(pmass))



#    # Get the minimum snapshot properties
#    verbose ? println("Loading snapshot $nsnap_min from simulation $nsim") : nothing
#    pids, ppos, pmass = csiborgtools.read.get_positions(nsim, nsnap_max, get_clumpid=false, verbose=false)

    JuliaCSiBORGTools.halo_parts(0, partids, clumpids)

end
