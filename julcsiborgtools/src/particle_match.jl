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
"""
    halo_parts(clumpid::Int, partids::Vector{<:Int}, clumpids::Vector{<:Int})

Return particle IDs belonging to a given clump.

# Arguments
- `clumpid::Integer`: the ID of the clump.
- `partids::Vector{<:Integer}`: vector of shape `(n_particles,)` with the particle IDs.
- `clumpids::Vector{<:Integer}`: vector of shape `(n_particles, )` with the particles' clump IDs.
"""
function halo_parts(clumpid::Integer, partids::Vector{<:Integer}, clumpids::Vector{<:Integer})
    return partids[clumpids.==clumpid]
end
