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
Script to test running the CSiBORG realisations matcher.
"""
import numpy
from argparse import ArgumentParser
from distutils.util import strtobool
from datetime import datetime
from os.path import join
try:
    import csiborgtools
except ModuleNotFoundError:
    import sys
    sys.path.append("../")
    import csiborgtools
import utils

# Argument parser
parser = ArgumentParser()
parser.add_argument("--nsim0", type=int)
parser.add_argument("--nsimx", type=int)
parser.add_argument("--nmult", type=float)
parser.add_argument("--overlap", type=lambda x: bool(strtobool(x)))
args = parser.parse_args()

# File paths
fout = join(
    utils.dumpdir, "overlap", "cross_{}_{}.npz".format(args.nsim0, args.nsimx))

print("{}: loading catalogues.".format(datetime.now()), flush=True)
cat0 = csiborgtools.read.HaloCatalogue(args.nsim0)
catx = csiborgtools.read.HaloCatalogue(args.nsimx)

matcher = csiborgtools.match.RealisationsMatcher()
print("{}: crossing the simulations.".format(datetime.now()), flush=True)
ref_indxs, cross_indxs, match_indxs, cross = matcher.cross(
    cat0, catx, overlap=args.overlap)

# Dump the result
print("Saving results to `{}`.".format(fout), flush=True)
with open(fout, "wb") as f:
    numpy.savez(fout, ref_indxs=ref_indxs, cross_indxs=cross_indxs,
                match_indxs=match_indxs, cross=cross)

print("All finished.", flush=True)
