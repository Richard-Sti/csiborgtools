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

dpi = 600
fout = "../plots/"
mplstyle = ["science"]


def latex_float(*floats, n=2):
    """
    Convert a float or a list of floats to a LaTeX string(s). Taken from [1].

    Parameters
    ----------
    floats : float or list of floats
        The float(s) to be converted.
    n : int, optional
        The number of significant figures to be used in the LaTeX string.

    Returns
    -------
    latex_floats : str or list of str
        The LaTeX string(s) representing the float(s).

    References
    ----------
    [1] https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python  # noqa
    """
    latex_floats = [None] * len(floats)
    for i, f in enumerate(floats):
        float_str = "{0:.{1}g}".format(f, n)
        if "e" in float_str:
            base, exponent = float_str.split("e")
            latex_floats[i] = r"{0} \times 10^{{{1}}}".format(base,
                                                              int(exponent))
        else:
            latex_floats[i] = float_str

    if len(floats) == 1:
        return latex_floats[0]
    return latex_floats
