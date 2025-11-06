# CSiBORGTools

`csiborgtools` is a Python library for analyzing "digital twin" cosmological simulations. These are N-body simulations whose initial conditions are inferred from observational data of the local Universe using the BORG (Bayesian Origin Reconstruction from Galaxies) algorithm. This allows for a direct comparison between the simulated universe and our own.

The features of this library include:
*   An interface to read particle snapshots, halo catalogues, and merger trees (if available) from a series of generations of digital twins, as well as standard cosmological simulations for comparison (e.g. Quijote, TNG300).
*   Support for computing large-scale structure properties, including density, velocity, and tidal tensor fields from the particle data using either the `pylians` library or an SPH kernel.
*   Support for calculating properties of dark matter halos, though this is currently limited to their center (via the shrinking sphere approach) and overdensity mass. This also assumes that bound particles have already been identified with a halo finder.
*   A framework for matching observed galaxy clusters to their simulated counterparts in these "digital twin" simulations, enabling robust comparisons between theory and observation.
*   A tool to calculate the overlap of dark matter halos between different simulations. This is useful for identifying whether a halo is a persistent structure across posterior samples from the BORG algorithm.
*   Support for creating cutouts from particle snapshots, allowing for detailed analysis of specific regions of interest.

### Halo Overlap Calculation

The halo overlap calculation is a method to quantify the similarity between dark matter halos from different simulations that are samples from the same BORG posterior. This allows identifying persistent structures across the posterior.

The process involves:
1.  **Density Field Generation**: For each halo, a 3D density field is created by assigning the mass of its particles to a grid.
2.  **Cell-by-Cell Comparison**: The density fields of the two halos are compared cell by cell.
3.  **Intersection Calculation**: An "intersection" value is computed for each cell based on the densities of the two halos in that cell.
4.  **Final Overlap Value**: The final overlap score is calculated based on the sum of intersections and the total mass of the two halos.

An overlap score near 1 indicates that the two halos are essentially the same object, while a score near 0 indicates they are distinct.



## Installation
```
git clone git@github.com:Richard-Sti/csiborgtools.git

# Go to the cloned directory
cd csiborgtools

# Create a virtual environment
python -m venv venv_csiborgtools
source venv_csiborgtools/bin/activate
python -m pip install --upgrade pip && python -m pip install --upgrade setuptools

# Finally install the cloned package
python -m pip install -e .
```

## License and Citation

If you use or find useful any of the code in this repository, please cite the following paper:

- *Evaluating the reconstruction of individual haloes in constrained cosmological simulations* (Stiskalek+2023) (https://arxiv.org/abs/2310.20672)

```
Copyright (C) 2024 Richard Stiskalek
This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
```

## Contributors
- Richard Stiskalek (University of Oxford)
- Harry Desmond (University of Portsmouth)
