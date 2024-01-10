# CSiBORG Tools

Tools for analysing the suite of Constrained Simulations in BORG (CSiBORG) simulations. The interface is designed to work with the following suites of simulations: *CSiBORG1* (dark matter-only RAMSES), *CSiBORG2* (dark matter-only Gadget4), *Quijote* (dark-matter only Gadget2), however with little effort it can support other simulations as well.

## Ongoing projects

### Data to calculate
- [ ] Process all CSiBORG1 catalogues (running).
- [ ] Calculate halo properties for CSiBORG1
- [ ] Calculate initial properties for CSiBORG1
- [ ] Calculate halo properties for CSiBORG2
- [ ] Calculate initial properties for CSiBORG2
- [ ] Calculate halo properties for Quijote
- [ ] Calculate initial properties for Quijote

### Consistent halo reconstruction
- [ ] Make a sketch of the overlap definition and add it to the paper.
- [ ] Re-calculate the overlaps for CSiBORG1, Quijote and CSiBORG2
- [x] Fix the script to calculate the initial lagrangian positions etc.

### Enviromental dependence of galaxy properties
- [x] Prepare the CSiBORG one particle files for SPH.
- [ ] Transfer, calculate the SPH density field for CSiBORG1 and transfer back.
- [x] Check that the velocity-field flipping of x and z coordinates is correct.
- [x] Evaluate and share the density field for SDSS and SDSSxALFALFA for both CSiBORG2 and random fields.
- [x] Check and verify the density field of galaxy colours (cannot do this now! Glamdring is super slow.)

#### Calculated data
##### SPH-density & velocity field
- *CSiBORG2_main*, *CSiBORG2_random*, *CSiBORG2_varysmall*
- Evaluated for SDSS and SDSSxALFALFA in: *CSiBORG2_main*, *CSiBORG2_random*

#### Radial velocity field
- *CSiBORG2_main


### Mass-assembly of massive clusters
- [ ] Make a list of nearby most-massive clusters.
- [ ] Write code to identify a counterpart of such clusters.
- [ ] Write code to make a plot of mass-assembly of all clusters within a certain mass range from the random simulations.
- [ ] Write code to compare mass-assembly of a specific cluster with respect to random ones.


### Effect of small-scale noise
- [ ] Study how the small-scale noise variation affects the overlap measure, halo concentration and spin.

### Gravitational-wave and large-scale structure
- [ ] Make the velocity field data available.
- [ ] Validate the velocity field results.
- [ ] Estimate the amplitude of the velocity field in radial shells around the observer, estimate analogous results for random simulations, and see if they agree within cosmic variance.
- [ ] Calculate power spectra of velocities and maybe velocity dispersion.

### CSiBORG meets X-ray
- [ ] Make available one example snapshot from the simulation. Mention the issue with x- and z-coordinates.

### CSiBORG advertising
- [ ] Decide on the webpage design and what to store there.
- [ ] Write a short letter describing the simulations.
