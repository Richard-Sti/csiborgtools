# CSiBORG tools

## :scroll: Short-term TODO
- [x] Calculate $M_{\rm vir}, R_{\rm vir}, c$ from $R_s, \rho_0, \ldots$
- [x] In `NFWPosterior` correct for the radius in which particles are fitted.
- [x] Calculate $M_{\rm 500c}$ by sphere shrinking
- [x] Change to log10 of the scale factor
- [x] Calculate uncertainty on $R_{\rm s}$, switch to `JAX` and get gradients.
- [ ] Add functions for converting the output file to box units.
- [ ] Verify the bulk flow of particles and the clump
- [ ] Check why for some halos $M_{500c} > M_{200c}$
- [x] Remove again BIC



## :hourglass: Long-term TODO
- [ ] Improve file naming system
- [ ] Calculate the cross-correlation in CSiBORG. Should see the scale of the constraints?


## :bulb: Open questions
- Propagate uncertainty of $\log R_{\rm s}$ to concentration