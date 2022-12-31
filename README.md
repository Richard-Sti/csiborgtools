# CSiBORG tools

## CSiBORG Matching

### TODO
- [x] Implement CIC binning or an alternative scheme for nearby objects.
- [x] Consistently locate region spanned by a single halo.
- [x] Write a script to perform the matching on a node.
- [x] Make a coarser grid for halos outside of the well resolved region.

### Questions
- What scaling of the search region? No reason for it to be a multiple of $R_{200c}$.
- How well can observed clusters be matched to CSiBORG? Do their masses agree?
- Is the number of clusters in CSiBORG consistent?


## CSiBORG Galaxy Environmental Dependence

### TODO
- [ ] Add gradient and Hessian of the overdensity field.
- [x] Write a script to smoothen an overdensity field, calculate the derived fields and evaluate them at the galaxy positions.


### Questions
- Environmental dependence of:
  - $M_*$, colour and SFR.
  - Galaxy alignment.
  - HI content.

- Fields to calculate:
    1. Overdensity field $\delta$
    2. Gradient and Hessian of $\delta$
    3. Gravitational field $\Phi$
    4. Gradient and Hessian of $\Phi$
