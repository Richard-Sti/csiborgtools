# CSiBORG tools

## Questions to answer
- How well can observed clusters be matched to CSiBORG? Do their masses agree?
- Is the number of clusters in CSiBORG consistent?
- Are any observed clusters suspiciously missing in CSiBORG?


## Short-term TODO
- [ ] Save the $z = 70$ clumps as an array of arrays or a Python dictionary whose keys are the clump indices
- [ ] Calculate the match. How much longer does this take? Worst case dump this on Glamdring.
- [ ] Switch to CIC binning. This appears to be simply a matter of replacing a point-like particle with a cell and then assigning its mass to the appropriate grid cells. Sounds like some convolution operation?


## Long-term TODO
- [ ] Implement a custom model for matchin galaxies to halos.


## Open questions
- What scaling of the search region? No reason for it to be a multiple of $R_{200c}$.
- Begin extracting the DM environmental properties at galaxy positions?