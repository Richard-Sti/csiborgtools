import csiborgtools
import numpy as np
from tqdm import tqdm


def calculate_hmf(simname, bin_edges, halofinder="FOF", max_distance=135):
    """
    Calculate the halo mass function for a given simulation from catalogues.
    """
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(simname)
    bounds = {"dist": (0, max_distance)}

    counts = np.full((len(nsims), len(bin_edges) - 1), np.nan)
    for i, nsim in enumerate(tqdm(nsims)):
        if "csiborg2_" in simname:
            kind = simname.split("_")[-1]
            if halofinder == "FOF":
                cat = csiborgtools.read.CSiBORG2Catalogue(
                    nsim, 99, kind, bounds=bounds)
            elif halofinder == "SUBFIND":
                cat = csiborgtools.read.CSiBORG2SUBFINDCatalogue(
                    nsim, 99, kind, kind, bounds=bounds)
            else:
                raise ValueError(f"Unknown halofinder: {halofinder}")
        else:
            raise ValueError(f"Unknown simname: {simname}")

        counts[i] = csiborgtools.number_counts(cat["totmass"], bin_edges)

    dx = np.diff(np.log10(bin_edges))
    if not np.all(dx == dx[0]):
        raise ValueError("The bin edges must be logarithmically spaced.")
    dx = dx[0]

    volume = 4 / 3 * np.pi * max_distance**3
    counts /= volume * dx

    return counts
