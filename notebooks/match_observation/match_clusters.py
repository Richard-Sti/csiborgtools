import csiborgtools
from tqdm import tqdm


def open_cat(nsim, simname, bounds):
    if "csiborg1" in simname:
        cat = csiborgtools.read.CSiBORG1Catalogue(nsim, bounds=bounds)
    elif "csiborg2" in simname:
        cat = csiborgtools.read.CSiBORG2Catalogue(
            nsim, 99, simname.split("_")[-1], bounds=bounds)
    else:
        raise ValueError(f"Unknown simulation name: {simname}.")

    return cat


def open_cats(simname, bounds):
    paths = csiborgtools.read.Paths(**csiborgtools.paths_glamdring)
    nsims = paths.get_ics(simname)
    catalogues = [None] * len(nsims)

    for i, nsim in enumerate(tqdm(nsims, desc="Opening catalogues")):
        catalogues[i] = open_cat(nsim, simname, bounds)

    return catalogues