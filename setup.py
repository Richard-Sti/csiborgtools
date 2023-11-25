from setuptools import find_packages, setup

# List of dependencies:
#   - Corrfunc  -> To be moved to a separate package.
#   - NumPy
#   - SciPy
#   - Numba
#   - Pylians
#   - tqdm
#   - healpy
#   - astropy
#   - scikit-learn
#   - joblib
#   - h5py
#   - MPI
#   - pyyaml
#   - taskmaster
#   - matplotlib
#   - scienceplots
#   - cache_to_disk



Pylians
Corrfunc
numpy
scipy
healpy
numba
tqdm
astropy
scikit-learn
treelib
pynbody
joblib


BUILD_REQ = ["numpy", "scipy"]
INSTALL_REQ = BUILD_REQ
INSTALL_REQ += ["Corrfunc",
                "Pylians",
                "numba",
                "tqdm",
                "healpy",
                "astropy",
                "scikit-learn",
                "h5py",
                "matplotlib",
                "scienceplots",
                "mpi4py",
                "pyyaml",
                "joblib",]


setup(
    name="csiborgtools",
    version="0.2",
    description="CSiBORG analysis tools",
    url="https://github.com/Richard-Sti/csiborgtools",
    author="Richard Stiskalek",
    author_email="richard.stiskalek@protonmail.com",
    license="GPL-3.0",
    packages=find_packages(),
    python_requires=">=3.8",
    build_requires=BUILD_REQ,
    setup_requires=BUILD_REQ,
    install_requires=INSTALL_REQ,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"]
)
