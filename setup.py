from setuptools import find_packages, setup

BUILD_REQ = ["numpy", "scipy"]
INSTALL_REQ = BUILD_REQ
INSTALL_REQ += [
                "numba",
                "tqdm",
                "healpy",
                "astropy",
                "scikit-learn",
                "h5py",
                "pynbody",
                "joblib",
                "quadax"
                ]

setup(
    name="csiborgtools",
    version="0.3",
    description="CSiBORG analysis tools",
    url="https://github.com/Richard-Sti/csiborgtools",
    author="Richard Stiskalek",
    author_email="richard.stiskalek@protonmail.com",
    license="GPL-3.0",
    packages=find_packages(),
    python_requires=">=3.6",
    build_requires=BUILD_REQ,
    setup_requires=BUILD_REQ,
    install_requires=INSTALL_REQ,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
        ]
)
