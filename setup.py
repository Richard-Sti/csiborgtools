from setuptools import find_packages, setup


BUILD_REQ = ["numpy>=1.17.3", "scipy<1.9.0"]
INSTALL_REQ = BUILD_REQ
INSTALL_REQ += ["scikit-learn>=1.1.0",
                "jax[cpu]>=0.3.23",
                "tqdm>=4.64.1",
                "astropy>=5.1",
                ],

setup(
    name="csiborgtools",
    version="0.1",
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
