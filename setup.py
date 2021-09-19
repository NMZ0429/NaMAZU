from setuptools import setup
import setuptools
import NaMAZU

DESCRIPTION = "NaMAZU: Pretty Usefull Library"
NAME = "NaMAZU"
AUTHOR = "NaMAZU Team"
AUTHOR_EMAIL = "bunbun@icloud.com"
URL = "https://github.com/NMZ0429/NaMAZU"
LICENSE = "MIT"
DOWNLOAD_URL = "https://github.com/NMZ0429/NaMAZU"
VERSION = NaMAZU.__version__
PYTHON_REQUIRES = ">=3.8.0"

INSTALL_REQUIRES = ["numpy", "pillow", "pytorch_lightning"]

PACKAGES = setuptools.find_packages()
CLASSIFIERS = [
    "Environment :: Console",
    "Natural Language :: English",
    # How mature is this project? Common values are
    #   3 - Alpha, 4 - Beta, 5 - Production/Stable
    "Development Status :: 4 - Beta",
    # Indicate who your project is intended for
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Image Recognition",
    "Topic :: Scientific/Engineering :: Information Analysis",
    # Pick your license as you wish
    "License :: OSI Approved :: Apache Software License",
    "Operating System :: OS Independent",
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]

with open("README.md", "r") as fp:
    readme = fp.read()
long_description = readme

setup(
    name=NAME,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer=AUTHOR,
    maintainer_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    # extras_require=EXTRAS_REQUIRE,
    packages=PACKAGES,
    classifiers=CLASSIFIERS,
)

