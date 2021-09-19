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
    # classifiers=CLASSIFIERS,
)

