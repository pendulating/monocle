"""The package list. pyproject.toml holds all the other settings.

`[tool.setuptools.packages.find]` walks each folder of the repo, multirun/ and
data/ included, and that takes more than 10 minutes. Thus this file names the
packages.
"""

from setuptools import find_packages, setup

setup(packages=find_packages(include=["research*", "sv*"]))
