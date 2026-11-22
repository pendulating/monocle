"""The package list. pyproject.toml holds all the other settings.

`[tool.setuptools.packages.find]` cannot find `hydra_plugins`, because it is a
namespace package (it has no __init__.py; hydra-submitit-launcher adds to the
same namespace). With `namespaces = true`, the finder walks each folder of the
repo, multirun/ and data/ included, and that takes more than 10 minutes.
Thus this file finds the regular packages and adds the plugin by name.
"""

from setuptools import find_packages, setup

setup(
    packages=find_packages(include=["dagspaces*", "research*", "sv*"])
    + ["hydra_plugins.mllmsci_searchpath"],
)
