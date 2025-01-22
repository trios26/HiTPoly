import os
import io
from setuptools import setup, find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding="utf-8") as f:
        return f.read()


setup(
    name="hitpoly",
    version="v0.1.0",
    author="Jurgis Ruza",
    description=("High throughput polymer electrolyte MD simulation setup"),
    license="MIT",
    url="https://github.com/learningmatter-mit/HiTPoly",
    packages=find_packages("."),
    long_description=read("README.md"),
    install_requires=[
        "numpy",
    ],
)
