Macromolecular Gymnasium
========================

[![Last release](https://img.shields.io/pypi/v/macromol_gym.svg)](https://pypi.python.org/pypi/macromol_gym)
[![Python version](https://img.shields.io/pypi/pyversions/macromol_gym.svg)](https://pypi.python.org/pypi/macromol_gym)
[![Documentation](https://img.shields.io/readthedocs/macromol_gym.svg)](https://macromol-gym.readthedocs.io/en/latest)
[![Test status](https://img.shields.io/github/actions/workflow/status/kalekundert/macromol_gym/test.yml?branch=master)](https://github.com/kalekundert/macromol_gym/actions)
[![Test coverage](https://img.shields.io/codecov/c/github/kalekundert/macromol_gym)](https://app.codecov.io/github/kalekundert/macromol_gym)
[![Last commit](https://img.shields.io/github/last-commit/kalekundert/macromol_gym?logo=github)](https://github.com/kalekundert/macromol_gym)

This package builds datasets of macromolecular structures, for machine 
learning.  Some properties of these datasets:

- Capable of processing the entire PDB.[^1]

- Each data point is a particular coordinate in a particular structure.

- Only data points with sufficient density, and that feature unique 
  combinations of molecules, are included.

- The resulting database is self-contained; all of the necessary coordinates 
  are copied into the database.

- Coordinates are taken from biological assemblies, not asymmetric units.

- Train/validation/test splits are made such that all structures which share an 
  InterPro domain/family end up in the same split.

[^1]: With the only exceptions being `8h2i` and `5zz8`.  Both of these 
structures have errors in their biological assemblies, which cause the 
assemblies to have huge volumes of empty space.  Because of the way some 
internal algorithms work, these particular errors end up requiring prohibitive 
amounts of memory to process.

