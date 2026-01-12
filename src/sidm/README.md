# SIDM Codebase

This package contains the core modules for the Self-Interacting Dark Matter (SIDM) analytical model.

## Modules

- `io.py`: Handling of data input/output, specifically reading HDF5 halo data.
- `profiles.py`: Definitions of density profiles (NFW, Hernquist, coreNFW, etc.) and Jeans modeling.
- `galhalo.py`: Galaxy-halo connection logic and adiabatic contraction models.
- `cosmo.py`: Cosmology-related functions and constants.
- `config.py`: Global configuration and constants.
- `sidm_aux.py`: Auxiliary utility functions.

## Installation / Usage
This directory is a Python package. Ensure `src` is in your `PYTHONPATH` or install it using `setup.py` (if added in future).
Scripts in `analysis/` and `examples/` automatically append `src` to the path.
