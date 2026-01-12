# SIDM: Self-Interacting Dark Matter Analytical Model

This repository contains the analytical model for the density profile of Self-Interacting Dark Matter (SIDM) halos with inhabitant galaxies.

## Structure

- **`src/sidm/`**: Core Python package containing the modeling logic, profiles, and cosmology functions.
- **`data/`**: Directory for storing HDF5 simulation data files.
- **`analysis/`**: Scripts for running systematic analysis on halo catalogs.
- **`examples/`**: Demonstration scripts showing how to use the model.
- **`tests/`**: Unit and integration tests.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy, SciPy, Matplotlib, h5py, pandas, lmfit

### Installation

No special installation is required if you are running scripts from the `analysis` or `examples` directories, as they are configured to find the `sidm` package.

To use the `sidm` package in your own external scripts, add `src` to your `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/SIDM/src
```

### Usage Example

To run the main analysis on a dataset:

```bash
cd analysis
python run_halo_analysis.py ../data/Halo_data_L025N376ReferenceSigmaVelDep30Anisotropic.hdf5
```

To see a demo of the solver:

```bash
cd examples
python demo_solve_sidm.py
```

## References

- Jiang et al. (2022) - *A semi-analytic study of self-interacting dark-matter haloes with baryons*
