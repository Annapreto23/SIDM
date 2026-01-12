# Analysis Scripts

This directory contains scripts for performing analysis on halo data.

## Scripts

- `run_halo_analysis.py`: (Formerly `class_phases.py`) Main script to run halo analysis, calculate cross-sections, and estimate errors.
  - **Usage**: `python run_halo_analysis.py <path_to_hdf5_file>`
  
- `plot_phase_space.py`: (Formerly `phase.py`) Script to generate phase space plots and compare theoretical models with simulation data.
  - **Usage**: `python plot_phase_space.py` (Edit file paths within script or rely on dummy data fallback)

## Dependencies
These scripts rely on the `sidm` package located in `../src`.
