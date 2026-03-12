# Solar Chromosphere Velocity Processor

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.com/yourusername/solar-chromosphere-velocity)

A Python package for processing solar chromospheric velocity data from CHASE mission, featuring:

- **Automatic spectral data preprocessing** for Hα and Fe lines
- **Disk exclusion** (configurable radius threshold, default: 0.95 solar radius)
- **Automatic active region detection**
- **Large-scale flow removal** (differential rotation, meridional circulation, limb shift)
- **Step-by-step visualization** at each processing stage
- **Parallel processing** for batch operations
- **Comprehensive model fitting** with parameter estimation

## New Feature: Disk Exclusion

The package now includes disk exclusion functionality that:
- Sets velocities beyond a configurable radius threshold (default: 0.95) to NaN
- Excludes these regions from active region detection
- Excludes these regions from model fitting
- Ensures cleaner analysis by removing noisy edge regions

## Installation

### From PyPI (Coming Soon)
```bash
pip install solar-chromosphere-velocity