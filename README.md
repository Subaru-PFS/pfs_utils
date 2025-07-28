# PFS Utilities

Common utility tools for the Subaru Prime Focus Spectrograph (PFS) Data Reduction Pipeline.

## Overview

The `pfs_utils` package provides a collection of utilities for working with data from the Prime Focus Spectrograph (PFS) instrument at the Subaru Telescope. These utilities support various aspects of the PFS Data Reduction Pipeline (DRP) and instrument operation.

PFS is a wide-field, multi-object spectrograph capable of simultaneously obtaining spectra for up to 2,400 astronomical targets. This package contains tools essential for processing PFS data and managing the instrument's components.

## Special Note on Dependencies

**Important**: `pfs_utils` (and its dependency on `pfs_datamodel`) is the only repository that is used in both the data reduction pipeline (DRP) code that exists in the `pfs` namespace as well as the instrument control software (ICS) code that is used by various actors. Because of this dual usage, further `pfs` dependencies should not be added to this module.

## Features

- **Coordinate Transformations**: Tools for transforming between different coordinate systems used by the PFS instrument, including:
  - Metrology Camera System (MCS) coordinates
  - Prime Focus Instrument (PFI) coordinates
  - Sky coordinates
  - Distortion correction and measurement

- **Fiber Management**: Utilities for working with the fiber system, including:
  - Fiber ID calculation and conversion
  - Fiber positioning and configuration
  - Cobra positioner management

- **Data Model Integration**: Tools for working with the PFS data model and Butler data management system

- **Instrument Configuration**: Constants and parameters for the PFS instrument configuration

## Installation

### Requirements

- Python 3.12 or later
- Dependencies listed in `pyproject.toml`

### EUPS Installation with LSST Stack

This package uses the Extended Unix Product System (EUPS) for dependency management and environment setup, which is part of the LSST Science Pipelines software stack. The LSST stack is a comprehensive framework for astronomical data processing that provides powerful tools for image processing, astrometry, and data management.

1. Ensure you have the LSST stack installed on your system. If not, follow the installation instructions at the [LSST Science Pipelines documentation](https://pipelines.lsst.io/install/index.html).

2. Once the LSST stack is set up, declare and setup this package using EUPS:
   ```
   eups declare -r /path/to/pfs_utils pfs_utils git
   setup -r /path/to/pfs_utils
   ```

3. The package's EUPS table file (`ups/pfs_utils.table`) will automatically set up the required dependencies within the LSST stack environment:
   - pfs_instdata
   - pfs_datamodel

### Standard Setup

Alternatively, you can install the package using pip:

```bash
pip install git+https://github.com/Subaru-PFS/pfs_utils.git
```

### Development Installation

```bash
git clone https://github.com/Subaru-PFS/pfs_utils.git
cd pfs_utils
pip install -e .
```

## Project Structure

- `python/pfs/utils/coordinates/`: Coordinate transformation utilities
- `python/pfs/utils/datamodel/`: Data model integration
- `python/pfs/utils/`: General utilities for fiber management, configuration, etc.
- `data/`: Data files used by the utilities
- `tests/`: Unit tests
- `docs/`: Documentation
- `notebooks/`: Jupyter notebooks with examples

## Dependencies

- `pfs-datamodel`: PFS data model package
- `numpy` (>= 2.0): Numerical computing
- `astropy`: Astronomical calculations
- `matplotlib`: Plotting and visualization
- `pandas`: Data manipulation
- `scipy`: Scientific computing
- `astroplan`: Observation planning
- `pytz`: Timezone handling

## Development 

### Contributing

Contributions to `pfs_utils` are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the tests to ensure they pass
5. Submit a pull request

### License

This project is part of the Subaru Prime Focus Spectrograph (PFS) project and is subject to the licensing terms of the PFS collaboration.

### Contact

For questions or issues related to this software, please contact the PFS software team or create an issue in the repository.
