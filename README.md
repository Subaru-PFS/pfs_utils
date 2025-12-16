# PFS Utilities

Common utility tools for the Subaru Prime Focus Spectrograph (PFS) Data Reduction Pipeline.

## Overview

The `pfs_utils` package provides a collection of utilities for working with data from the Prime Focus Spectrograph (PFS)
instrument at the Subaru Telescope. These utilities support various aspects of the PFS Data Reduction Pipeline (DRP) and
instrument operation.

PFS is a wide-field, multi-object spectrograph capable of simultaneously obtaining spectra for up to 2,400 astronomical
targets. This package contains tools essential for processing PFS data and managing the instrument's components.

## Special Note on Dependencies

**Important**: `pfs_utils` (and its dependency on `pfs_datamodel`) is the only repository that is used in both the data
reduction pipeline (DRP) code that exists in the `pfs` namespace as well as the instrument control software (ICS) code
that is used by various actors. Because of this dual usage, further `pfs` dependencies should not be added to this
module.

## Features

- **Coordinate Transformations**: Tools for transforming between different coordinate systems used by the PFS
  instrument, including:
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

## Usage

### Database usage

- Authentication: Passwords are expected to be managed externally by libpq (e.g., via `~/.pgpass`). The helpers use
  `psycopg` through SQLAlchemy and do not embed passwords.

- Engine caching (singleton per DSN URL): `DB` now caches a SQLAlchemy `Engine` per DSN URL (as built by
  `DB._build_url()`). Multiple `DB` instances that point to the same DSN URL will share the same underlying connection
  pool/engine. Creating a `DB` instance does not make the class itself a singleton anymore. If you change the `dsn` (or
  connection parameters) such that the URL changes, the old engine is disposed and a new one will be created lazily on
  next use.

The two most common operations are `query` (reading) and `insert` (writing). By default, `query` returns a pandas
`DataFrame`, and `insert` accepts a pandas `DataFrame` for bulk inserts. Both also support other convenient options.

#### Connecting

You can use the `DB` class directly or the convenience subclasses `OpDB`/`QaDB` that provide default connection
settings. Engines are shared per DSN URL, so separate `DB` objects with identical connection settings will reuse the
same engine and connection pool.

```python
from pfs.utils.database.db import DB
from pfs.utils.database.opdb import OpDB

# 1) Construct DB() with explicit parameters
db1 = DB(dbname="opdb", user="pfs", host="localhost", port=5432)
with db1.connection() as conn:
    conn.exec_driver_sql("SELECT 1")

# 2) Another DB() with the same DSN URL will share the same engine/pool
db2 = DB(dbname="opdb", user="pfs", host="localhost", port=5432)
assert db1 is not db2  # different DB objects
# but operations reuse the same underlying Engine (cached per DSN URL)
with db2.connection() as conn:
    conn.exec_driver_sql("SELECT 1")

# 3) Operational DB convenience class (uses project defaults)
opdb = OpDB()
```

#### query — default (DataFrame) and alternatives

```python
from pfs.utils.database.opdb import OpDB
opdb = OpDB()

frame_id = 123456

# Default returns a pandas DataFrame. 
df = opdb.query_dataframe(
    "SELECT pfs_visit_id, issued_at FROM pfs_visit ORDER BY pfs_visit_id DESC LIMIT 5"
)

# Query with named parameters. `query` is an alias for `query_dataframe`.
df2 = opdb.query(
    "SELECT * FROM agc_match WHERE agc_exposure_id = :frame_id",
    params={"frame_id": frame_id},
)

# Return a single row as a pandas Series
row_series = opdb.query_series(
    "SELECT * FROM agc_match WHERE agc_exposure_id = :frame_id ORDER BY spot_id LIMIT 1",
    params={"frame_id": frame_id},
)

# Return all rows as a NumPy array of Row objects (back-compat style)
rows_array = opdb.query_array(
    "SELECT agc_exposure_id, spot_id FROM agc_match WHERE agc_exposure_id = :frame_id ORDER BY spot_id",
    params={"frame_id": frame_id},
)

# Return a single scalar value
num_detections = opdb.query_scalar(
    "SELECT COUNT(*) FROM agc_match WHERE agc_exposure_id = :frame_id",
    params={"frame_id": frame_id},
)
```

See other query variants in the API docs.

#### insert — default (DataFrame) and alternatives

```python
import pandas as pd
from pfs.utils.database.opdb import OpDB

opdb = OpDB()

# 1) Bulk insert with a DataFrame. 
# Column names must match the destination table columns.
df_to_insert = pd.DataFrame([
    {"agc_exposure_id": 123456, "spot_id": 1, "x": 10.5, "y": -2.3},
    {"agc_exposure_id": 123456, "spot_id": 2, "x": 11.1, "y": -2.0},
])
opdb.insert_dataframe(table="agc_match", df=df_to_insert)

# 2) Insert a single row using keyword arguments.
opdb.insert_kw("agc_match", agc_exposure_id=123456, spot_id=3, x=10.9, y=-2.1)

# 3) DataFrame options: include index (default: False) or adjust chunksize (default: 10000).
opdb.insert_dataframe(table="agc_match", df=df_to_insert, index=True, chunksize=5000)

# 4) Generic `insert` is an alias for `insert_dataframe`.
opdb.insert(table="agc_match", df=df_to_insert)
```

#### Reusing a single connection

Each helper acquires a pooled connection for the duration of the call. To run multiple statements in the same session,
use the connection context manager:

```python
from sqlalchemy import text
import pandas as pd
from pfs.utils.database.opdb import OpDB

opdb = OpDB()

# Trivial example to re-use connection for multiple operations. 
# Note that this re-creates the default of `query` but less efficiently.
with opdb.connection() as conn:
    conn.execute(text("SET LOCAL statement_timeout = 5000"))
    
    # Get the columns from the `results` metadata.
    res = conn.execute(text("SELECT * FROM pfs_visit WHERE false"))
    column_names = list(res.keys())
    
    # Get the results as a numpy array with original types.
    visits_array = opdb.query_array(
        "SELECT * FROM pfs_visit ORDER BY pfs_visit_id DESC LIMIT 10", 
        conn=conn, 
    )
    
    # Create custom dataframe.
    visits = pd.DataFrame(visits_array, columns=column_names)
```

Notes

- Connection pooling: `DB`/`OpDB` cache a SQLAlchemy `Engine` with pooling. Each helper method checks out a connection for the duration of the call. Use `db.connection()` to explicitly reuse a single connection.

## Installation

### Requirements

- Python 3.12 or later
- Dependencies listed in `pyproject.toml`

### EUPS Installation with LSST Stack

This package uses the Extended Unix Product System (EUPS) for dependency management and environment setup, which is part
of the LSST Science Pipelines software stack. The LSST stack is a comprehensive framework for astronomical data
processing that provides powerful tools for image processing, astrometry, and data management.

1. Ensure you have the LSST stack installed on your system. If not, follow the installation instructions at
   the [LSST Science Pipelines documentation](https://pipelines.lsst.io/install/index.html).

2. Once the LSST stack is set up, declare and setup this package using EUPS:
   ```
   eups declare -r /path/to/pfs_utils pfs_utils git
   setup -r /path/to/pfs_utils
   ```

3. The package's EUPS table file (`ups/pfs_utils.table`) will automatically set up the required dependencies within the
   LSST stack environment:
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

This project is part of the Subaru Prime Focus Spectrograph (PFS) project and is subject to the licensing terms of the
PFS collaboration.

### Contact

For questions or issues related to this software, please contact the PFS software team or create an issue in the
repository.
