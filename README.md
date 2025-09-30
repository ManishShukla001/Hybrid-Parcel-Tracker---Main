# Hybrid Parcel Tracker (HPT)

A high-performance hybrid C++/Python parcel tracking system optimized for atmospheric Lagrangian transport simulations using ERA5 meteorological data.

## Overview

This advanced parcel tracking system combines the computational power of C++ with the flexibility of Python to deliver production-ready performance for geoscientific applications. The hybrid architecture achieves exceptional speed while maintaining ease of use for researchers and developers. For comprehensive technical documentation see: **[System Documentation](https://manishshukla001.github.io/Hybrid-Parcel-Tracker---Main/info_page/system_documentation.html)**

### Key Features

- **🚀 High Performance**: Optimized C++ RK4 integration and 3D trilinear interpolation
- **🔄 Hybrid Architecture**: Python front-end for configuration and visualization, C++ back-end for computation
- **📊 Data Flexibility**: Supports both local CSV files and real-time ERA5 API downloads
- **🌍 Scientific Accuracy**: Proper meteorological coordinate transformations and boundary conditions
- **💾 Production Ready**: Checkpointing system for long-running simulations
- **📈 Visualization**: Built-in Cartopy-based plotting with trajectory analysis
- **🔧 Easy Integration**: Modern Python packaging with automatic compilation

## Installation

### Prerequisites

- Python 3.8 or higher
- C++ compiler (GCC, Clang, or MSVC)
- CMake (optional, we use setuptools)

### Required Python Packages

```bash
pip install numpy pandas scipy matplotlib cartopy tqdm pybind11 xarray cdsapi
```

### Build and Install

1. Clone or download this repository
2. Navigate to the project directory
3. Install in development mode:

```bash
pip install -e .
```

This will compile the C++ extensions and install the Python package.

### Verify Installation

```python
import particle_engine_cpp
print("C++ engine available!")
```

If the C++ engine is not available, the system will automatically fall back to Python-only mode (slower but functional).

## Usage

### Basic Usage

```python
from hybrid_particle_tracker.particle_tracker import HybridParticleTracker
import numpy as np
from pathlib import Path

# Configuration for CSV mode
config = {
    'data_source': "CSV",  # Data source: "CSV" or "API"
    'csv_base_dir': Path("/path/to/your/era5_csv_output"),
    'output_dir': Path("/path/to/output"),
    'checkpoint_dir': Path("/path/to/output/checkpoints"),
    'output_format': "CSV",  # "CSV" or "NETCDF"
    'pressure_levels': np.array([200, 250, 300, 350, 400, 450, 500,
                                550, 600, 650, 700, 750, 775, 800, 825,
                                850, 875, 900, 925, 950, 975, 1000]),
    # Parcel initialization
    'initialization_lat_range': (-15, 30),
    'initialization_lon_range': (30, 120),
    'initialization_pressure_levels': [1000, 925, 850, 700, 500, 300],
    'initialization_spacing_km': 10,
    # Simulation parameters
    'total_simulation_hours': 479,
    'data_interval_hours': 1,
    'simulation_step_hours': 0.5,
    'output_interval_hours': 1,
    'checkpoint_interval_hours': 6,
}

# Initialize and run
tracker = HybridParticleTracker(config)
tracker.run_simulation(resume=True)
```

### API Data Source Mode

```python
from hybrid_particle_tracker.particle_tracker import HybridParticleTracker
import numpy as np
from pathlib import Path
from datetime import datetime

# Configuration for API mode (downloads ERA5 data from Copernicus Climate Data Store)
config = {
    'data_source': "API",  # Use CDS API to download data
    'api_data_dir': Path("./api_files"),  # Directory for downloaded NetCDF files
    'output_dir': Path("./hybrid_particle_output"),
    'checkpoint_dir': Path("./hybrid_particle_output/checkpoints"),
    'output_format': "CSV",  # Output format: "CSV" or "NETCDF"
    'pressure_levels': np.array([200, 250, 300, 350, 400, 450, 500,
                                550, 600, 650, 700, 750, 775, 800, 825,
                                850, 875, 900, 925, 950, 975, 1000]),
    # Time configuration for API downloads
    'simulation_start_datetime': "2023-01-01 00:00:00",  # YYYY-MM-DD HH:MM:SS UTC
    'simulation_start_hour': 0,  # Optional: Offset from start_datetime
    'simulation_end_datetime': "2023-01-03 00:00:00",  # Optional: End datetime
    'total_simulation_hours': 48,  # Total simulation duration
    'data_interval_hours': 1,
    'simulation_step_hours': 0.5,
    'output_interval_hours': 1,
    'checkpoint_interval_hours': 6,
    # Parcel initialization
    'initialization_lat_range': (-15, 30),
    'initialization_lon_range': (30, 120),
    'initialization_pressure_levels': [1000, 925, 850, 700, 500, 300],
    'initialization_spacing_km': 10,
    # Visualization extent (optional)
    'plot_lat_range': (-20, 40),
    'plot_lon_range': (20, 130),
}

# Initialize and run
tracker = HybridParticleTracker(config)
tracker.run_simulation(resume=True)
```

### Running the Example

```bash
cd examples
python run_simulation.py
```

Make sure to adjust the paths in the example script to match your data location.

## Architecture

### File Structure

```
Hybrid-Parcel-Tracker/
├── src/                       # Source code directory
│   ├── cpp/                   # C++ implementation (< 1KB total)
│   │   ├── particle_engine.cpp    # Main C++ engine with Python bindings
│   │   ├── particle_engine.h      # Engine header
│   │   ├── interpolator.cpp       # Fast 3D trilinear interpolation
│   │   ├── interpolator.h         # Interpolation header
│   │   ├── rk4_integrator.cpp     # RK4 numerical integration
│   │   └── rk4_integrator.h       # RK4 header
│   └── python/                # Python interface (~5KB total)
│       ├── __init__.py            # Package initialization
│       ├── particle_tracker.py    # Main simulation orchestrator
│       ├── data_loader.py         # CSV/ERA5 data loading utilities
│       ├── api_data_handler.py    # Copernicus ERA5 API client
│       └── visualization.py       # Cartopy plotting and analysis
├── examples/                  # User examples and tools
│   ├── run_simulation.py       # Main configuration example
│   └── api_request.py          # ERA5 data download utility
├── Info_page/                 # Technical documentation
│   └── system_documentation.html
├── .gitignore                 # Git exclusion rules
├── pyproject.toml             # Modern Python packaging (PEP 621)
├── setup.py                   # Legacy build configuration
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Component Responsibilities

#### C++ Components
- **RegularGrid3DInterpolator**: Fast trilinear interpolation for 3D velocity fields
- **RK4Integrator**: Optimized Runge-Kutta 4th order integration
- **ParticleEngine**: Main interface coordinating interpolation and integration

#### Python Components
- **VelocityDataLoader**: Handles CSV file loading and preprocessing
- **APIDataHandler**: Downloads and processes ERA5 data from CDS API (NetCDF)
- **ParticleVisualizer**: Creates plots and visualizations with Cartopy
- **HybridParticleTracker**: Main simulation orchestrator coordinating all components

## Configuration

### Required Configuration Parameters

#### General Parameters
- `data_source`: "CSV" or "API" - Data input method
- `output_dir`: Directory for simulation outputs
- `checkpoint_dir`: Directory for checkpoint files
- `output_format`: "CSV" or "NETCDF" - Output file format
- `pressure_levels`: Array of pressure levels matching your data

#### CSV Mode Parameters (when data_source is "CSV")
- `csv_base_dir`: Path to directory containing u, v, w velocity CSV files

#### API Mode Parameters (when data_source is "API")
- `api_data_dir`: Directory to store downloaded NetCDF files
- `simulation_start_datetime`: Start time in "YYYY-MM-DD HH:MM:SS" UTC format
- `simulation_start_hour`: Optional offset from start_datetime for CSV-relative timing
- `simulation_end_datetime`: Optional end time in "YYYY-MM-DD HH:MM:SS" UTC format
- `total_simulation_hours`: Total simulation duration (fallback if simulation_end_datetime not provided)

#### Parcel Initialization Parameters
- `initialization_lat_range`: (min_lat, max_lat) for parcel distribution
- `initialization_lon_range`: (min_lon, max_lon) for parcel distribution
- `initialization_pressure_levels`: List of pressure levels for initialization [1000, 925, 850, etc.]
- `initialization_spacing_km`: Grid spacing in kilometers

#### Simulation Timing Parameters
- `data_interval_hours`: Time interval between velocity data files
- `simulation_step_hours`: Parcel position update time step
- `output_interval_hours`: How often to save parcel positions
- `checkpoint_interval_hours`: How often to save checkpoints

#### Optional Visualization Parameters
- `plot_lat_range`: (min_lat, max_lat) for plot extent
- `plot_lon_range`: (min_lon, max_lon) for plot extent

### Data Input Formats

#### CSV Mode (Local File Processing)
The system expects CSV files organized as:
```
csv_base_dir/
├── u/
│   ├── u_200_1.csv
│   ├── u_200_2.csv
│   └── ...
├── v/
│   ├── v_200_1.csv
│   └── ...
└── w/
    ├── w_200_1.csv
    └── ...
```

Each CSV file should contain columns: `Latitude`, `Longitude`, and the velocity component (`u`, `v`, or `w`).

#### API Mode (Copernicus Climate Data Store)
ERA5 data is automatically downloaded when using API mode. Files are saved as NetCDF in the `api_data_dir`:
```
api_data_dir/
├── api_daily_20230101.nc
├── api_daily_20230102.nc
└── ...
```

Downloaded data includes: u/v/w components, specific humidity (q), and temperature (t).

### Output Formats

#### CSV Output
Parcels are saved as CSV files with columns: `id`, `latitude`, `longitude`, `pressure`, and optionally `specific_humidity`, `temperature`.

#### NetCDF Output
Parcels are saved in NetCDF format with proper metadata and coordinate systems for scientific analysis.

## Performance Tips

1. **Use C++ Engine**: Ensure the C++ extension compiles successfully for maximum performance
2. **Optimize Batch Size**: The system automatically determines optimal batch sizes
3. **Memory Management**: Large simulations benefit from frequent garbage collection
4. **Checkpointing**: Use regular checkpoints to avoid losing progress
5. **Data Preprocessing**: Ensure CSV files are clean and properly formatted

## Troubleshooting

### C++ Compilation Issues

If the C++ extension fails to compile:
1. Check that you have a compatible C++ compiler
2. Ensure pybind11 is installed: `pip install pybind11`
3. Try installing with verbose output: `pip install -e . -v`

### Memory Issues

For large simulations:
1. Reduce the number of particles by increasing `spacing_km`
2. Increase `checkpoint_interval_hours` to save memory
3. Monitor system memory usage during simulation

### Data Loading Errors

If CSV files fail to load:
1. Check file paths and permissions
2. Verify CSV format matches expected structure
3. Ensure all required pressure levels have corresponding files

## Additional Resources

### Detailed Technical Documentation

For comprehensive technical documentation including mathematical formulations, algorithm details, and system architecture diagrams, see:
**[System Documentation](Info_page/system_documentation.html)**

This includes:
- Detailed RK4 integration equations
- Trilinear interpolation mathematics
- Complete system flowcharts and diagrams
- API/Scalar transport implementation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Authors

- **Manish Shukla** - Postdoctoral student, Indian Institute of Technology Hyderabad (manishshukla01@live.com)
- **R. Maheshwaran** - Assistant Professor, Indian Institute of Technology, Hyderabad

## Acknowledgments

This research software was developed at the Indian Institute of Technology Hyderabad for advanced atmospheric modeling and climate research applications. Special thanks to the Copernicus Climate Change Service (C3S) for providing the ERA5 reanalysis dataset and National Centre for Medium Range Weather Forecasting (NCMWRF) for IMDAA reanalysis dataset.
