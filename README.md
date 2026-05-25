# Hybrid Particle Tracker (HPT)

A high-performance, hybrid C++/Python particle tracking system optimized for atmospheric particle simulations using ERA5, HRRR, or custom NetCDF/CSV meteorological datasets.

## Overview

The Hybrid Particle Tracker (HPT) provides a significant performance improvement over pure Python lagrangian models by moving computationally intensive numerical integration and 3D interpolation routines to optimized C++ (utilizing OpenMP for multi-threaded parallel execution) while keeping data I/O, setup, API calls, and visualization in a flexible Python interface.

### Key Features

* **High Performance**: Particle updates run **10-50x faster** by executing Runge-Kutta 4th order (RK4) integration and 3D trilinear interpolation in C++.
* **Parallel Execution**: Supports OpenMP multi-threading to parallelize integration steps over large particle populations.
* **Flexible I/O (CSV, NetCDF, or API)**:
  * **CSV**: Legacy grid-separated components.
  * **NetCDF (NC)**: Load local `.nc` files (e.g. combined HRRR/ERA5 datasets).
  * **API**: Download hourly data dynamically from the Copernicus Climate Data Store (CDS).
* **Thermodynamic Analysis Tracking**:
  * **Simple Subtraction**: Tracks raw changes in temperature ($\Delta T$).
  * **Potential Temperature & Dry Static Energy**: Factors out adiabatic processes along the trajectory.
  * **Full Anomaly Decomposition (Papritz & Röthlisberger)**: Decomposes temperature anomalies ($T'$) into integrated seasonality, advection, adiabatic, and diabatic drivers.
* **Flexible Saving formats**: Save output states in standard CSV or as structured NetCDF files (using xarray and netcdf4 backend).
* **Checkpointing**: Automatic resilience checkpointing to resume simulations from intermediate states.

---

## Installation & Setup

### Prerequisites
* Python 3.8 or higher
* A C++ compiler supporting C++17 (GCC, Clang, or MSVC)
* OpenMP library installed (for parallel mode)

### Required Python Packages
Dependencies are declared in `requirements.txt` and `pyproject.toml`. Install them together with the package using:
```bash
pip install numpy pandas scipy matplotlib cartopy tqdm pybind11 xarray netcdf4 cdsapi seaborn
```

### Build and Install (Unified Script)
We provide a unified [build.py](file: build.py) script to clean previous build artifacts, install dependencies, build/compile the C++ bindings in editable mode, and run integration validation:
```bash
python build.py
```
This compiles the C++ codebase to produce the `particle_engine_cpp` module and links it to the local `hybrid_particle_tracker` Python package.

---

## Usage

To run the tracking simulation, customize the configuration in your runner script (similar to [run_simulation.py](file: examples/run_simulation.py)):

```python
import numpy as np
from pathlib import Path
from hybrid_particle_tracker.particle_tracker import HybridParticleTracker

# Configuration Dictionary
config = {
    # Data source can be "CSV", "NC" / "NETCDF", or "API"
    'data_source': "NC",
    'nc_data_dir': Path("./examples/nc_files"),
    'nc_file_pattern': None, # Auto-detects combined_YYYYMMDD_HH.nc
    
    # Execution mode: "serial" or "parallel" (enables OpenMP)
    'execution_mode': "parallel",
    
    # Storage details
    'output_dir': Path("./examples/hybrid_particle_output"),
    'output_format': "NETCDF", # Save outputs as "NETCDF" or "CSV"
    'checkpoint_dir': Path("./examples/hybrid_particle_output/checkpoints"),
    
    # Particle Initialization Grid
    'initialization_lat_range': (-15, 65),
    'initialization_lon_range': (-135, -45),
    'initialization_pressure_levels': [1000, 950, 850, 700, 500, 300],
    'initialization_spacing_km': 30,
    
    # Simulation Timing 
    'simulation_start_datetime': "2025-03-21 00:00:00",
    'simulation_end_datetime': "2025-03-21 10:00:00",
    'simulation_start_hour': 0,
    'total_simulation_hours': 10,
    'data_interval_hours': 1,
    'simulation_step_hours': 0.25,
    'output_interval_hours': 1,
    'checkpoint_interval_hours': 3,
    
    # Thermodynamic Tracking Mode
    # Options: 'NONE', 'SIMPLE_SUBTRACTION', 'POTENTIAL_TEMPERATURE', 'FULL_DECOMPOSITION'
    'thermo_mode': 'POTENTIAL_TEMPERATURE',
    'thermo_bg_file': Path("./climatology_background.nc"), # Required for FULL_DECOMPOSITION
}

# Initialize and run
tracker = HybridParticleTracker(config)
tracker.run_simulation(resume=True)
```

Running the examples:
```bash
cd examples
python run_simulation.py
```

---

## Running on HPC Clusters (PBS / Conda)

HPT is designed to run efficiently on High-Performance Computing (HPC) clusters using scheduler engines like PBS. Computing nodes on typical HPC clusters do not have internet access, so HPT uses an **offline build workflow** that compiles the C++ library in-place.

### PBS Job Submission Script

An example job script [run_job.pbs](file: run_job.pbs) is provided in the repository root. You can submit it to the cluster scheduler using `qsub`:

```bash
qsub run_job.pbs
```

Here is a template of the PBS job configuration:

```bash
#!/bin/bash
#PBS -N HPT_Simulation
#PBS -l select=1:ncpus=90
#PBS -l walltime=240:00:00
#PBS -q your_queue_name
#PBS -j oe
#PBS -o HPT_Simulation.log
#PBS -e HPT_error.log

# --- 1. Environment & Compiler Setup ---
# Load GCC and CMake to compile the C++ extension on the cluster node
module load gcc
module load cmake

# Activate your pre-configured Conda/Miniforge environment containing dependencies
# (e.g., numpy, pandas, scipy, xarray, netcdf4, etc.)
source $HOME/miniforge3/bin/activate env_name

# Navigate to the workspace directory
cd "$PBS_O_WORKDIR"

# --- 2. Offline Build (In-Place) ---
# Compiles the C++ .so file directly in the folder without checking remote packages
python setup.py build_ext --inplace

# --- 3. Runtime Configuration ---
# Add working directory to PYTHONPATH to locate the built binaries and src python packages
export PYTHONPATH="$PBS_O_WORKDIR:$PYTHONPATH"

# Map the OpenMP thread count to the requested number of CPUs
if [ -n "$PBS_NP" ]; then
    export OMP_NUM_THREADS=$PBS_NP
else
    export OMP_NUM_THREADS=90
fi

# --- 4. Run Simulation ---
python examples/run_simulation.py
```

### Key Considerations for HPC Runs:
1. **Offline In-Place Compilation**: Computing nodes are usually offline. Instead of running `pip install -e .` (which attempts to query external PyPI servers), run `python setup.py build_ext --inplace` to build the compiled `.so` library locally.
2. **Environment Variable Configuration**: Running `export PYTHONPATH="$PBS_O_WORKDIR:$PYTHONPATH"` ensures the in-place compiled `particle_engine_cpp` module is discoverable by the Python interpreter without requiring system-wide permissions.
3. **OMP Thread Allocation**: Mapping `OMP_NUM_THREADS` dynamically to `$PBS_NP` allows the C++ engine to scale parallel operations to exactly the number of CPU cores allocated by the PBS scheduler.

---

## File Architecture

```
HPT_Texas/
├── src/
│   ├── cpp/                                # C++ core implementation
│   │   ├── particle_engine.cpp             # Main C++ engine & pybind11 bindings
│   │   ├── particle_engine.h               # Main header
│   │   ├── interpolator.cpp                # 3D trilinear interpolation
│   │   ├── interpolator.h                  # Interpolator header
│   │   ├── rk4_integrator.cpp              # Serial Runge-Kutta 4 integration
│   │   ├── rk4_integrator.h                # Serial integrator header
│   │   ├── rk4_integrator_parallel.cpp     # Parallel RK4 using OpenMP
│   │   ├── rk4_integrator_parallel.h       # Parallel integrator header
│   │   └── thermo_state.h                  # Thermodynamic calculation module
│   └── python/                             # Python orchestrator package
│       ├── __init__.py                     # Package imports & namespaces
│       ├── particle_tracker.py             # Main orchestrator & controller class
│       ├── data_loader.py                  # CSV data loader
│       ├── nc_data_loader.py               # NetCDF data loader (HRRR/ERA5)
│       ├── api_data_handler.py             # CDS API loader
│       ├── data_downloader_method2.py      # ERA5 Method 2 Climatology downloader
│       └── visualization.py                # Trajectory plotting utilities
├── examples/
│   ├── run_simulation.py                   # Example execution script
│   └── api_request.py                      # Raw client API download script
|
├── additional_files/
│   ├── debug_cpp_interpolater              # Scripts to convert/combine HRRR and ERA5 datasets
│   ├── Input_Data_Preparation/             # Scripts to convert/combine HRRR and ERA5 datasets
│   ├── compare_thermo_results.py           # Comparison script for thermo mode validations
│   ├── verify_performance.py               # Speed benchmarker script
│   ├── Imdaa_Extract*.py                   # Scripts to extract data from Imdaa files
│   └── README_Details_of_additional_files  # More details about the additional files
│  
├── setup.py                                # C++ Compilation config
├── pyproject.toml                          # Modern PEP 518 packaging configuration
├── requirements.txt                        # Python dependencies
└── build.py                                # Unified build, install, and validation script
```

---

## Component Details

### Thermodynamic Tracking Modes
1. **`SIMPLE_SUBTRACTION`**: Tracks temperature changes purely via basic subtraction:
   $$\Delta T = T_t - T_0$$
2. **`POTENTIAL_TEMPERATURE` / `DRY_STATIC_ENERGY`**:
   Factors out expansion/compression by tracking:
   * Potential Temperature ($\theta$):
     $$\theta = T \left(\frac{P_0}{P}\right)^\kappa$$
     where $P_0 = 1000$ hPa and $\kappa \approx 0.286$.
   * Dry Static Energy (DSE):
     $$DSE = C_p T + g z$$
     where $C_p = 1004 \text{ J kg}^{-1}\text{K}^{-1}$ and $g = 9.81 \text{ m s}^{-2}$.
3. **`FULL_DECOMPOSITION` (Papritz & Röthlisberger)**:
   Decomposes anomaly changes ($T'$) into four integrated physical drivers along the trajectory:
   * **Seasonality**: $-\int \frac{\partial \overline{T}}{\partial t} d\tau$
   * **Advection**: $-\int \mathbf{v} \cdot \nabla_h \overline{T} d\tau$
   * **Adiabatic**: $\int \left[\frac{\kappa \overline{T}}{p} - \frac{\partial \overline{T}}{\partial p}\right] \omega d\tau$
   * **Diabatic**: $\int \left(\frac{p}{p_0}\right)^\kappa \frac{D\theta}{Dt} d\tau$

### Background Climatology Data Preparation
`FULL_DECOMPOSITION` requires a climatological background mean temperature field ($\overline{T}$) and its gradients. Use the background downloader utility to fetch and prepare this from the CDS API:
```bash
python src/python/data_downloader_method2.py --year 2025 --month 3 --days 21 22 --out_file climatology_background.nc
```
This utility fetches the raw ERA5 background fields over a 21-day centered window across historical baselines and computes horizontal ($\nabla_h \overline{T}$), vertical ($\frac{\partial \overline{T}}{\partial p}$), and temporal ($\frac{\partial \overline{T}}{\partial t}$) derivatives.

---

## Troubleshooting

### Console Encoding Errors
On Windows environments, stdout redirections might trigger `UnicodeEncodeError` due to local codepage conflicts. Ensure you use the ASCII-safe [build.py](file: build.py) utility which prints ASCII status messages.

### C++ Compiler Errors
* If compilation fails due to missing OpenMP, make sure OpenMP is installed on your system:
  * **Windows**: OpenMP is built into MSVC compilers.
  * **Linux**: Install `libomp-dev` or `libgomp`.
  * **Mac**: Install `libomp` via Homebrew (`brew install libomp`).
* Try executing `python build.py` to get full traceback error output.