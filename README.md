# Hybrid Particle Tracker (HPT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://www.python.org/)
[![C++ Bindings](https://img.shields.io/badge/C%2B%2B-pybind11-blue)](https://github.com/pybind/pybind11)
[![OpenMP Acceleration](https://img.shields.io/badge/OpenMP-accelerated-orange)](https://www.openmp.org/)
[![System Documentation](https://img.shields.io/badge/docs-system%20documentation-brightgreen)](https://manishshukla001.github.io/Hybrid-Parcel-Tracker---Main/)

HPT is a high-performance, hybrid C++/Python Lagrangian particle tracking system optimized for simulating air parcel trajectories using ERA5, HRRR, or custom NetCDF/CSV meteorological datasets.

---

## 📖 System Documentation

We have prepared a comprehensive, interactive HTML system documentation illustrating coordinates, algorithms, flows, and interpolation details:
* **Live Webpage (Recommended)**: [https://manishshukla001.github.io/Hybrid-Parcel-Tracker---Main/](https://manishshukla001.github.io/Hybrid-Parcel-Tracker---Main/)
* **Local Workspace**: Open [index.html](file: index.html) or [system_documentation.html](file: info_page/system_documentation.html) in your browser.

---

## 👥 Authors & Academic Context

* **Manish Shukla** (Postdoctoral Fellow) — [manishshukla01@live.com](mailto:manishshukla01@live.com)
* **R. Maheshwaran** (Assistant Professor)
* **Institution**: Indian Institute of Technology Hyderabad (IIT Hyderabad), India
* **License**: Licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🚀 Key Features

* **High Performance Core**: Integration and interpolation routines run in C++, offering **10-50x speedups** over pure Python advection models.
* **Parallel Execution**: Multi-threaded parallel processing of particle populations using OpenMP.
* **Flexible I/O Formats**:
  * **CSV**: Individual, grid-separated velocity files.
  * **NetCDF (NC)**: Integrated multi-variable datasets (e.g. combined HRRR/ERA5 files).
  * **Copernicus API**: On-the-fly downloads from the Copernicus Climate Data Store (CDS).
* **Advanced Thermodynamic Tracking**:
  * *Simple Subtraction*: Tracks basic temperature changes ($\Delta T$).
  * *Potential Temperature & Dry Static Energy*: Filters out adiabatic expansion/compression along trajectories.
  * *Full Decomposition (Papritz & Röthlisberger)*: Computes integrated seasonality, advection, adiabatic, and diabatic temperature anomaly drivers.
* **Resilient Checkpointing**: Automatic state serialization to pause and resume runs without loss.

---

## 🛠️ Installation & Build

### Prerequisites
1. Python 3.8 or higher
2. C++ Compiler supporting C++17 (`gcc`, `clang`, or `MSVC`)
3. OpenMP libraries installed (built into MSVC; for Linux install `libomp-dev`)

### Python Dependencies
Install all package requirements:
```bash
pip install numpy pandas scipy matplotlib cartopy tqdm pybind11 xarray netcdf4 cdsapi seaborn
```

### Build & Verify (Unified Script)
We provide a unified [build.py](file: build.py) script to clean previous build artifacts, install dependencies, compile C++ bindings, install the package locally, and run core module validation:
```bash
python build.py
```

---

## 💻 Usage

To run the tracking simulation, configure the simulation parameters in your runner script (see [run_simulation.py](file: examples/run_simulation.py)):

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

## 🌐 Running on HPC Clusters (PBS)

HPT supports execution on High-Performance Computing (HPC) clusters using schedulers like PBS. Because compute nodes usually run offline, HPT compiles the library in-place.

You can submit jobs via the scheduler using `qsub`:
```bash
qsub run_job.pbs
```

An example template for [run_job.pbs](file: run_job.pbs) is provided below:
```bash
#!/bin/bash
#PBS -N HPT_Simulation
#PBS -l select=1:ncpus=90
#PBS -l walltime=240:00:00
#PBS -q your_queue_name
#PBS -j oe
#PBS -o HPT_Simulation.log
#PBS -e HPT_error.log

# Load compiler and tools
module load gcc
module load cmake

# Activate Conda environment
source $HOME/miniforge3/bin/activate env_name

# Navigate to working directory
cd "$PBS_O_WORKDIR"

# Compile C++ extensions in-place (offline friendly)
python setup.py build_ext --inplace

# Add folder to Python Path
export PYTHONPATH="$PBS_O_WORKDIR:$PYTHONPATH"

# Map requested CPUs to OpenMP Thread Count
if [ -n "$PBS_NP" ]; then
    export OMP_NUM_THREADS=$PBS_NP
else
    export OMP_NUM_THREADS=90
fi

# Run simulation
python examples/run_simulation.py
```

---

## 📂 Repository File Structure

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
│
├── additional_files/
│   ├── debug_cpp_interpolater              # Interpolator debugger utility
│   ├── Input_Data_Preparation/             # Scripts to convert/combine HRRR and ERA5 datasets
│   ├── compare_thermo_results.py           # Comparison script for thermo mode validations
│   ├── verify_performance.py               # Speed benchmarker script
│   ├── Imdaa_Extract*.py                   # Scripts to extract data from Imdaa files
│   └── README_Details_of_additional_files  # More details about the additional files
│  
├── setup.py                                # C++ Compilation config
├── pyproject.toml                          # Modern PEP 518 packaging configuration
├── requirements.txt                        # Python dependencies
├── index.html                              # Redirect landing page for GitHub Pages documentation
└── build.py                                # Unified build, install, and validation script
```

---

## 🌡️ Thermodynamic Tracking Modes

### 1. Simple Subtraction (`SIMPLE_SUBTRACTION`)
Calculates the basic temperature difference along the parcel's trajectory:
$$\Delta T = T_t - T_{t_0}$$

### 2. Potential Temperature & Dry Static Energy (`POTENTIAL_TEMPERATURE`)
Factors out expansion and compression along vertical movements:
* **Potential Temperature** ($\theta$):
  $$\theta = T \left(\frac{P_0}{P}\right)^\kappa$$
  where $P_0 = 1000 \text{ hPa}$ and $\kappa \approx 0.286$ (specific dry air gas constant ratio $R/C_p$).
* **Dry Static Energy** ($DSE$):
  $$DSE = C_p T + g z$$
  where $C_p = 1004 \text{ J kg}^{-1}\text{K}^{-1}$, $g \approx 9.81 \text{ m s}^{-2}$, and $z$ is geopotential height.

### 3. Full Anomaly Decomposition (`FULL_DECOMPOSITION`)
Decomposes temperature anomaly variations ($T'$) into four integrated physical drivers along paths:
$$T'(x,t) - T'(x_0,t_0) = \underbrace{-\int_{t_0}^{t} \frac{\partial \overline{T}}{\partial t} d\tau}_{\text{Seasonality}} \underbrace{-\int_{t_0}^{t} \mathbf{v} \cdot \nabla_h \overline{T} d\tau}_{\text{Advective}} + \underbrace{\int_{t_0}^{t} \left[\frac{\kappa \overline{T}}{p} - \frac{\partial \overline{T}}{\partial p}\right] \omega d\tau}_{\text{Adiabatic}} + \underbrace{\int_{t_0}^{t} \left(\frac{p}{p_0}\right)^\kappa \frac{D\theta}{Dt} d\tau}_{\text{Diabatic}}$$

To prepare the background climatology temperature field ($\overline{T}$ and derivatives), run the background download script prior to the simulation:
```bash
python src/python/data_downloader_method2.py --year 2025 --month 3 --days 21 22 --out_file climatology_background.nc
```

---

## 🔧 Troubleshooting

### Console Encoding Errors
On Windows environments, stdout redirections might trigger `UnicodeEncodeError` due to local codepage conflicts. Ensure you use the ASCII-safe [build.py](file: build.py) utility which prints ASCII status messages.

### OpenMP Compilation Errors
If the C++ build fails due to a missing compiler or OpenMP library:
* **macOS**: Install LLVM/Clang and OpenMP via homebrew: `brew install libomp`.
* **Linux**: Ensure `libomp-dev` is installed.
* **Windows**: OpenMP is built directly into MSVC. If using MinGW, install `pthreads` and `gomp` extensions.

---

## 📚 References

Shukla, M., Ganapathiraju, S. A., Pérez-Alarcón, A., & Rathinasamy, M. (2026). Hybrid Parcel Tracker (HPT) – A Python-based framework to analyse moisture movement during extreme precipitation events. Atmospheric Research, 109136. https://doi.org/10.1016/j.atmosres.2026.109136
