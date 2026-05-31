#!/usr/bin/env python3
"""
Example script to run the hybrid particle tracker simulation
"""

import numpy as np
from pathlib import Path
import sys

# Import the installed package
try:
    from hybrid_particle_tracker.particle_tracker import HybridParticleTracker
except ImportError:
    # Fallback to direct import from src
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))
    from particle_tracker import HybridParticleTracker


def main():
    """Run the particle tracking simulation"""
    
    # Configuration - adjust these paths to match your setup
    config = {
        # Data Source: "CSV", "API", or "NC"
        'data_source': "API", # Choose "API" to use Copernicus API, "CSV" for local CSV, "NC" for NetCDF
        
        # Execution Mode: "serial" (default) or "parallel" (uses OpenMP)
        'execution_mode': "parallel", 
        
        # Data paths (adjust to your actual paths)
        'csv_base_dir': Path("./examples/CSV"), # Used if data_source is "CSV"
        'api_data_dir': Path("./examples/api_files"), # Used if data_source is "API"
        'nc_data_dir': Path("./examples/nc_files"), # Used if data_source is "NC"
        'nc_file_pattern': None, # Optional: pattern to match files. If None, auto-detects "combined_YYYYMMDD_HH.nc"
        
        'output_dir': Path("./examples/hybrid_particle_output"), # General output
        'output_format': "NETCDF", # "CSV" or "NETCDF"
        'checkpoint_dir': Path("./examples/hybrid_particle_output/checkpoints"),
        
        # ERA5 Pressure Levels (must match your CSV data)
        # For NC mode, these are automatically extracted from the files.
        'pressure_levels': np.array([
            100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800,
            825, 850, 875, 900, 925, 950, 975, 1000
        ], dtype=int),
        
        # Particle Initialization Parameters
        'initialization_lat_range': (25, 35),  # Latitude range (min, max)
        'initialization_lon_range': (-105, -95),  # Longitude range (min, max)
        # Pressure levels for initialization (e.g., [1000, 850, 500]). If None, uses a default range.
        'initialization_pressure_levels': [1000,  970,  940,  910,  880,  850,  820,  790,  760,  730,  700, 670,  640,  610,  580,  550,  520,  490,  460,  430,  400,  370, 340,  310,  280,  250,  220,  190, 160, 130],#[np.arange(1000, 200 - 30, -30)],#[1000, 925, 850, 700, 500, 300], 
        'initialization_spacing_km': 10,       # Grid spacing in kilometers for initialization
        
        # Simulation Parameters
        'simulation_start_datetime': "2025-03-21 00:00:00", # Matched to available NC files (combined_20250321_00.nc)
        'simulation_end_datetime': "2025-03-21 02:00:00",   # Used if data_source is "API" or to limit NC duration
        'simulation_start_hour': 0,     # Absolute hour to start the simulation
                                        
        'total_simulation_hours': 2,    # Run for 2 hours
        'data_interval_hours': 1,       # How often new velocity data files are available (NC files are hourly)
        'simulation_step_hours': 0.25,   # Particle position update interval
        'output_interval_hours': 1,     # How often to save particle positions and plots
        'checkpoint_interval_hours': 48, # How often to save checkpoints
        'max_pending_save_tasks': 3,    # Max background saving tasks queued to prevent memory issues (use 1 for near-synchronous)
        
        # Visualization extent (optional, defaults in visualizer if not provided here)
        'plot_lat_range': (-30, 80),
        'plot_lon_range': (-150, -20),
        
        # Thermodynamic Tracking Parameters
        # Mode options: 'NONE', 'SIMPLE_SUBTRACTION', 'POTENTIAL_TEMPERATURE', 'FULL_DECOMPOSITION'
        'thermo_mode': 'POTENTIAL_TEMPERATURE', 
        # Required if thermo_mode is 'FULL_DECOMPOSITION' If you choose to run 'FULL_DECOMPOSITION', just make sure to use data_downloader_method2.py beforehand to generate the background file, and point the 'thermo_bg_file' to that resulting file.
        'thermo_bg_file': Path("./climatology_background.nc"), 
    }
    
    # Initialize the hybrid tracker
    print("Initializing Hybrid Particle Tracker...")
    tracker = HybridParticleTracker(config)
    
    # Check if C++ engine is available
    # Check if C++ engine is available
    if tracker.cpp_engine is not None:
        print("[SUCCESS] C++ engine loaded successfully - high performance mode enabled")
    else:
        print("[WARNING] C++ engine not available - running in Python-only mode (slower)")
        print("  To enable C++ acceleration, run: pip install -e .")
    
    # Run the simulation
    print("\nStarting simulation...")
    try:
        tracker.run_simulation(resume=True)
        print("\n[SUCCESS] Simulation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n[WARNING] Simulation interrupted by user")
        print("  Progress has been saved in checkpoints")
        
    except Exception as e:
        print(f"\n[ERROR] Simulation failed with error: {e}")
        print("  Check the error messages above for details")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
