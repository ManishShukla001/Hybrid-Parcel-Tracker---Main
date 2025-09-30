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

    config = {
        # Data Source: "CSV" or "API"
        'data_source': "API",  # Choose "API" to use Copernicus API
        'csv_base_dir': Path("D:/Manish/ERA_Data/Data/csv_files"),  # Used if data_source is "CSV"
        'api_data_dir': Path("./api_files"),  # Used if data_source is "API"

        # Output settings
        'output_dir': Path("./hybrid_particle_output"),
        'output_format': "NETCDF",  # "CSV" or "NETCDF"
        'checkpoint_dir': Path("./hybrid_particle_output/checkpoints"),

        # ERA5 Pressure Levels (must match your CSV data)
        'pressure_levels': np.array([
            100, 150, 200, 250, 300, 350, 400, 450, 500,
            550, 600, 650, 700, 750, 775, 800, 825,
            850, 875, 900, 925, 950, 975, 1000
        ], dtype=int),

        # Particle Initialization Parameters
        'initialization_lat_range': (-15, 30),  # Latitude range (min, max)
        'initialization_lon_range': (30, 120),  # Longitude range (min, max)
        'initialization_pressure_levels': [1000, 925, 850, 700, 500, 300],
        'initialization_spacing_km': 10,         # Grid spacing in kilometers

        # Simulation Timing Parameters
        'simulation_start_datetime': "2023-01-01 00:00:00",  # Used if data_source is "API" (YYYY-MM-DD HH:MM:SS UTC)
        'simulation_end_datetime': "2023-01-03 00:00:00",    # Used if data_source is "API" (YYYY-MM-DD HH:MM:SS UTC)
        'simulation_start_hour': 0,        # Absolute hour to start (CSV mode: file index, API mode: offset)
        'total_simulation_hours': 48,       # Duration in hours
        'data_interval_hours': 1,           # How often velocity data files are available
        'simulation_step_hours': 0.5,       # Particle position update interval
        'output_interval_hours': 1,         # How often to save particles and plots
        'checkpoint_interval_hours': 6,     # How often to save checkpoints

        # Optional visualization extent (defaults will be used if not provided)
        'plot_lat_range': (-20, 40),
        'plot_lon_range': (20, 130),
    }
    
    # Initialize the hybrid tracker
    print("Initializing Hybrid Particle Tracker...")
    tracker = HybridParticleTracker(config)
    
    # Check if C++ engine is available
    if tracker.cpp_engine is not None:
        print("✓ C++ engine loaded successfully - high performance mode enabled")
    else:
        print("⚠ C++ engine not available - running in Python-only mode (slower)")
        print("  To enable C++ acceleration, run: pip install -e .")
    
    # Run the simulation
    print("\nStarting simulation...")
    try:
        tracker.run_simulation(resume=True)
        print("\n✓ Simulation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠ Simulation interrupted by user")
        print("  Progress has been saved in checkpoints")
        
    except Exception as e:
        print(f"\n✗ Simulation failed with error: {e}")
        print("  Check the error messages above for details")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

