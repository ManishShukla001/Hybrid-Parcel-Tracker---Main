#!/usr/bin/env python3
"""
Performance verification script
"""
import numpy as np
from pathlib import Path
import sys
import time
import os

# Import the installed package or fallback
try:
    from hybrid_particle_tracker.particle_tracker import HybridParticleTracker
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))
    from particle_tracker import HybridParticleTracker

def main():
    config = {
        'data_source': "NC",
        'execution_mode': "parallel",
        'nc_data_dir': Path("./examples/nc_files"),
        'nc_file_pattern': None,
        'output_dir': Path("./examples/verification_output"),
        'output_format': "NETCDF",
        'checkpoint_dir': Path("./examples/verification_output/checkpoints"),
        
        'pressure_levels': np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 825, 850, 875, 900, 925, 950, 975, 1000], dtype=int),
        
        'initialization_lat_range': (-15, 65),
        'initialization_lon_range': (-135, -45),
        'initialization_pressure_levels': [1000, 850, 500],
        'initialization_spacing_km': 150,
        
        'simulation_start_datetime': "2025-03-21 00:00:00",
        'simulation_end_datetime': "2025-03-21 00:05:00", # very short run
        'simulation_start_hour': 0,
        'total_simulation_hours': 0.1, # 6 minutes
        'data_interval_hours': 1,
        'simulation_step_hours': 0.05, # 3 minutes per step
        'output_interval_hours': 0.05, # Output every step
        'checkpoint_interval_hours': 1,
        
        'plot_lat_range': (-30, 80),
        'plot_lon_range': (-150, -20),
    }
    
    # Clean output dir
    import shutil
    if config['output_dir'].exists():
        shutil.rmtree(config['output_dir'])
    
    print("Initializing Tracker...")
    tracker = HybridParticleTracker(config)
    
    if tracker.cpp_engine is not None:
        print("[SUCCESS] C++ engine loaded")
    else:
        print("[WARNING] C++ engine NOT loaded")

    print("Starting verification simulation...")
    start_time = time.time()
    tracker.run_simulation(resume=False)
    end_time = time.time()
    
    print(f"\nSimulation took {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
