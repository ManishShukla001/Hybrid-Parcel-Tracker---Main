#!/usr/bin/env python3
"""
Direct execution script for the hybrid particle tracker
This script imports modules directly without requiring package installation
"""

import numpy as np
from pathlib import Path
import sys

# Add src/python to path for direct imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))

# Import modules directly
from particle_tracker import HybridParticleTracker


def main():
    """Run the particle tracking simulation"""
    
    # Configuration - adjust these paths to match your setup
    config = {
        # Data paths (adjust to your actual paths)
        'csv_base_dir': Path("/scratch/maheswaram.civil.iith/ERA/era5_csv_output"),
        'output_dir': Path("/scratch/maheswaram.civil.iith/ERA/hybrid_particle_output"),
        'checkpoint_dir': Path("/scratch/maheswaram.civil.iith/ERA/hybrid_particle_output/checkpoints"),
        
        # ERA5 Pressure Levels (must match your CSV data)
        'pressure_levels': np.array([
            200, 250, 300, 350, 400, 450, 500,
            550, 600, 650, 700, 750, 775, 800, 825,
            850, 875, 900, 925, 950, 975, 1000
        ], dtype=int),
        
        # Simulation Parameters
        'total_simulation_hours': 479,  # 20 days - 1 hour
        'data_interval_hours': 1,       # How often new velocity data files are available
        'simulation_step_hours': 0.5,   # Particle position update interval
        'output_interval_hours': 1,     # How often to save particle positions and plots
        'checkpoint_interval_hours': 6, # How often to save checkpoints
    }
    
    # Initialize the hybrid tracker
    print("Initializing Hybrid Particle Tracker...")
    tracker = HybridParticleTracker(config)
    
    # Check if C++ engine is available
    if tracker.cpp_engine is not None:
        print("✓ C++ engine loaded successfully - high performance mode enabled")
    else:
        print("⚠ C++ engine not available - running in Python-only mode (slower)")
        print("  The C++ extension was compiled but may not be properly linked")
    
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
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())