#!/usr/bin/env python3
"""
Test script to verify the direct import approach works
"""

import sys
from pathlib import Path

# Add src/python to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "python"))

def test_imports():
    """Test if all modules can be imported directly"""
    print("Testing direct imports...")
    
    try:
        from particle_tracker import HybridParticleTracker
        print("✓ particle_tracker imported successfully")
        
        from data_loader import VelocityDataLoader
        print("✓ data_loader imported successfully")
        
        from visualization import ParticleVisualizer
        print("✓ visualization imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_cpp_engine():
    """Test if C++ engine is available"""
    print("\nTesting C++ engine...")
    
    try:
        import particle_engine_cpp
        print("✓ C++ engine imported successfully")
        
        engine = particle_engine_cpp.ParticleEngine(1800.0, 1.0)
        print("✓ C++ engine created successfully")
        
        return True
        
    except ImportError:
        print("⚠ C++ engine not available")
        return False
    except Exception as e:
        print(f"⚠ C++ engine test failed: {e}")
        return False

def test_tracker_creation():
    """Test if the tracker can be created"""
    print("\nTesting tracker creation...")
    
    try:
        from particle_tracker import HybridParticleTracker
        import numpy as np
        
        # Minimal config for testing
        config = {
            'csv_base_dir': Path("/tmp"),  # Dummy path
            'output_dir': Path("/tmp"),
            'checkpoint_dir': Path("/tmp"),
            'pressure_levels': np.array([1000, 850, 500]),
            'total_simulation_hours': 1,
            'data_interval_hours': 1,
            'simulation_step_hours': 0.5,
            'output_interval_hours': 1,
            'checkpoint_interval_hours': 1,
        }
        
        tracker = HybridParticleTracker(config)
        print("✓ Tracker created successfully")
        
        if tracker.cpp_engine is not None:
            print("✓ C++ engine is available in tracker")
        else:
            print("⚠ C++ engine not available in tracker")
        
        return True
        
    except Exception as e:
        print(f"✗ Tracker creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=== Direct Import Test ===")
    
    success = True
    
    if not test_imports():
        success = False
    
    if not test_cpp_engine():
        print("  (This is OK - will run in Python-only mode)")
    
    if not test_tracker_creation():
        success = False
    
    print("\n=== Test Results ===")
    if success:
        print("✓ All tests passed!")
        print("\nYou can now run the simulation with:")
        print("  cd examples")
        print("  python run_simulation_direct.py")
    else:
        print("✗ Some tests failed")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())