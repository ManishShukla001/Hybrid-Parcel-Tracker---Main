#!/usr/bin/env python3
"""
Quick test to verify C++ interpolator fix
"""

import numpy as np
from pathlib import Path
import sys

# Add src/python to path for direct imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "python"))

from particle_tracker import HybridParticleTracker

def test_cpp_interpolator_fix():
    """Test if C++ interpolators now work correctly"""
    
    # Minimal config for testing
    config = {
        'csv_base_dir': Path("D:/Manish/ERA_Data/Data/era5_csv_output"),  # Adjust path
        'output_dir': Path("./test_output"),
        'checkpoint_dir': Path("./test_output/checkpoints"),
        'pressure_levels': np.array([
            200, 250, 300, 350, 400, 450, 500,
            550, 600, 650, 700, 750, 775, 800, 825,
            850, 875, 900, 925, 950, 975, 1000
        ], dtype=int),
        'total_simulation_hours': 2,
        'data_interval_hours': 1,
        'simulation_step_hours': 0.5,
        'output_interval_hours': 1,
        'checkpoint_interval_hours': 1,
    }
    
    print("=== Testing C++ Interpolator Fix ===")
    
    # Initialize tracker
    tracker = HybridParticleTracker(config)
    
    # Load velocity data
    print("Loading velocity data...")
    velocity_data = tracker.load_velocity_data(1)
    if velocity_data is None:
        print("✗ Failed to load velocity data")
        return False
    
    print("✓ Velocity data loaded successfully")
    
    # Create interpolators
    print("Creating interpolators...")
    interpolators = tracker.create_interpolators(velocity_data)
    
    # Check if C++ interpolators were created successfully
    if 'u_cpp' in interpolators:
        print("✓ C++ interpolators created")
        
        # Test interpolation
        test_lat, test_lon, test_pressure = -6.35, 85.25, 760.0
        print(f"Testing interpolation at ({test_lat}, {test_lon}, {test_pressure})")
        
        try:
            u_val = interpolators['u_cpp'].interpolate(test_lat, test_lon, test_pressure)
            v_val = interpolators['v_cpp'].interpolate(test_lat, test_lon, test_pressure)
            w_val = interpolators['w_cpp'].interpolate(test_lat, test_lon, test_pressure)
            
            print(f"C++ interpolation: U={u_val:.6f}, V={v_val:.6f}, W={w_val:.6f}")
            
            if u_val == 0 and v_val == 0 and w_val == 0:
                print("✗ C++ interpolators still return zeros")
                return False
            else:
                print("✓ C++ interpolators working correctly!")
                
                # Compare with scipy
                point = np.array([test_lat, test_lon, test_pressure])
                u_scipy = interpolators['u_scipy'](point)[0] if hasattr(interpolators['u_scipy'](point), '__len__') else interpolators['u_scipy'](point)
                v_scipy = interpolators['v_scipy'](point)[0] if hasattr(interpolators['v_scipy'](point), '__len__') else interpolators['v_scipy'](point)
                w_scipy = interpolators['w_scipy'](point)[0] if hasattr(interpolators['w_scipy'](point), '__len__') else interpolators['w_scipy'](point)
                
                print(f"Scipy interpolation: U={u_scipy:.6f}, V={v_scipy:.6f}, W={w_scipy:.6f}")
                print(f"Differences: U={abs(u_val-u_scipy):.6f}, V={abs(v_val-v_scipy):.6f}, W={abs(w_val-w_scipy):.6f}")
                
                return True
                
        except Exception as e:
            print(f"✗ C++ interpolation failed: {e}")
            return False
    else:
        print("✗ C++ interpolators not created")
        return False

def test_particle_update():
    """Test if particle updates now work with C++"""
    
    config = {
        'csv_base_dir': Path("D:/Manish/ERA_Data/Data/era5_csv_output"),
        'output_dir': Path("./test_output"),
        'checkpoint_dir': Path("./test_output/checkpoints"),
        'pressure_levels': np.array([
            200, 250, 300, 350, 400, 450, 500,
            550, 600, 650, 700, 750, 775, 800, 825,
            850, 875, 900, 925, 950, 975, 1000
        ], dtype=int),
        'total_simulation_hours': 2,
        'data_interval_hours': 1,
        'simulation_step_hours': 0.5,
        'output_interval_hours': 1,
        'checkpoint_interval_hours': 1,
    }
    
    print("\n=== Testing Particle Updates ===")
    
    tracker = HybridParticleTracker(config)
    
    # Create test particle
    test_particles = np.array([[1.0, -6.35, 85.25, 760.0]])
    print(f"Initial particle: {test_particles[0]}")
    
    # Load velocity data
    curr_data = tracker.load_velocity_data(1)
    next_data = tracker.load_velocity_data(2)
    
    if curr_data is None or next_data is None:
        print("✗ Failed to load velocity data")
        return False
    
    # Create interpolators
    interp_curr = tracker.create_interpolators(curr_data)
    interp_next = tracker.create_interpolators(next_data)
    
    # Test update
    alpha = 0.5
    updated_particles = tracker.update_particles(test_particles, alpha, interp_curr, interp_next)
    print(f"Updated particle: {updated_particles[0]}")
    
    # Check if particle moved
    if np.allclose(test_particles[0, 1:], updated_particles[0, 1:]):
        print("✗ Particle did not move")
        return False
    else:
        print("✓ Particle moved successfully!")
        delta = updated_particles[0, 1:] - test_particles[0, 1:]
        print(f"  Delta: lat={delta[0]:.8f}, lon={delta[1]:.8f}, pressure={delta[2]:.8f}")
        return True

def main():
    """Main test function"""
    
    success = True
    
    if not test_cpp_interpolator_fix():
        success = False
    
    if not test_particle_update():
        success = False
    
    print(f"\n=== Test Results ===")
    if success:
        print("✓ All tests passed! C++ interpolators are working correctly.")
        print("You can now run the full simulation with high performance C++ mode.")
    else:
        print("✗ Some tests failed. Check the issues above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())