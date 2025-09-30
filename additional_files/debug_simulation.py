#!/usr/bin/env python3
"""
Debug version of the simulation to identify why particles aren't moving
"""

import numpy as np
from pathlib import Path
import sys

# Add src/python to path for direct imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src" / "python"))

# Import modules directly
from particle_tracker import HybridParticleTracker


def debug_velocity_data(tracker, hour_index=1):
    """Debug velocity data loading"""
    print(f"\n=== Debugging Velocity Data for Hour {hour_index} ===")
    
    try:
        velocity_data = tracker.load_velocity_data(hour_index)
        if velocity_data is None:
            print("✗ Failed to load velocity data")
            return False
            
        u_data, v_data, w_data = velocity_data
        
        print(f"✓ Loaded velocity data successfully")
        print(f"  U data shape: lats={len(u_data[0])}, lons={len(u_data[1])}, pressures={len(u_data[2])}, values={len(u_data[3])}")
        print(f"  Lat range: {u_data[0].min():.2f} to {u_data[0].max():.2f}")
        print(f"  Lon range: {u_data[1].min():.2f} to {u_data[1].max():.2f}")
        print(f"  Pressure range: {u_data[2].min():.0f} to {u_data[2].max():.0f}")
        
        # Check for non-zero values
        u_nonzero = np.count_nonzero(u_data[3])
        v_nonzero = np.count_nonzero(v_data[3])
        w_nonzero = np.count_nonzero(w_data[3])
        
        print(f"  Non-zero values: U={u_nonzero}/{len(u_data[3])}, V={v_nonzero}/{len(v_data[3])}, W={w_nonzero}/{len(w_data[3])}")
        
        # Check value ranges
        print(f"  U range: {u_data[3].min():.6f} to {u_data[3].max():.6f}")
        print(f"  V range: {v_data[3].min():.6f} to {v_data[3].max():.6f}")
        print(f"  W range: {w_data[3].min():.6f} to {w_data[3].max():.6f}")
        
        if u_nonzero == 0 and v_nonzero == 0 and w_nonzero == 0:
            print("⚠ WARNING: All velocity values are zero!")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error loading velocity data: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_interpolators(tracker, velocity_data):
    """Debug interpolator creation and testing"""
    print(f"\n=== Debugging Interpolators ===")
    
    try:
        interpolators = tracker.create_interpolators(velocity_data)
        print("✓ Created interpolators successfully")
        
        # Test interpolation at a sample point
        test_lat, test_lon, test_pressure = -6.35, 85.25, 760.0
        print(f"Testing interpolation at ({test_lat}, {test_lon}, {test_pressure})")
        
        # Test C++ interpolators if available
        if 'u_cpp' in interpolators:
            try:
                u_val = interpolators['u_cpp'].interpolate(test_lat, test_lon, test_pressure)
                v_val = interpolators['v_cpp'].interpolate(test_lat, test_lon, test_pressure)
                w_val = interpolators['w_cpp'].interpolate(test_lat, test_lon, test_pressure)
                
                print(f"  C++ interpolation: U={u_val:.6f}, V={v_val:.6f}, W={w_val:.6f}")
                
                if u_val == 0 and v_val == 0 and w_val == 0:
                    print("⚠ WARNING: C++ interpolation returned all zeros!")
                else:
                    print("✓ C++ interpolation working correctly")
                    
            except Exception as e:
                print(f"✗ C++ interpolation failed: {e}")
        else:
            print("⚠ C++ interpolators not available")
        # Always test scipy interpolators (they should be available as fallback)
        try:
            point = np.array([test_lat, test_lon, test_pressure])
            u_interp = interpolators.get('u_scipy', interpolators['u'])
            v_interp = interpolators.get('v_scipy', interpolators['v'])
            w_interp = interpolators.get('w_scipy', interpolators['w'])
            
            u_val = u_interp(point)
            v_val = v_interp(point)
            w_val = w_interp(point)
            
            # Handle array returns
            if hasattr(u_val, '__len__') and len(u_val) == 1:
                u_val, v_val, w_val = u_val[0], v_val[0], w_val[0]
            elif hasattr(u_val, '__len__'):
                u_val, v_val, w_val = float(u_val), float(v_val), float(w_val)
            
            print(f"  Scipy interpolation: U={u_val:.6f}, V={v_val:.6f}, W={w_val:.6f}")
            
            if u_val == 0 and v_val == 0 and w_val == 0:
                print("⚠ WARNING: Scipy interpolation returned all zeros!")
            else:
                print("✓ Scipy interpolation working correctly")
                
        except Exception as e:
            print(f"✗ Scipy interpolation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating interpolators: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_particle_update(tracker):
    """Debug a single particle update"""
    print(f"\n=== Debugging Particle Update ===")
    
    try:
        # Create a single test particle
        test_particles = np.array([[1.0, -6.35, 85.25, 760.0]])
        print(f"Initial particle: {test_particles[0]}")
        
        # Load velocity data
        curr_data = tracker.load_velocity_data(1)
        next_data = tracker.load_velocity_data(2)
        
        if curr_data is None or next_data is None:
            print("✗ Failed to load velocity data for testing")
            return False
        
        # Create interpolators
        interp_curr = tracker.create_interpolators(curr_data)
        interp_next = tracker.create_interpolators(next_data)
        
        # Test update with alpha = 0.5
        alpha = 0.5
        print(f"Testing update with alpha = {alpha}")
        
        updated_particles = tracker.update_particles(test_particles, alpha, interp_curr, interp_next)
        print(f"Updated particle: {updated_particles[0]}")
        
        # Check if particle moved
        if np.allclose(test_particles[0, 1:], updated_particles[0, 1:]):
            print("⚠ WARNING: Particle did not move!")
            return False
        else:
            print("✓ Particle moved successfully")
            delta = updated_particles[0, 1:] - test_particles[0, 1:]
            print(f"  Delta: lat={delta[0]:.8f}, lon={delta[1]:.8f}, pressure={delta[2]:.8f}")
            return True
        
    except Exception as e:
        print(f"✗ Error in particle update: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Debug main function"""
    print("=== Particle Tracker Debug Session ===")
    
    # Minimal config for debugging
    config = {
        'csv_base_dir': Path("D:/Manish/ERA_Data/Data/era5_csv_output"),  # Adjust this path
        'output_dir': Path("./debug_output"),
        'checkpoint_dir': Path("./debug_output/checkpoints"),
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
    
    # Initialize tracker
    print("Initializing tracker...")
    tracker = HybridParticleTracker(config)
    
    if tracker.cpp_engine is not None:
        print("✓ C++ engine available")
    else:
        print("⚠ C++ engine not available - using Python fallback")
    
    # Debug steps
    success = True
    
    # 1. Debug velocity data loading
    if not debug_velocity_data(tracker, 1):
        success = False
    
    # 2. Debug interpolators
    if success:
        velocity_data = tracker.load_velocity_data(1)
        if velocity_data and not debug_interpolators(tracker, velocity_data):
            success = False
    
    # 3. Debug particle update
    if success:
        if not debug_particle_update(tracker):
            success = False
    
    print(f"\n=== Debug Results ===")
    if success:
        print("✓ All debug tests passed - particles should move correctly")
    else:
        print("✗ Debug tests failed - check the issues above")
        print("\nCommon issues:")
        print("1. Check if CSV files exist and contain non-zero velocity data")
        print("2. Verify coordinate ranges match between particles and velocity data")
        print("3. Check if interpolation is working correctly")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())