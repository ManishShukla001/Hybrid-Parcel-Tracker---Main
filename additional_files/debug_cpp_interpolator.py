#!/usr/bin/env python3
"""
Debug script specifically for C++ interpolator issues
"""

import numpy as np
from pathlib import Path
import sys

# Add src/python to path for direct imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "python"))

from data_loader import VelocityDataLoader

def debug_cpp_interpolator():
    """Debug C++ interpolator creation and data passing"""
    
    # Load some test data
    csv_base_dir = Path("D:/Manish/ERA_Data/Data/era5_csv_output")  # Adjust path
    pressure_levels = np.array([200, 250, 300, 350, 400, 450, 500,
                               550, 600, 650, 700, 750, 775, 800, 825,
                               850, 875, 900, 925, 950, 975, 1000], dtype=int)
    
    loader = VelocityDataLoader(csv_base_dir, pressure_levels)
    
    print("=== Loading velocity data ===")
    u_data = loader.load_velocity_field('u', 1)
    if u_data is None:
        print("Failed to load velocity data")
        return
    
    lats, lons, pressures, values = u_data
    print(f"Data loaded: {len(lats)} lats, {len(lons)} lons, {len(pressures)} pressures")
    print(f"Lat range: {lats.min():.2f} to {lats.max():.2f}")
    print(f"Lon range: {lons.min():.2f} to {lons.max():.2f}")
    print(f"Pressure range: {pressures.min():.0f} to {pressures.max():.0f}")
    print(f"Values shape: {len(values)}, expected: {len(lats) * len(lons) * len(pressures)}")
    print(f"Values range: {values.min():.6f} to {values.max():.6f}")
    print(f"Non-zero values: {np.count_nonzero(values)}/{len(values)}")
    
    # Check coordinate ordering
    print(f"\nFirst few lats: {lats[:5]}")
    print(f"Last few lats: {lats[-5:]}")
    print(f"Lat sorted ascending: {np.array_equal(lats, np.sort(lats))}")
    print(f"Lat sorted descending: {np.array_equal(lats, np.sort(lats)[::-1])}")
    
    print(f"\nFirst few lons: {lons[:5]}")
    print(f"Last few lons: {lons[-5:]}")
    print(f"Lon sorted ascending: {np.array_equal(lons, np.sort(lons))}")
    
    print(f"\nFirst few pressures: {pressures[:5]}")
    print(f"Last few pressures: {pressures[-5:]}")
    print(f"Pressure sorted ascending: {np.array_equal(pressures, np.sort(pressures))}")
    print(f"Pressure sorted descending: {np.array_equal(pressures, np.sort(pressures)[::-1])}")
    
    # Test C++ interpolator
    print(f"\n=== Testing C++ Interpolator ===")
    try:
        import particle_engine_cpp
        
        # Create interpolator
        print("Creating C++ interpolator...")
        interpolator = particle_engine_cpp.RegularGrid3DInterpolator(
            lats.tolist(), lons.tolist(), pressures.tolist(), values.tolist()
        )
        print("✓ C++ interpolator created successfully")
        
        # Get bounds
        bounds = interpolator.get_bounds()
        print(f"C++ interpolator bounds: lat=[{bounds[0]:.2f}, {bounds[1]:.2f}], "
              f"lon=[{bounds[2]:.2f}, {bounds[3]:.2f}], pressure=[{bounds[4]:.0f}, {bounds[5]:.0f}]")
        
        # Test interpolation at various points
        test_points = [
            # Center point
            (lats[len(lats)//2], lons[len(lons)//2], pressures[len(pressures)//2]),
            # Corner points
            (lats[0], lons[0], pressures[0]),
            (lats[-1], lons[-1], pressures[-1]),
            # Test point from debug
            (-6.35, 85.25, 760.0),
            # Points within bounds
            (0.0, 75.0, 850.0),
            (15.0, 60.0, 500.0)
        ]
        
        print(f"\nTesting interpolation at various points:")
        for i, (lat, lon, pressure) in enumerate(test_points):
            try:
                value = interpolator.interpolate(lat, lon, pressure)
                in_bounds = (bounds[0] <= lat <= bounds[1] and 
                           bounds[2] <= lon <= bounds[3] and 
                           bounds[4] <= pressure <= bounds[5])
                print(f"  Point {i+1}: ({lat:6.2f}, {lon:6.2f}, {pressure:6.0f}) -> {value:10.6f} "
                      f"{'(in bounds)' if in_bounds else '(OUT OF BOUNDS)'}")
            except Exception as e:
                print(f"  Point {i+1}: ({lat:6.2f}, {lon:6.2f}, {pressure:6.0f}) -> ERROR: {e}")
        
        # Test with scipy for comparison
        print(f"\n=== Comparing with Scipy ===")
        from scipy.interpolate import RegularGridInterpolator
        
        # Reshape for scipy
        nlat, nlon, npres = len(lats), len(lons), len(pressures)
        grid = values.reshape(nlat, nlon, npres)
        scipy_interp = RegularGridInterpolator(
            (lats, lons, pressures), grid, bounds_error=False, fill_value=0.0
        )
        
        print("Testing same points with scipy:")
        for i, (lat, lon, pressure) in enumerate(test_points):
            try:
                point = np.array([lat, lon, pressure])
                scipy_value = scipy_interp(point)
                if hasattr(scipy_value, '__len__'):
                    scipy_value = scipy_value[0]
                cpp_value = interpolator.interpolate(lat, lon, pressure)
                print(f"  Point {i+1}: Scipy={scipy_value:10.6f}, C++={cpp_value:10.6f}, "
                      f"Diff={abs(scipy_value - cpp_value):10.6f}")
            except Exception as e:
                print(f"  Point {i+1}: ERROR: {e}")
        
    except ImportError:
        print("C++ engine not available")
    except Exception as e:
        print(f"Error testing C++ interpolator: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    debug_cpp_interpolator()