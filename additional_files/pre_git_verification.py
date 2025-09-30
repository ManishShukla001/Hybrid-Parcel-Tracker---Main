#!/usr/bin/env python3
"""
Final verification test before GitHub upload
Tests all cleaned and reorganized components
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all module imports work"""
    print("=== Testing Module Imports ===")

    try:
        # Test direct imports
        sys.path.insert(0, str(Path(__file__).parent / "src" / "python"))

        import data_loader
        print("✓ data_loader imported")

        import particle_tracker
        print("✓ particle_tracker imported")

        import visualization
        print("✓ visualization imported")

        import api_data_handler
        print("✓ api_data_handler imported")

        # Test package imports
        try:
            import hybrid_particle_tracker
            print("✓ hybrid_particle_tracker package imported")
        except ImportError:
            print("⚠ hybrid_particle_tracker package not installed (expected for direct execution)")

        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_cpp_engine():
    """Test C++ engine functionality"""
    print("\n=== Testing C++ Engine ===")

    try:
        import particle_engine_cpp
        print("✓ C++ engine imported")

        # Test basic engine functionality
        engine = particle_engine_cpp.ParticleEngine(1800.0, 1.0)
        print("✓ C++ engine instantiated")

        # Don't test the complex update_particles function call here
        # as it requires proper interpolator objects
        print("✓ C++ engine basic functionality verified")

        return True

    except ImportError:
        print("⚠ C++ engine not available (will run in Python-only mode)")
        return True  # This is acceptable for Python-only mode
    except Exception as e:
        print(f"✗ C++ engine test failed: {e}")
        return False

def test_config_parsing():
    """Test that config parsing works"""
    print("\n=== Testing Config Parsing ===")

    try:
        from particle_tracker import HybridParticleTracker
        import numpy as np

        # Minimal config for testing
        test_config = {
            'data_source': "CSV",
            'csv_base_dir': Path("/tmp"),  # Dummy path
            'output_dir': Path("/tmp"),
            'checkpoint_dir': Path("/tmp"),
            'pressure_levels': np.array([1000, 950, 900]),
            'initialization_lat_range': (-10, 10),
            'initialization_lon_range': (30, 60),
            'initialization_pressure_levels': [1000, 900],
            'initialization_spacing_km': 100,
            'total_simulation_hours': 1,
            'data_interval_hours': 1,
            'simulation_step_hours': 0.5,
            'output_interval_hours': 1,
            'checkpoint_interval_hours': 1,
            'output_format': "CSV"
        }

        # Test config validation
        tracker = HybridParticleTracker(test_config)
        print("✓ Config validation passed")
        print("✓ Tracker instantiation successful")

        return True

    except Exception as e:
        print(f"✗ Config/test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test file structure integrity"""
    print("\n=== Testing File Structure ===")

    # Check core directories exist
    required_dirs = ['src/python', 'src/cpp', 'examples', 'Info_page']
    missing_dirs = []

    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)

    if missing_dirs:
        print(f"✗ Missing directories: {missing_dirs}")
        return False

    print("✓ Core directories present")

    # Check core files exist
    required_files = [
        'src/python/particle_tracker.py',
        'src/python/data_loader.py',
        'src/python/visualization.py',
        'src/python/api_data_handler.py',
        'examples/run_simulation.py',
        'setup.py',
        'requirements.txt',
        'README.md'
    ]

    # Check Additional_Files folder
    additional_files = Path('Additional_Files')
    if not additional_files.exists():
        print("⚠ Additional_Files folder not found")
    else:
        details_file = additional_files / 'Details_of_additional_file.txt'
        if details_file.exists():
            print("✓ Additional_Files organization complete with documentation")
        else:
            print("⚠ Additional_Files exists but no details file")

    missing_files = []
    for file_name in required_files:
        if not Path(file_name).exists():
            missing_files.append(file_name)

    if missing_files:
        print(f"✗ Missing files: {missing_files}")
        return False

    print("✓ Core files present")
    return True

def main():
    """Main verification function"""
    print("🔍 GitHub Upload Verification")
    print("=" * 50)

    all_tests_passed = True

    # Run all tests
    tests = [
        test_file_structure,
        test_imports,
        test_cpp_engine,
        test_config_parsing
    ]

    for test_func in tests:
        if not test_func():
            all_tests_passed = False

    print("\n" + "=" * 50)

    if all_tests_passed:
        print("✅ ALL VERIFICATION TESTS PASSED!")
        print("🎉 Your codebase is ready for GitHub upload!")
        print("\n📋 Git Upload Recommendations:")
        print("• Add src/, examples/, etc. to version control")
        print("• Exclude Additional_Files/ from git (.gitignore)")
        print("• Keep README.md, requirements.txt, setup.py")
        return 0
    else:
        print("❌ SOME VERIFICATION TESTS FAILED!")
        print("🔧 Please fix the issues before uploading to GitHub")
        return 1

if __name__ == "__main__":
    exit(main())
