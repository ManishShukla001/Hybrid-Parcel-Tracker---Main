#!/usr/bin/env python3
"""
Test script to verify the build works correctly
"""

import sys
import subprocess


def test_cpp_build():
    """Test if the C++ extension can be built and imported"""
    print("Testing C++ extension build...")
    
    try:
        # Try to build the extension
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", ".", "--verbose"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print("Build failed:")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
        print("Build successful!")
        
        # Try to import the extension
        try:
            import particle_engine_cpp
            print("✓ C++ extension imported successfully")
            
            # Test basic functionality
            engine = particle_engine_cpp.ParticleEngine(1800.0, 1.0)
            print("✓ C++ engine created successfully")
            return True
            
        except ImportError as e:
            print(f"✗ Failed to import C++ extension: {e}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Build timed out")
        return False
    except Exception as e:
        print(f"✗ Build failed with exception: {e}")
        return False


def test_python_components():
    """Test if Python components work"""
    print("\nTesting Python components...")
    
    try:
        sys.path.insert(0, "src/python")
        
        from particle_tracker import HybridParticleTracker
        from data_loader import VelocityDataLoader
        from visualization import ParticleVisualizer
        
        print("✓ All Python components imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import Python components: {e}")
        return False


def main():
    """Main test function"""
    print("=== Hybrid Particle Tracker Build Test ===")
    
    # Test Python components first (faster)
    if not test_python_components():
        print("\n✗ Python component test failed")
        return 1
    
    # Test C++ build
    if not test_cpp_build():
        print("\n⚠ C++ build failed - will run in Python-only mode")
        print("This is slower but still functional")
    else:
        print("\n✓ C++ extension build successful - high performance mode available")
    
    print("\n=== Test Complete ===")
    print("You can now run the simulation with:")
    print("  cd examples")
    print("  python run_simulation.py")
    
    return 0


if __name__ == "__main__":
    exit(main())