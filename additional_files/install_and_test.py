#!/usr/bin/env python3
"""
Installation and test script for the hybrid particle tracker
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✓ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with return code {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main():
    """Main installation and test process"""
    print("=== Hybrid Particle Tracker Installation ===")
    
    # Uninstall any existing version
    print("\nUninstalling any existing version...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "hybrid_particle_tracker", "-y"], 
                  capture_output=True)
    
    # Install dependencies
    if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      "Installing dependencies"):
        return 1
    
    # Build and install the package
    if not run_command([sys.executable, "-m", "pip", "install", "-e", "."], 
                      "Building and installing package"):
        return 1
    
    # Test the installation
    print("\n=== Testing Installation ===")
    
    # Test C++ engine
    try:
        import particle_engine_cpp
        print("✓ C++ engine imported successfully")
        
        engine = particle_engine_cpp.ParticleEngine(1800.0, 1.0)
        print("✓ C++ engine created successfully")
        cpp_available = True
        
    except ImportError:
        print("⚠ C++ engine not available - will run in Python-only mode")
        cpp_available = False
    except Exception as e:
        print(f"⚠ C++ engine test failed: {e}")
        cpp_available = False
    
    # Test Python package
    try:
        from hybrid_particle_tracker.particle_tracker import HybridParticleTracker
        from hybrid_particle_tracker.data_loader import VelocityDataLoader
        from hybrid_particle_tracker.visualization import ParticleVisualizer
        print("✓ Python package imported successfully")
        
    except ImportError as e:
        print(f"✗ Failed to import Python package: {e}")
        return 1
    
    print("\n=== Installation Complete ===")
    
    if cpp_available:
        print("✓ High-performance C++ mode available")
    else:
        print("⚠ Running in Python-only mode (slower but functional)")
    
    print("\nTo run the simulation:")
    print("  cd examples")
    print("  python run_simulation.py")
    print("\nMake sure to adjust the paths in run_simulation.py to match your data location.")
    
    return 0


if __name__ == "__main__":
    exit(main())