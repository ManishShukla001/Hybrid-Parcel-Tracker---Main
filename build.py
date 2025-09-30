#!/usr/bin/env python3
"""
Build script for the hybrid particle tracker
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
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed with return code {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main():
    """Main build process"""
    print("=== Hybrid Particle Tracker Build Script ===")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        return 1
    
    print(f"✓ Python {sys.version}")
    
    # Install dependencies
    if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      "Installing Python dependencies"):
        print("Failed to install dependencies")
        return 1
    
    # Build and install the package
    if not run_command([sys.executable, "-m", "pip", "install", "-e", "."], 
                      "Building and installing the package"):
        print("Failed to build package")
        return 1
    
    # Test the installation
    print("\nTesting installation...")
    try:
        import particle_engine_cpp
        print("✓ C++ engine loaded successfully")
        
        # Test basic functionality
        engine = particle_engine_cpp.ParticleEngine(1800.0, 1.0)  # 30 min steps, 1 hour data interval
        print("✓ C++ engine initialization successful")
        
    except ImportError:
        print("⚠ C++ engine not available - will run in Python-only mode")
        print("  This is slower but still functional")
    except Exception as e:
        print(f"⚠ C++ engine test failed: {e}")
    
    # Test Python components
    try:
        from src.python.particle_tracker import HybridParticleTracker
        from src.python.data_loader import VelocityDataLoader
        from src.python.visualization import ParticleVisualizer
        print("✓ Python components loaded successfully")
    except ImportError as e:
        print(f"✗ Failed to import Python components: {e}")
        return 1
    
    print("\n=== Build Complete ===")
    print("To run the simulation:")
    print("  cd examples")
    print("  python run_simulation.py")
    print("\nMake sure to adjust the paths in run_simulation.py to match your data location.")
    
    return 0


if __name__ == "__main__":
    exit(main())