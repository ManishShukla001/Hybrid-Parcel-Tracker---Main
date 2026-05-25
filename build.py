#!/usr/bin/env python3
"""
Unified build and verification script for the hybrid particle tracker
"""

import subprocess
import sys
from pathlib import Path
import shutil

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("[OK] Success")
        if result.stdout:
            # Only print first few lines of stdout if it's very long
            lines = result.stdout.splitlines()
            if len(lines) > 20:
                print("Output (truncated):")
                for line in lines[:10]:
                    print(f"  {line}")
                print("  ...")
                for line in lines[-10:]:
                    print(f"  {line}")
            else:
                print(f"Output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Failed with return code {e.returncode}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False

def main():
    """Main build and validation process"""
    print("=== Hybrid Particle Tracker Build & Verification ===")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("[FAIL] Python 3.8 or higher is required")
        return 1
    
    print(f"Python version: {sys.version}")
    
    # Clean previous build artifacts
    print("\nCleaning previous build artifacts...")
    folders_to_clean = ["build", "dist", "hybrid_particle_tracker.egg-info"]
    for folder in folders_to_clean:
        path = Path(folder)
        if path.exists():
            try:
                shutil.rmtree(path)
                print(f"[OK] Removed directory: {folder}")
            except Exception as e:
                print(f"[WARN] Could not remove {folder}: {e}")

    # Remove any stray .pyd / .so files in the root to ensure clean build loading
    root_pyd_files = list(Path(".").glob("particle_engine_cpp*.pyd")) + list(Path(".").glob("particle_engine_cpp*.so"))
    for pyd_file in root_pyd_files:
        try:
            pyd_file.unlink()
            print(f"[OK] Removed compiled binary in root: {pyd_file.name}")
        except Exception as e:
            print(f"[WARN] Could not remove {pyd_file.name}: {e}")

    # Uninstall any existing version first
    print("\nUninstalling any existing package version...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "hybrid_particle_tracker", "-y"], 
                  capture_output=True)
    
    # Install dependencies
    if not run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      "Installing Python dependencies"):
        print("[FAIL] Failed to install dependencies")
        return 1
    
    # Build and install the package in editable mode
    if not run_command([sys.executable, "-m", "pip", "install", "-e", "."], 
                      "Building and installing the package"):
        print("[FAIL] Failed to build and install package")
        return 1
    
    # Test the installation
    print("\n=== Running Verification Tests ===")
    
    # Test C++ engine loading
    try:
        import particle_engine_cpp
        print("[OK] C++ engine (particle_engine_cpp) loaded successfully")
        
        # Test basic initialization
        engine = particle_engine_cpp.ParticleEngine(1800.0, 1.0)
        print("[OK] C++ engine initialization successful")
        cpp_available = True
    except Exception as e:
        print(f"[WARN] C++ engine not available or test failed: {e}")
        print("  System will fall back to Python-only mode where applicable.")
        cpp_available = False
        
    # Test Python components
    try:
        from hybrid_particle_tracker.particle_tracker import HybridParticleTracker
        from hybrid_particle_tracker.data_loader import VelocityDataLoader
        from hybrid_particle_tracker.visualization import ParticleVisualizer
        from hybrid_particle_tracker.nc_data_loader import NCDataLoader
        from hybrid_particle_tracker.api_data_handler import APIDataHandler
        from hybrid_particle_tracker.data_downloader_method2 import download_era5_data
        
        print("[OK] All Python core components loaded successfully")
        print("[OK] New NetCDF and API data loader components loaded successfully")
        python_available = True
    except ImportError as e:
        print(f"[FAIL] Failed to import Python components: {e}")
        python_available = False
        return 1
    
    print("\n=== Build & Verification Complete ===")
    if cpp_available and python_available:
        print("[OK] High-performance C++ & Python mode is fully operational!")
    elif python_available:
        print("[WARN] Running in Python-only fallback mode (C++ engine unavailable).")
    
    print("\nTo run the simulation:")
    print("  cd examples")
    print("  python run_simulation.py")
    return 0

if __name__ == "__main__":
    sys.exit(main())