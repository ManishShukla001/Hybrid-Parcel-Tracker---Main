from setuptools import setup, Extension, find_packages
import sys
import os
import shutil
from pathlib import Path

# --- 0. Automatic Clean Step ---
# If setup.py is run to build, install, or package, clean up previous stale artifacts
# to prevent linking errors or cached object conflicts.
# We avoid cleaning for metadata/help/egg-info commands to prevent breaking PEP 517 workflows.
if (len(sys.argv) > 1 and 
    any(cmd in sys.argv for cmd in ['build', 'build_ext', 'install', 'develop', 'bdist_wheel']) and 
    not any(h in sys.argv for h in ['--help', '-h', '--version', 'clean', 'egg_info', 'dist_info'])):
    
    print("setup.py: Cleaning previous build directories and binaries for a fresh compilation...")
    for folder in ["build", "dist", "hybrid_particle_tracker.egg-info"]:
        path = Path(folder)
        if path.exists():
            try:
                shutil.rmtree(path)
            except Exception:
                pass
    for file in list(Path(".").glob("particle_engine_cpp*.pyd")) + list(Path(".").glob("particle_engine_cpp*.so")):
        try:
            file.unlink()
        except Exception:
            pass

# --- 1. Dynamic Dependency & Build Configuration ---
# We define a custom build_ext command that lazily checks for and configures
# build dependencies (numpy, pybind11) only when building extensions.
# This prevents setup.py from crashing during metadata discovery or dependency resolution.

try:
    from pybind11.setup_helpers import build_ext as _build_ext
    pybind11_available = True
except ImportError:
    from setuptools.command.build_ext import build_ext as _build_ext
    pybind11_available = False

class BuildExt(_build_ext):
    def build_extensions(self):
        try:
            import numpy as np
            import pybind11
        except ImportError as e:
            print("\n" + "="*70)
            print("CRITICAL BUILD ERROR: Missing build dependencies.")
            print(f"Could not import: {e.name}")
            print("HINT: Ensure numpy and pybind11 are installed in your current environment:")
            print("      pip install numpy pybind11")
            print("      If using a cluster, make sure your python environment (e.g. 'hpt_master') is active.")
            print("="*70 + "\n")
            sys.exit(1)
            
        # Dynamically inject the numpy and pybind11 include paths
        for ext in self.extensions:
            if ext.name == "particle_engine_cpp":
                np_inc = np.get_include()
                pb_inc = pybind11.get_include()
                if np_inc not in ext.include_dirs:
                    ext.include_dirs.append(np_inc)
                if pb_inc not in ext.include_dirs:
                    ext.include_dirs.append(pb_inc)
                    
        super().build_extensions()

# --- 2. C++ Extension Configuration ---
# Fallback to standard setuptools Extension if pybind11 is not yet installed.
try:
    from pybind11.setup_helpers import Pybind11Extension
    ext_modules = [
        Pybind11Extension(
            "particle_engine_cpp",
            [
                "src/cpp/particle_engine.cpp",
                "src/cpp/interpolator.cpp",
                "src/cpp/rk4_integrator.cpp",
                "src/cpp/rk4_integrator_parallel.cpp",
            ],
            include_dirs=["src/cpp"],
            language='c++',
            cxx_std=17,
            define_macros=[
                ("_USE_MATH_DEFINES", None),
            ],
            # Enable OpenMP for parallel processing
            extra_compile_args=['/openmp'] if sys.platform == 'win32' else ['-fopenmp', '-O3'],
            extra_link_args=[] if sys.platform == 'win32' else ['-fopenmp'],
        ),
    ]
except ImportError:
    # Use standard Extension; compiler flags for C++17 will be appended manually
    ext_compile_args = ['/openmp'] if sys.platform == 'win32' else ['-fopenmp', '-O3']
    if sys.platform != 'win32':
        ext_compile_args.append('-std=c++17')
    else:
        ext_compile_args.append('/std:c++17')
        
    ext_modules = [
        Extension(
            "particle_engine_cpp",
            [
                "src/cpp/particle_engine.cpp",
                "src/cpp/interpolator.cpp",
                "src/cpp/rk4_integrator.cpp",
                "src/cpp/rk4_integrator_parallel.cpp",
            ],
            include_dirs=["src/cpp"],
            language='c++',
            define_macros=[
                ("_USE_MATH_DEFINES", None),
            ],
            extra_compile_args=ext_compile_args,
            extra_link_args=[] if sys.platform == 'win32' else ['-fopenmp'],
        ),
    ]

setup(
    name="hybrid_particle_tracker",
    version="1.0.0",
    author="Particle Tracking Team",
    description="High-performance hybrid C++/Python particle tracking system",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    zip_safe=False,
    python_requires=">=3.8",
    
    # Dependencies are managed offline via scripts or environments
    install_requires=[
        # "numpy>=1.20.0",
        # "pandas>=1.3.0", 
        # "scipy>=1.7.0",
    ],
    
    packages=["hybrid_particle_tracker"],
    package_dir={"hybrid_particle_tracker": "src/python"},
    package_data={"hybrid_particle_tracker": ["*.py"]},
    include_package_data=True,
)