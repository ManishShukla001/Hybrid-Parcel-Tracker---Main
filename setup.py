from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
from setuptools import setup, Extension, find_packages
import numpy as np

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "particle_engine_cpp",
        [
            "src/cpp/particle_engine.cpp",
            "src/cpp/interpolator.cpp",
            "src/cpp/rk4_integrator.cpp",
        ],
        include_dirs=[
            pybind11.get_include(),
            np.get_include(),
            "src/cpp",
        ],
        language='c++',
        cxx_std=17,
        define_macros=[("_USE_MATH_DEFINES", None)],
    ),
]

setup(
    name="hybrid_particle_tracker",
    version="1.0.0",
    author="Manish Shukla, R. Maheshwaran",
    author_email="manishshukla01@live.com",
    description="High-performance hybrid C++/Python particle tracking system",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "cartopy>=0.20.0",
        "tqdm>=4.60.0",
        "cdsapi>=0.6.1",
        "xarray>=0.20.0", # For reading NetCDF files
        "pybind11>=2.10.0",
    ],
    packages=["hybrid_particle_tracker"],
    package_dir={"hybrid_particle_tracker": "src/python"},
    package_data={"hybrid_particle_tracker": ["*.py"]},
    include_package_data=True,
)
