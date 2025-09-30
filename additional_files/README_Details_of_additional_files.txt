Pro# Details of Additional Files
===========================

## File Organization
This folder 'Additional_Files' contains diagnostic, development, testing, and convenience files
that are not part of the core particle simulation workflow but may be useful for:
- Debugging issues
- Testing consistency
- Building/installing on different systems
- Advanced users who need customization

## Files in This Folder

### Build & Development Tools
build.py                              - Custom build script that compiles C++ extension
debug_cpp_interpolator.py            - Script for debugging C++ interpolator issues
debug_simulation.py                  - Debug version of simulation to identify particle movement issues
install_and_test.py                  - Installation and basic testing script
pre_git_verification.py             - Final verification test before GitHub upload
test_build.py                        - Verify that the build works correctly
test_cpp_fix.py                      - Test C++ interpolator fixes
test_direct.py                       - Test direct import functionality
test_simulation_consistency.py       - Comprehensive test comparing C++ vs Python/Scipy results

### Alternative Approaches
run_simulation_direct.py             - Version that imports modules directly (no package installation needed)

### Legacy Code & Documentation
Structure_of_NC_file.txt             - Internal documentation about NetCDF file structure

## Pre-Processing IMDAA to CSV converter Codes
Imdaa_Extractu.py
Imdaa_Extractv.py
Imdaa_Extractw.py

### When to Use These Files:

1. **For Development & Debugging:**
   - test_simulation_consistency.py   → Verify C++ matches Python implementation
   - debug_simulation.py            → Test if particles are moving
   - debug_cpp_interpolator.py       → Test C++ interpolation

2. **For Installation Issues:**
   - build.py                       → Custom build if setup.py fails
   - install_and_test.py            → Complete installation + verification
   - test_build.py                  → Quick build verification

3. **For Advanced Users:**
   - run_simulation_direct.py       → Run without package installation

### Files You Can Safely Delete:
- test_simulation_consistency.py   (very long output, use only when needed)
- debug_*.py                      (only needed when troubleshooting)


## Important Notes:
- Most debugging tools are only needed during development, not regular use

