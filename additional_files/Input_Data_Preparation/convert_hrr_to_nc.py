
import xarray as xr
import pandas as pd
from pathlib import Path
import numpy as np
import warnings
import gc
import os
import shutil

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
SOURCE_DIR = Path("./hrrr")
OUTPUT_DIR = Path("./hrrr_data_nc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LEVELS = [
    100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800,
    825, 850, 875, 900, 925, 950, 975, 1000
]

VAR_MAPPING = {
    't': 't', 'u': 'u', 'v': 'v', 'w': 'w', 'q': 'q', 'gh': 'z'
}

def load_hour_raw(file_path):
    """Loads a single GRIB file, filters, and returns dataset (loaded in memory)"""
    try:
        # Open with cfgrib
        # forcing indexpath='' prevents creation of .idx files which sometimes corrupt
        ds = xr.open_dataset(
            file_path, 
            engine='cfgrib', 
            backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}, 'indexpath': ''}
        )
        
        # Select levels
        available_levels = ds['isobaricInhPa'].values
        levels_to_keep = [l for l in TARGET_LEVELS if l in available_levels]
        ds = ds.sel(isobaricInhPa=levels_to_keep)
        
        # Rename
        rename_dict = {k: v for k, v in VAR_MAPPING.items() if k in ds.data_vars}
        if 'isobaricInhPa' in ds.coords:
            rename_dict['isobaricInhPa'] = 'level'
        ds = ds.rename(rename_dict)
        
        # Keep vars
        vars_to_keep = [v for v in VAR_MAPPING.values() if v in ds.data_vars]
        ds = ds[vars_to_keep]
        
        # Load to memory and close file to be safe
        ds = ds.load() 
        return ds
        
    except Exception as e:
        print(f"Error opening {file_path}: {e}")
        return None

def process_date(date_str):
    date_path = SOURCE_DIR / date_str
    if not date_path.exists():
        return

    output_file = OUTPUT_DIR / f"api_daily_{date_str.replace('-', '')}.nc"
    if output_file.exists():
        print(f"Output file {output_file} exists. Deleting to restart...")
        output_file.unlink()

    print(f"Processing {date_str} (Robust Append Mode)...")
    
    # Identify valid files
    valid_files = [] 
    for hour in range(24):
        f = date_path / f"hrrr.t{hour:02d}z.wrfprsf00.grib2"
        if f.exists():
            valid_files.append((hour, f))
    
    if not valid_files:
        print("No files found.")
        return

    print(f"  Found {len(valid_files)} hourly files.")
    
    import netCDF4

    # Loop through files
    for i, (hour, fpath) in enumerate(valid_files):
        print(f"    Hour {hour:02d}...")
        
        ds_h = load_hour_raw(fpath)
        if ds_h is None: continue
        
        # Add time dimension with correct value
        valid_time = pd.Timestamp(f"{date_str} {hour:02d}:00:00")
        ds_h = ds_h.expand_dims(time=[valid_time])
        
        if i == 0:
            # First file: Write standard NetCDF with unlimited time dimension
            # Set compression
            encoding = {var: {'zlib': True, 'complevel': 5} for var in ds_h.data_vars}
            ds_h.to_netcdf(output_file, mode='w', unlimited_dims=['time'], encoding=encoding)
            print("      Initialized NetCDF.")
        else:
            # Append using netCDF4 library directly
            # This bypasses xarray 'append_dim' compatibility issues
            try:
                with netCDF4.Dataset(output_file, 'a') as nc:
                    # Append Time Coordinate
                    # Convert timestamp to netcdf numeric value based on units
                    time_var = nc.variables['time']
                    # We use netCDF4.date2num if needed, or simple arithmetic if units are simple
                    # But easiest is to trust that xarray wrote 'hours since ...' or 'nanoseconds since ...'
                    # and usually we can append the raw value or calculate it.
                    
                    # Safer approach: Calculate offset relative to first time
                    # But we need to know the units.
                    
                    # Let's peek at the units
                    units = time_var.units
                    calendar = getattr(time_var, 'calendar', 'standard')
                    
                    # Convert current time to num
                    val = netCDF4.date2num([valid_time], units=units, calendar=calendar)[0]
                    
                    # Append time
                    idx = time_var.shape[0]
                    time_var[idx] = val
                    
                    # Append Variables
                    for var_name in ds_h.data_vars:
                         if var_name in nc.variables:
                             # Ensure dimensions match (ignoring time)
                             # ds_h[var_name] is (1, level, y, x)
                             # write to nc[var_name][idx, :, :, :]
                             nc.variables[var_name][idx, ...] = ds_h[var_name].values[0, ...]
                             
                print("      Appended.")
                
            except Exception as e:
                print(f"      Append failed: {e}")
                
        ds_h.close()
        del ds_h
        gc.collect()

    print(f"  Done. Saved to {output_file}")


def main():
    print(f"Scanning {SOURCE_DIR}...", flush=True)
    if not SOURCE_DIR.exists(): return
    date_folders = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name)==8])
    for f in date_folders:
        try:
            pd.to_datetime(f.name, format="%Y%m%d")
            process_date(f.name)
        except: pass

if __name__ == "__main__":
    main()