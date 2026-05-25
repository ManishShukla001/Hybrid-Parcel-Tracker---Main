
import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.interpolate as interp
from scipy.spatial import Delaunay
import os
import gc
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
HRRR_DIR = Path("./hrrr_data_nc") # Adjusted based on file finding logic
ERA5_DIR = Path("./api_files")
OUTPUT_DIR = Path("./combined_nc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target Grid Definition (Regular)
LAT_MIN, LAT_MAX = -20.0, 70.0
LON_MIN, LON_MAX = -140.0, -40.0
RESOLUTION = 0.03  # degrees (~3km)

# Target Lat/Lon Arrays
TARGET_LATS = np.arange(LAT_MIN, LAT_MAX + RESOLUTION/2, RESOLUTION) # Inclusive
TARGET_LONS = np.arange(LON_MIN, LON_MAX + RESOLUTION/2, RESOLUTION)
# Ensure sorted
TARGET_LATS.sort()
TARGET_LONS.sort()

VAR_MAPPING_ERA5 = {
    't': 't', 'u': 'u', 'v': 'v', 'w': 'w', 'q': 'q', 'z': 'z'
}

def get_hrrr_reference_info():
    """Finds a valid HRRR NC file to extract Levels and Coordinates."""
    input_dir = Path("./hrrr_data_nc")
    files = list(input_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError("No HRRR NetCDF files found in ./hrrr_data_nc")
    
    ds = xr.open_dataset(files[0], decode_cf=False)
    
    # Levels
    levels = ds['level'].values.astype(float)
    
    # Lat/Lon
    if 'latitude' in ds.variables:
        lat = ds['latitude'].values
        lon = ds['longitude'].values
    elif 'lat' in ds.variables:
        lat = ds['lat'].values
        lon = ds['lon'].values
    else:
        # Fallback
        pass
        
    ds.close()
    return levels, lat, lon

def normalize_longitude(lon):
    """Converts 0..360 to -180..180"""
    return (lon + 180) % 360 - 180

def build_vertical_weights(source_levels, target_levels):
    """
    Computes interpolation weights for converting Source Levels -> Target Levels.
    Returns: list of (src0, src1, w)
    """
    src_sorted_idx = np.argsort(source_levels)[::-1]
    src_levels_sorted = source_levels[src_sorted_idx]
    
    indices = []
    
    for t_lev in target_levels:
        if t_lev >= src_levels_sorted[0]:
            indices.append((src_sorted_idx[0], src_sorted_idx[0], 0.0))
            continue
        if t_lev <= src_levels_sorted[-1]:
            indices.append((src_sorted_idx[-1], src_sorted_idx[-1], 0.0))
            continue
            
        idx = np.searchsorted(-src_levels_sorted, -t_lev)
        idx_above = idx - 1
        idx_below = idx
        
        p_above = src_levels_sorted[idx_above]
        p_below = src_levels_sorted[idx_below]
        
        w = (p_above - t_lev) / (p_above - p_below)
        indices.append((src_sorted_idx[idx_above], src_sorted_idx[idx_below], w))

    return indices

def process_hour(date_str, hour_idx, hrrr_interp_info):
    """
    Processes a SINGLE HOUR and writes to a separate NetCDF file.
    hrrr_interp_info: (DelaunayObject, levels)
    """
    try:
        # Construct output filename
        # Hour formatting: 00, 01, ...
        output_filename = f"combined_{date_str}_{hour_idx:02d}.nc"
        output_path = OUTPUT_DIR / output_filename
        
        if output_path.exists():
            print(f"Skipping {output_filename} (Exists)")
            return output_filename
            
        print(f"Starting {output_filename}...")
        
        hrrr_file = Path(f"./hrrr_data_nc/api_daily_{date_str}.nc")
        era5_file = Path(f"./api_files/api_daily_{date_str}.nc")
        
        # Open Files
        ds_hrrr = xr.open_dataset(hrrr_file, decode_cf=False)
        if 'step' in ds_hrrr.variables:
             if 'dtype' in ds_hrrr['step'].attrs: del ds_hrrr['step'].attrs['dtype']
        ds_hrrr = xr.decode_cf(ds_hrrr)
        
        ds_era5 = xr.open_dataset(era5_file, decode_times=False)
        
        # Get target levels from info
        target_levels = hrrr_interp_info['levels'] 
        hrrr_tri = hrrr_interp_info['tri']
        
        # Verify ERA5 coordinates
        if 'isobaricInhPa' in ds_era5.coords:
            era5_levs = ds_era5['isobaricInhPa'].values
        elif 'level' in ds_era5.coords:
            era5_levs = ds_era5['level'].values
        elif 'pressure_level' in ds_era5.coords:
            era5_levs = ds_era5['pressure_level'].values
        else:
            raise ValueError("ERA5 level coord not found")
            
        v_weights = build_vertical_weights(era5_levs, target_levels)

        # Select Slice
        # Handle time dim name
        if 'valid_time' in ds_era5.dims:
            era5_time_ds = ds_era5.isel(valid_time=hour_idx)
            # time_val = ds_era5['valid_time'].isel(valid_time=hour_idx).values
        else:
            era5_time_ds = ds_era5.isel(time=hour_idx)
            # time_val = ds_era5['time'].isel(time=hour_idx).values
            
        # Get Time Attributes for copying
        time_var_name = 'valid_time' if 'valid_time' in ds_era5.variables else 'time'
        time_orig_val = ds_era5[time_var_name].values[hour_idx]
        time_units = ds_era5[time_var_name].attrs.get('units', 'hours since 1900-01-01')

        # Create NetCDF
        import netCDF4
        n_lats = len(TARGET_LATS)
        n_lons = len(TARGET_LONS)
        n_levs = len(target_levels)
        
        # Target Grid Mesh (Computed once per process? Or re-computed?)
        # Computed once here, it's fast (ms for meshgrid, slow for find_simplex)
        # BUT we have `hrrr_tri` passed in.
        # We need `target_pts` to check validity.
        # It's better to compute target_pts inside the process to avoid passing huge array.
        target_lon_mesh, target_lat_mesh = np.meshgrid(TARGET_LONS, TARGET_LATS)
        target_pts = np.vstack([target_lat_mesh.ravel(), target_lon_mesh.ravel()]).T
        
        # Find Simplex Indices (The most expensive CPU part, ~30s-60s)
        # Can we avoid re-computing this 24 times for the same date?
        # Yes, if we passed `simplex_indices` in `hrrr_interp_info`.
        # But `hrrr_interp_info` is shared across ALL dates/hours.
        # So yes, we should pre-compute simplex_indices in MAIN and pass it!
        # Wait, if output grid is constant, simplex_indices is constant.
        # PASS IT!
        simplex_indices = hrrr_interp_info['simplex_indices']
        valid_hrrr_mask = simplex_indices != -1

        with netCDF4.Dataset(output_path, 'w', format='NETCDF4') as nc:
            # Dims: Time=1 (or unlimited?), Level, Lat, Lon
            nc.createDimension('time', 1)
            nc.createDimension('level', n_levs)
            nc.createDimension('latitude', n_lats)
            nc.createDimension('longitude', n_lons)
            
            v_t = nc.createVariable('time', 'f8', ('time',))
            v_t[0] = time_orig_val
            v_t.units = time_units
            
            v_l = nc.createVariable('level', 'f4', ('level',))
            v_l[:] = target_levels
            v_l.units = 'hPa'
            
            v_lat = nc.createVariable('latitude', 'f4', ('latitude',))
            v_lat[:] = TARGET_LATS
            v_lat.units = 'degrees_north'
            
            v_lon = nc.createVariable('longitude', 'f4', ('longitude',))
            v_lon[:] = TARGET_LONS
            v_lon.units = 'degrees_east'

            # Chunking: (1, 1, 500, 500)
            chunks = (1, 1, 500, 500)
            
            vars_to_proc = ['t', 'u', 'v', 'w', 'q', 'z']
            
            e_lats = ds_era5['latitude'].values
            e_lons = ds_era5['longitude'].values
            if e_lats[0] > e_lats[-1]:
                e_lats = e_lats[::-1]
                invert_lat = True
            else:
                invert_lat = False
            
            for var in vars_to_proc:
                hrrr_avail =  var in ds_hrrr.variables or (var == 'z' and 'gh' in ds_hrrr.variables)
                era_avail = VAR_MAPPING_ERA5.get(var, var) in ds_era5.variables
                
                if hrrr_avail or era_avail:
                     v_nc = nc.createVariable(var, 'f4', ('time', 'level', 'latitude', 'longitude'), 
                                              zlib=True, complevel=4, chunksizes=chunks, fill_value=np.nan)
                     
                     # ---------------------------------------------------------
                     # Processing Logic (Level by Level)
                     # ---------------------------------------------------------
                     v_name_era = VAR_MAPPING_ERA5.get(var, var)
                     
                     # Extract ERA5 Volume (Lev, Lat, Lon)
                     era5_vol = None
                     if v_name_era in era5_time_ds:
                         era5_vol = era5_time_ds[v_name_era].values 
                         if invert_lat: era5_vol = era5_vol[:, ::-1, :]
                         era5_vol = np.moveaxis(era5_vol, 0, -1) #(Lat, Lon, Lev)
                         
                     # Extract HRRR Volume (Lev, Y, X)
                     hrrr_vol = None
                     if hrrr_avail:
                         hrrr_v_name = 'gh' if var == 'z' and 'gh' in ds_hrrr else var
                         try:
                             hrrr_vol = ds_hrrr[hrrr_v_name].isel(time=hour_idx).values 
                         except:
                             hrrr_vol = None
                             
                     # Iterate Levels
                     for lev_idx, _ in enumerate(target_levels):
                         final_slab_flat = np.full((len(target_pts),), np.nan, dtype=np.float32)
                         
                         # Interpolate ERA5
                         if era5_vol is not None:
                             src0, src1, w = v_weights[lev_idx]
                             slice0 = era5_vol[:, :, src0]
                             slice1 = era5_vol[:, :, src1]
                             
                             rgi0 = interp.RegularGridInterpolator((e_lats, e_lons), slice0, bounds_error=False, fill_value=np.nan)
                             val0 = rgi0(target_pts)
                             
                             if src0 == src1:
                                 val_interp = val0
                             else:
                                 rgi1 = interp.RegularGridInterpolator((e_lats, e_lons), slice1, bounds_error=False, fill_value=np.nan)
                                 val1 = rgi1(target_pts)
                                 val_interp = val0 * (1-w) + val1 * w
                                 del val1, rgi1
                             
                             final_slab_flat[:] = val_interp
                             del val0, rgi0, val_interp
                             
                         # Overlay HRRR
                         if hrrr_vol is not None:
                             try:
                                 h_layer = hrrr_vol[lev_idx, :, :]
                                 h_flat = h_layer.ravel()
                                 
                                 # Triangulation
                                 hrrr_lnd = interp.LinearNDInterpolator(hrrr_tri, h_flat)
                                 
                                 # Subset
                                 pts_subset = target_pts[valid_hrrr_mask]
                                 hrrr_vals = hrrr_lnd(pts_subset)
                                 
                                 hrrr_nan_mask = np.isnan(hrrr_vals)
                                 current_vals = final_slab_flat[valid_hrrr_mask]
                                 np.putmask(current_vals, ~hrrr_nan_mask, hrrr_vals)
                                 final_slab_flat[valid_hrrr_mask] = current_vals
                                 
                                 del hrrr_lnd, hrrr_vals, h_flat, h_layer
                             except:
                                 pass
                                 
                         # Write
                         slab_2d = final_slab_flat.reshape(n_lats, n_lons)
                         v_nc[0, lev_idx, :, :] = slab_2d
                         del final_slab_flat, slab_2d
                         
                     del era5_vol, hrrr_vol
                     gc.collect()

        ds_hrrr.close()
        ds_era5.close()
        # validate_output(output_path) # Optional: Skip for speed in parallel mode? or Keep?
        # Keeping it is good for verification, but prints might interleave.
        return output_filename

    except Exception as e:
        print(f"ERROR in {date_str} H{hour_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("Preparing Global Interpolation Info...")
    t_levs, h_lats, h_lons = get_hrrr_reference_info()
    
    h_lons_norm = normalize_longitude(h_lons)
    hrrr_pts = np.vstack([h_lats.ravel(), h_lons_norm.ravel()]).T
    
    print(f"  HRRR Grid Points: {len(hrrr_pts)}")
    print("  Building Delaunay Triangulation...")
    tri = Delaunay(hrrr_pts)
    
    # Pre-compute Simplex Indices for Target Grid (MASSIVE SPEEDUP for workers)
    print("  Pre-computing Target Grid Simplex Indices...")
    target_lon_mesh, target_lat_mesh = np.meshgrid(TARGET_LONS, TARGET_LATS)
    target_pts = np.vstack([target_lat_mesh.ravel(), target_lon_mesh.ravel()]).T
    
    simplex_indices = tri.find_simplex(target_pts)
    
    hrrr_info = {
        'levels': t_levs,
        'tri': tri,
        'simplex_indices': simplex_indices
    }
    
    # Scan Dates
    hrrr_files = sorted(list(Path("./hrrr_data_nc").glob("api_daily_*.nc")))
    tasks = []
    
    for f in hrrr_files:
        d = f.stem.split('_')[-1]
        for h in range(24):
            tasks.append((d, h))
            
    print(f"Found {len(tasks)} hourly tasks.")
    
    # Parallel Config
    MAX_WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', 50)) 
    # User said 60-70 cores available. Default to 50 for safety.
    if MAX_WORKERS == 1: MAX_WORKERS = 60 # Check if interactive shell shows 1
    
    print(f"Starting execution with {MAX_WORKERS} workers...")
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_hour, d, h, hrrr_info): (d, h) for d, h in tasks}
        
        for f in as_completed(futures):
            d, h = futures[f]
            try:
                res = f.result()
                if res:
                    print(f"[DONE] {res}")
                else:
                    print(f"[FAIL] {d} Hour {h}")
            except Exception as e:
                print(f"[CRASH] {d} Hour {h}: {e}")

if __name__ == "__main__":
    main()
