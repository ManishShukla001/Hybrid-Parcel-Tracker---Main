import xarray as xr
import numpy as np
import cdsapi
import os
import argparse
from datetime import datetime

def download_era5_data(year, month, days, area, output_file):
    print(f"Downloading ERA5 data for {year}-{month:02d} to {output_file}...")
    c = cdsapi.Client()
    
    # Define pressure levels (common levels used in the tracker)
    pressure_levels = [
        '100', '150', '200', '250', '300', '400', '500', 
        '600', '700', '850', '925', '1000'
    ]
    
    # We need temperature (t) for the full decomposition
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'temperature'
            ],
            'pressure_level': pressure_levels,
            'year': str(year),
            'month': f"{month:02d}",
            'day': [f"{d:02d}" for d in days],
            'time': [
                '00:00', '01:00', '02:00', '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00', '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00', '21:00', '22:00', '23:00',
            ],
            'area': area, # [North, West, South, East]
        },
        output_file)
    print("Download complete.")

def compute_thermo_background(input_file, output_file):
    print(f"Computing thermodynamic background from {input_file}...")
    ds = xr.open_dataset(input_file)
    
    # Assume coordinates are 'time', 'level', 'latitude', 'longitude'
    # Calculate T_mean (for simplification, here we use daily mean as the slowly varying background)
    # Alternatively, use a rolling mean or simply the exact field if 'climatological' means smoothed.
    # The requirement says "climatological background data (T_mean)".
    # For a specific event, a 3-day or 7-day rolling mean can act as background. 
    # Let's use a 24-hour rolling mean.
    
    print("Calculating T_mean...")
    t_mean = ds['t'].rolling(time=24, center=True).mean().bfill('time').ffill('time')
    
    # Calculate gradients of T_mean
    # Earth radius in meters
    R = 6371000.0
    
    # Lat/Lon in radians
    lat_rad = np.deg2rad(ds.latitude)
    lon_rad = np.deg2rad(ds.longitude)
    
    print("Calculating horizontal gradients...")
    # grad_t_lat = d(T_mean) / dy
    dy = R * np.gradient(lat_rad)
    # xarray differentiate or numpy gradient
    # We'll use xarray differentiate, but need to divide by dy
    # dt/dlat gives K/degree. 1 degree = pi/180 rad. dy = R * dlat
    dt_dlat = t_mean.differentiate('latitude') # K/degree
    grad_t_lat = dt_dlat / (R * np.pi / 180.0) # K/m
    
    # grad_t_lon = d(T_mean) / dx
    # dx = R * cos(lat) * dlon
    dt_dlon = t_mean.differentiate('longitude') # K/degree
    cos_lat = np.cos(lat_rad)
    # Broadcast cos_lat to match dimensions if necessary, but xarray handles it
    grad_t_lon = dt_dlon / (R * np.pi / 180.0 * cos_lat) # K/m
    
    print("Calculating vertical gradient...")
    # grad_t_p = d(T_mean) / dp
    # Level is in hPa, so multiply by 100 to get Pa
    pressure_pa = ds.level * 100.0
    # Create a DataArray for pressure in Pa to differentiate with respect to it
    t_mean_pa = t_mean.assign_coords(level_pa=("level", pressure_pa.data))
    t_mean_pa = t_mean_pa.swap_dims({"level": "level_pa"})
    grad_t_p = t_mean_pa.differentiate('level_pa') # K/Pa
    # Swap back
    grad_t_p = grad_t_p.swap_dims({"level_pa": "level"}).drop_vars("level_pa")
    
    print("Calculating temporal derivative...")
    # dt_mean_dt = d(T_mean) / dt
    # time is datetime64, difference in nanoseconds, convert to seconds
    # differentiate('time') returns values per ns
    dt_mean_dt_ns = t_mean.differentiate('time')
    dt_mean_dt = dt_mean_dt_ns * 1e9 # K/s
    
    print("Saving to dataset...")
    out_ds = xr.Dataset({
        't_mean': t_mean.astype(np.float32),
        'grad_t_lat': grad_t_lat.astype(np.float32),
        'grad_t_lon': grad_t_lon.astype(np.float32),
        'grad_t_p': grad_t_p.astype(np.float32),
        'dt_mean_dt': dt_mean_dt.astype(np.float32)
    })
    
    out_ds.to_netcdf(output_file)
    print(f"Preprocessed background data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and preprocess ERA5 data for Method 2")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--days", type=int, nargs='+', required=True)
    parser.add_argument("--area", type=float, nargs=4, default=[90, -180, -90, 180], help="North West South East")
    parser.add_argument("--download_only", action="store_true")
    parser.add_argument("--process_only", action="store_true")
    parser.add_argument("--raw_file", type=str, default="era5_raw_t.nc")
    parser.add_argument("--out_file", type=str, default="era5_thermo_bg.nc")
    
    args = parser.parse_args()
    
    if not args.process_only:
        download_era5_data(args.year, args.month, args.days, args.area, args.raw_file)
        
    if not args.download_only:
        compute_thermo_background(args.raw_file, args.out_file)
