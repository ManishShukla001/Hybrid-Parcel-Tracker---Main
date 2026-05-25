
import xarray as xr
import numpy as np
from pathlib import Path
import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict
import gc
import pandas as pd

class NCDataLoader:
    """
    Handles loading and preprocessing of velocity field data from NetCDF files.
    """

    def __init__(self, nc_data_dir: Path, nc_file_pattern: Optional[str] = None):
        """
        Initialize the NCDataLoader.

        Args:
            nc_data_dir: Directory containing .nc files.
            nc_file_pattern: Optional pattern to match file names. If None, tries to auto-detect.
                             Example: "combined_{year}{month}{day}_{hour}.nc"
        """
        self.nc_data_dir = Path(nc_data_dir)
        self.nc_file_pattern = nc_file_pattern
        self.files_map = {} # Maps datetime (or hour index) to file path
        
        # Metadata to be extracted
        self.pressure_levels = None
        self.latitude_values = None
        self.longitude_values = None
        self.time_units = None
        self.base_time = None
        
        self._discover_files()
        self._extract_metadata(next(iter(self.files_map.values())))

    def _discover_files(self):
        """
        Scans the directory to map files to timestamps/hours.
        Assumes file naming convention: combined_YYYYMMDD_HH.nc
        """
        print(f"Scanning {self.nc_data_dir} for .nc files...")
        
        if not self.nc_data_dir.exists():
             raise FileNotFoundError(f"NetCDF data directory not found: {self.nc_data_dir}")

        # Regex for combined_YYYYMMDD_HH.nc
        # Adjust regex based on actual file naming if different
        pattern = re.compile(r"combined_(\d{4})(\d{2})(\d{2})_(\d{2})\.nc")
        
        found_files = 0
        for file_path in self.nc_data_dir.glob("*.nc"):
            match = pattern.match(file_path.name)
            if match:
                year, month, day, hour = map(int, match.groups())
                dt = datetime(year, month, day, hour, tzinfo=timezone.utc)
                self.files_map[dt] = file_path
                found_files += 1
                
        if found_files == 0:
            # Fallback scan all .nc files and look for pattern match in looser way or error
            print(f"WARNING: No files matching 'combined_YYYYMMDD_HH.nc' found. Checking all .nc files.")
            # Implementation could be expanded for more complex patterns if needed
        
        print(f"Found {found_files} NetCDF files.")
        if found_files > 0:
             # Sort files by datetime
            self.sorted_datetimes = sorted(self.files_map.keys())
            self.start_datetime = self.sorted_datetimes[0]
            self.end_datetime = self.sorted_datetimes[-1]
            print(f"Data range: {self.start_datetime} to {self.end_datetime}")

    def _extract_metadata(self, file_path: Path):
        """
        Extracts pressure levels, latitude, longitude from the first file.
        """
        print(f"Extracting metadata from {file_path.name}...")
        try:
            with xr.open_dataset(file_path) as ds:
                # Identify coordinate names (standard names often vary)
                # Look for 'level', 'pressure', 'isobaricInhPa' etc.
                if 'level' in ds.coords:
                    self.pressure_levels = ds['level'].values
                elif 'pressure' in ds.coords:
                    self.pressure_levels = ds['pressure'].values
                else:
                    # Try to find a variable with 'pressure' in standard_name
                    pass # logic to be added if needed
                
                if self.pressure_levels is not None:
                    # Sort pressure levels ascending
                    self.pressure_levels = np.sort(self.pressure_levels) # Ensure sorted for consistency
                
                if 'latitude' in ds.coords:
                     self.latitude_values = np.sort(ds['latitude'].values)
                elif 'lat' in ds.coords:
                     self.latitude_values = np.sort(ds['lat'].values)

                if 'longitude' in ds.coords:
                     self.longitude_values = np.sort(ds['longitude'].values)
                elif 'lon' in ds.coords:
                     self.longitude_values = np.sort(ds['lon'].values)
                     
                print(f"Metadata extracted: {len(self.pressure_levels)} pressure levels, "
                      f"{len(self.latitude_values)} latitudes, {len(self.longitude_values)} longitudes.")

        except Exception as e:
            raise RuntimeError(f"Failed to extract metadata from {file_path}: {e}")

    def load_velocity_field(self, variable: str, target_datetime: datetime) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load velocity field for a specific variable and datetime.
        Handles loading from the correct file.
        Returns flattened values array for C++ compatibility, along with coords.
        """
        # Find the file that contains this datetime
        # Since files are hourly (based on name), we look for exact match in files_map
        # or find nearest if appropriate (usually exact match is needed for simulation step)
        
        # Round target_datetime to nearest hour to find the file key
        # Assuming data is hourly.
        file_dt = target_datetime.replace(minute=0, second=0, microsecond=0)
        
        if file_dt not in self.files_map:
            print(f"WARNING: No NetCDF file found for timestamp {file_dt}")
            return None
            
        file_path = self.files_map[file_dt]
        print(f"Loading {variable} from {file_path.name}...")
        
        try:
            # Open dataset
            with xr.open_dataset(file_path) as ds:
                if variable not in ds:
                    print(f"WARNING: Variable {variable} not found in {file_path.name}")
                    return None
                
                # Check dimensions order: we expect (time, level, lat, lon) or similar
                # We need to extract the specific time slice if the file contains multiple times
                # But 'combined_..._HH.nc' implies single time per file usually.
                # Let's check 'time' dimension.
                
                data_var = ds[variable]
                
                # Setup selectors
                sel_dict = {}
                
                # Handle time selection if 'time' dim exists
                if 'time' in data_var.dims:
                    # Use 'nearest' or specific selection?
                    # The file name timestamp is likely the time value.
                    # We select the single time step in the file.
                    sel_dict['time'] = ds['time'].values[0] # Assume one time step per file for now based on 'combined_..._00.nc' logic
                
                # We select ALL pressure levels, lat, lon
                # But we must ensure the order matches self.pressure_levels sorted order
                
                # Select and load data
                # .sel(level=self.pressure_levels) ensures we get them in the sorted order we stored
                if 'level' in data_var.dims:
                     sel_dict['level'] = self.pressure_levels
                elif 'pressure' in data_var.dims:
                     sel_dict['pressure'] = self.pressure_levels
                     
                if 'latitude' in data_var.dims:
                     sel_dict['latitude'] = self.latitude_values
                elif 'lat' in data_var.dims:
                     sel_dict['lat'] = self.latitude_values
                     
                if 'longitude' in data_var.dims:
                     sel_dict['longitude'] = self.longitude_values
                elif 'lon' in data_var.dims:
                     sel_dict['lon'] = self.longitude_values
                
                # Load data into memory
                try:
                    data_slice = data_var.sel(**sel_dict).load()
                except Exception as e_sel:
                    print(f"Selection failed: {e_sel}. Trying raw load.")
                    data_slice = data_var.load()
                
                values = data_slice.values
                
                # Handle shapes
                # Expected shape: (pressure, lat, lon) or (lat, lon, pressure) 
                # ParticleTracker code expects data corresponding to:
                # lat_coords (1D), lon_coords (1D), pressure_coords (1D)
                # and values_flat corresponding to (lat, lon, pressure) order usually?
                
                # Let's check ParticleTracker.create_interpolators:
                # nlat, nlon, npres = len(u_data[0]), len(u_data[1]), len(u_data[2])
                # u_grid = u_data[3].reshape(nlat, nlon, npres)
                # So it expects (nlat, nlon, npres).
                
                # xarray usually gives (level, lat, lon) or (time, level, lat, lon).
                # If we selected single time, we have (level, lat, lon).
                
                # We want (lat, lon, level).
                # Transpose needed.
                
                dims = data_slice.dims
                # Map dims to axes
                # We want order: lat, lon, level
                
                transpose_order = []
                if 'latitude' in dims: transpose_order.append('latitude')
                elif 'lat' in dims: transpose_order.append('lat')
                
                if 'longitude' in dims: transpose_order.append('longitude')
                elif 'lon' in dims: transpose_order.append('lon')
                
                if 'level' in dims: transpose_order.append('level')
                elif 'pressure' in dims: transpose_order.append('pressure')
                
                if len(transpose_order) != 3:
                     # Check if singular dimensions were dropped.
                     # If values shape is (lat, lon) because level=1?
                     pass

                data_transposed = data_slice.transpose(*transpose_order)
                values_ordered = data_transposed.values
                
                # Replace NaNs with 0 (or fill value)
                if np.isnan(values_ordered).any():
                    values_ordered = np.nan_to_num(values_ordered, nan=0.0)
                
                values_flat = values_ordered.flatten() # defaults to 'C' order (row-major)
                
                return self.latitude_values, self.longitude_values, self.pressure_levels, values_flat

        except Exception as e:
            print(f"ERROR reading {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_velocity_fields_for_timestep(self, target_datetime: datetime, absolute_sim_hour_for_log: Optional[float] = None) -> Optional[Tuple]:
        """
        Load all velocity components (u, v, w) and scalars (q, t) for a specific datetime.
        Optimized to open the file once.
        """
        # Round target_datetime to nearest hour to find the file key
        file_dt = target_datetime.replace(minute=0, second=0, microsecond=0)
        
        if file_dt not in self.files_map:
            print(f"WARNING: No NetCDF file found for timestamp {file_dt}")
            return None
            
        file_path = self.files_map[file_dt]
        print(f"Loading velocity fields from {file_path.name}...")
        
        try:
            with xr.open_dataset(file_path) as ds:
                # Prepare selectors
                sel_dict = {}
                
                # Time selection
                if 'time' in ds.coords:
                     sel_dict['time'] = ds['time'].values[0]

                # Level selection
                if 'level' in ds.coords:
                     sel_dict['level'] = self.pressure_levels
                elif 'pressure' in ds.coords:
                     sel_dict['pressure'] = self.pressure_levels

                # Lat/Lon selection
                if 'latitude' in ds.coords:
                     sel_dict['latitude'] = self.latitude_values
                elif 'lat' in ds.coords:
                     sel_dict['lat'] = self.latitude_values
                     
                if 'longitude' in ds.coords:
                     sel_dict['longitude'] = self.longitude_values
                elif 'lon' in ds.coords:
                     sel_dict['lon'] = self.longitude_values

                # Helper to load and format a single variable
                def load_var(var_name):
                    if var_name not in ds:
                        return None
                    
                    try:
                        data_slice = ds[var_name].sel(**sel_dict).load()
                    except Exception:
                        # Fallback if selection fails (e.g. slight precision issues)
                        data_slice = ds[var_name].load()
                    
                    # Transpose to (lat, lon, pressure)
                    dims = data_slice.dims
                    transpose_order = []
                    
                    if 'latitude' in dims: transpose_order.append('latitude')
                    elif 'lat' in dims: transpose_order.append('lat')
                    
                    if 'longitude' in dims: transpose_order.append('longitude')
                    elif 'lon' in dims: transpose_order.append('lon')
                    
                    if 'level' in dims: transpose_order.append('level')
                    elif 'pressure' in dims: transpose_order.append('pressure')
                    
                    if len(transpose_order) < 3:
                        # Handle missing dims if necessary, or just return as is (flatten will handle it)
                        pass

                    data_transposed = data_slice.transpose(*transpose_order)
                    values = data_transposed.values
                    
                    if np.isnan(values).any():
                        values = np.nan_to_num(values, nan=0.0)
                        
                    return (self.latitude_values, self.longitude_values, self.pressure_levels, values.flatten())

                # Load essential velocity variables
                u_data = load_var('u')
                v_data = load_var('v')
                w_data = load_var('w')
                
                if u_data is None or v_data is None or w_data is None:
                     print(f"ERROR: Failed to load essental velocity components (u,v,w) from {file_path.name}")
                     return None

                # Extract scalar variables dynamically
                target_scalars = ['q', 't', 'clwc', 'ciwc', 'crwc', 'cswc', 'cc']
                scalars_data = {}
                for scalar in target_scalars:
                    s_data = load_var(scalar)
                    if s_data is not None:
                        scalars_data[scalar] = s_data

                return u_data, v_data, w_data, scalars_data

        except Exception as e:
            print(f"ERROR reading {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

