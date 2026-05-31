"""
Handles downloading and processing of ERA5 data via CDS API.
"""
try:
    import cdsapi
    CDSAPI_AVAILABLE = True
except ImportError:
    cdsapi = None
    CDSAPI_AVAILABLE = False

import xarray as xr
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
import time
import os
from typing import Tuple, Optional, List, Dict, Any

class APIDataHandler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_data_dir = Path(config['api_data_dir'])
        self.api_data_dir.mkdir(parents=True, exist_ok=True)
        self.pressure_levels_str = [str(p) for p in config['pressure_levels']]
        
        # Area: [North, West, South, East]
        # Config lat_range is (min_lat, max_lat), lon_range is (min_lon, max_lon)
        # CDS API area is [max_lat, min_lon, min_lat, max_lon]
        init_lat_range = config.get('initialization_lat_range', (-90, 90))
        init_lon_range = config.get('initialization_lon_range', (-180, 180))
        self.area = [init_lat_range[1], init_lon_range[0], init_lat_range[0], init_lon_range[1]]

        self.area = [init_lat_range[1], init_lon_range[0], init_lat_range[0], init_lon_range[1]]

        self.client = None # Lazy initialization to avoid connection in offline mode if files exist
        
        # Dataset cache to avoid re-opening daily files repeatedly
        self.cached_ds = None
        self.cached_filepath = None

    def _get_daily_filepath(self, date_obj: datetime.date) -> Path:
        """Generates the filepath for a daily NetCDF file."""
        return self.api_data_dir / f"api_daily_{date_obj.strftime('%Y%m%d')}.nc"

    def _download_daily_data(self, date_obj: datetime.date) -> bool:
        """Downloads ERA5 data for a specific day if it doesn't exist."""
        filepath = self._get_daily_filepath(date_obj)
        if filepath.exists():
            print(f"API data for {date_obj.strftime('%Y-%m-%d')} already exists: {filepath}")
            return True

        print(f"Downloading API data for {date_obj.strftime('%Y-%m-%d')} to {filepath}...")
        
        request = {
            "product_type": "reanalysis",
            "format": "netcdf", # Changed from data_format for clarity
            "variable": [
                "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
                "specific_humidity", "temperature", "geopotential",
                "specific_cloud_liquid_water_content", "specific_cloud_ice_water_content",
                "specific_rain_water_content", "specific_snow_water_content",
                "fraction_of_cloud_cover"
            ],
            "pressure_level": self.pressure_levels_str,
            "year": str(date_obj.year),
            "month": f"{date_obj.month:02d}",
            "day": f"{date_obj.day:02d}",
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": self.area,
            # "download_format": "unarchived" # This option might not be valid for all CDS datasets/APIs
        }

        try:
            if self.client is None:
                 self.client = cdsapi.Client(retry_max=5, sleep_max=120)
            self.client.retrieve("reanalysis-era5-pressure-levels", request, str(filepath))
            print(f"Successfully downloaded {filepath}")
            return True
        except Exception as e:
            print(f"ERROR downloading data for {date_obj.strftime('%Y-%m-%d')}: {e}")
            if filepath.exists(): # Remove partial file if download failed
                os.remove(filepath)
            return False

    def load_fields_for_hour(self, target_datetime_utc: datetime) -> Optional[Tuple]:
        """
        Loads u, v, w fields for a specific UTC datetime.
        Returns data in the format: (u_tuple, v_tuple, w_tuple, q_tuple, t_tuple)
        where each _tuple is (lats_np, lons_np, pressures_np, flattened_values_np)
        Also loads q and t but doesn't return them in this tuple for now.
        """
        target_date = target_datetime_utc.date()
        if not self._download_daily_data(target_date):
            return None

        filepath = self._get_daily_filepath(target_date)
        print(f"Loading fields from {filepath} for hour {target_datetime_utc.hour} UTC...")

        # Open and cache daily dataset if it's a new file
        if self.cached_filepath != filepath:
            if self.cached_ds is not None:
                try:
                    self.cached_ds.close()
                except Exception as e_close:
                    print(f"WARNING: Error closing cached dataset: {e_close}")
            self.cached_ds = None
            self.cached_filepath = None
            import gc
            gc.collect()
            
            # Load the entire dataset into memory to avoid frequent disk I/O when slicing hourly data
            if self.config.get('cache_in_memory', True):
                print(f"Opening and loading NetCDF dataset into memory: {filepath}")
                try:
                    with xr.open_dataset(filepath) as ds_temp:
                        self.cached_ds = ds_temp.load()
                    self.cached_filepath = filepath
                except Exception as e_open:
                    print(f"ERROR: Failed to load dataset {filepath} into memory: {e_open}")
                    return None
            else:
                print(f"Opening NetCDF dataset (lazy access): {filepath}")
                try:
                    self.cached_ds = xr.open_dataset(filepath)
                    self.cached_filepath = filepath
                except Exception as e_open:
                    print(f"ERROR: Failed to open dataset {filepath}: {e_open}")
                    return None

        ds = self.cached_ds
        if ds is None:
            return None

        try:
            # Select data for the specific hour
            data_for_hour = None
            time_coord_name = None

            # Try common time coordinate names
            possible_time_coords = ['time', 'valid_time', 'forecast_time', 't']
            for tc_name in possible_time_coords:
                if tc_name in ds.coords:
                    time_coord_name = tc_name
                    break
            
            if not time_coord_name:
                print(f"ERROR: Could not find a recognizable time coordinate in {filepath}. Available coords: {list(ds.coords.keys())}")
                return None

            # Attempt to select by datetime64 value first
            try:
                time_to_select_dt64 = np.datetime64(target_datetime_utc.replace(tzinfo=None))
                data_for_hour = ds.sel({time_coord_name: time_to_select_dt64})
                print(f"Selected time using datetime64: {time_to_select_dt64}")
            except KeyError:
                # If direct datetime selection fails, try selecting by hour index if the time dim is just 0-23
                if ds[time_coord_name].ndim == 1 and len(ds[time_coord_name]) == 24: # Assuming 24 hourly slices
                    hour_index = target_datetime_utc.hour
                    data_for_hour = ds.isel({time_coord_name: hour_index})
                    print(f"Selected time using hour index: {hour_index} (for target hour {target_datetime_utc.hour})")
                else:
                    print(f"ERROR: Time {time_to_select_dt64} not found in {filepath} using coordinate '{time_coord_name}'.")
                    print(f"Available times for '{time_coord_name}': {ds[time_coord_name].values}")
                    return None
            except Exception as e_sel:
                print(f"ERROR during time selection with '{time_coord_name}': {e_sel}")
                return None

            if data_for_hour is None:
                print(f"ERROR: Could not select data for hour {target_datetime_utc.hour} from {filepath}")
                return None

            lats_np = data_for_hour['latitude'].values
            lons_np = data_for_hour['longitude'].values
            # Ensure pressure levels from config are sorted ascending (required for interpolators)
            self.pressure_levels = np.sort(self.config['pressure_levels'])
            pressures_np = self.pressure_levels

            # Find the pressure coordinate name in the dataset
            pressure_coord_name = None
            possible_p_coords = ['pressure_level', 'level', 'isobaricInhPa']
            for pc in possible_p_coords:
                if pc in data_for_hour.coords:
                    pressure_coord_name = pc
                    break
            
            if not pressure_coord_name:
                print(f"ERROR: Could not find pressure coordinate in {filepath}. Available: {list(data_for_hour.coords.keys())}")
                return None

            # Reindex data to match the sorted pressure levels from config
            try:
                data_for_hour = data_for_hour.reindex({pressure_coord_name: pressures_np})
            except Exception as e_reindex:
                 print(f"ERROR reindexing pressure levels: {e_reindex}")
                 return None
            
            # Verify reindexing worked (no all-NaN slices introduced due to mismatch)
            if data_for_hour['u'].isnull().all():
                 print("ERROR: All data became NaN after pressure reindexing. Check if configured pressure levels match file levels.")
                 return None

            # Ensure ascending order for lat/lon coordinates
            if not np.all(np.diff(lats_np) > 0): # ERA5 latitudes are usually descending
                lats_np = np.sort(lats_np) # Sort ascending
                data_for_hour = data_for_hour.reindex(latitude=lats_np)
            if not np.all(np.diff(lons_np) > 0):
                lons_np = np.sort(lons_np)
                data_for_hour = data_for_hour.reindex(longitude=lons_np)

            # Extract, ensure (lat, lon, pressure) order, then flatten
            u_values = data_for_hour['u'].transpose('latitude', 'longitude', pressure_coord_name).values.flatten()
            v_values = data_for_hour['v'].transpose('latitude', 'longitude', pressure_coord_name).values.flatten()
            w_values = data_for_hour['w'].transpose('latitude', 'longitude', pressure_coord_name).values.flatten() # 'w' is vertical_velocity (Pa/s)

            u_tuple = (lats_np, lons_np, pressures_np, np.nan_to_num(u_values, nan=0.0))
            v_tuple = (lats_np, lons_np, pressures_np, np.nan_to_num(v_values, nan=0.0))
            w_tuple = (lats_np, lons_np, pressures_np, np.nan_to_num(w_values, nan=0.0))

            target_scalars = ['q', 't', 'z', 'clwc', 'ciwc', 'crwc', 'cswc', 'cc']
            scalars_data = {}
            for scalar in target_scalars:
                if scalar in data_for_hour:
                    s_values = data_for_hour[scalar].transpose('latitude', 'longitude', pressure_coord_name).values.flatten()
                    scalars_data[scalar] = (lats_np, lons_np, pressures_np, np.nan_to_num(s_values, nan=0.0))
                else:
                    print(f"WARNING: Variable '{scalar}' not found in {filepath} for hour {target_datetime_utc.hour} UTC. Skipping calculation.")

            print(f"DEBUG APIDataHandler: u_tuple[3] is None? {u_tuple[3] is None}")
            print(f"DEBUG APIDataHandler: v_tuple[3] is None? {v_tuple[3] is None}")
            print(f"DEBUG APIDataHandler: w_tuple[3] is None? {w_tuple[3] is None}")

            return u_tuple, v_tuple, w_tuple, scalars_data

        except Exception as e:
            print(f"ERROR processing NetCDF file {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None

if __name__ == '__main__':
    # Example usage (for testing APIDataHandler directly)
    test_config = {
        'api_data_dir': Path("./test_api_files"),
        'pressure_levels': np.array([1000, 850, 500]),
        'initialization_lat_range': (0, 10), # N, S for area
        'initialization_lon_range': (70, 80), # W, E for area
    }
    handler = APIDataHandler(test_config)
    test_dt = datetime(2023, 1, 1, 5, 0, 0, tzinfo=timezone.utc) # Test for 5 AM UTC
    data = handler.load_fields_for_hour(test_dt)
    if data:
        print("Successfully loaded data for hour.")
        print(f"U-data lat shape: {data[0][0].shape}, lon shape: {data[0][1].shape}, val len: {len(data[0][3])}")