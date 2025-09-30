"""
Handles downloading and processing of ERA5 data via CDS API.
"""
import cdsapi
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

        self.client = cdsapi.Client(retry_max=5, sleep_max=120) # Add retry logic

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
                "specific_humidity", "temperature"
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

        try:
            with xr.open_dataset(filepath) as ds:
                # --- Debug: Print dataset structure (uncomment if needed for further debugging) ---
                #print(f"--- Inspecting NetCDF: {filepath} ---")
                #print(f"Dataset Coords: {ds.coords}")
                #print(f"Dataset Dimensions: {ds.dims}")
                #print(f"Dataset Data Vars: {list(ds.data_vars.keys())}")
                # Example: print details of a specific variable
                #if 'u' in ds.data_vars:
                    #print(f"Details for 'u' variable: {ds['u']}")
                # --- End Debug ---

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
                # Pressure levels from config, assuming they match the file's 'level' coordinate
                pressures_np = self.config['pressure_levels'] 

                # Ensure ascending order for coordinates as expected by interpolators
                if not np.all(np.diff(lats_np) > 0): # ERA5 latitudes are usually descending
                    lats_np = np.sort(lats_np) # Sort ascending
                    data_for_hour = data_for_hour.reindex(latitude=lats_np)
                if not np.all(np.diff(lons_np) > 0):
                    lons_np = np.sort(lons_np)
                    data_for_hour = data_for_hour.reindex(longitude=lons_np)
                # Pressures from config are already sorted. Ensure 'level' in data matches.
                # The pressure coordinate is likely named 'pressure_level' or similar from the API.

                # Extract, ensure (lat, lon, pressure) order, then flatten
                # xarray usually returns (time, pressure_level, lat, lon). After .sel(time=...), it's (pressure_level, lat, lon)
                # We need to transpose to (latitude, longitude, pressure_level) for consistency with CSV loader
                u_values = data_for_hour['u'].transpose('latitude', 'longitude', 'pressure_level').values.flatten()
                v_values = data_for_hour['v'].transpose('latitude', 'longitude', 'pressure_level').values.flatten()
                w_values = data_for_hour['w'].transpose('latitude', 'longitude', 'pressure_level').values.flatten() # 'w' is vertical_velocity (Pa/s)

                u_tuple = (lats_np, lons_np, pressures_np, np.nan_to_num(u_values, nan=0.0))
                v_tuple = (lats_np, lons_np, pressures_np, np.nan_to_num(v_values, nan=0.0))
                w_tuple = (lats_np, lons_np, pressures_np, np.nan_to_num(w_values, nan=0.0))

                q_tuple, t_tuple = None, None
                if 'q' in data_for_hour:
                    q_values = data_for_hour['q'].transpose('latitude', 'longitude', 'pressure_level').values.flatten()
                    q_tuple = (lats_np, lons_np, pressures_np, np.nan_to_num(q_values, nan=0.0))
                else:
                    print(f"WARNING: Variable 'q' (specific_humidity) not found in {filepath} for hour {target_datetime_utc.hour} UTC. Skipping q calculation for this hour.")
                    # Create a placeholder tuple with None for values if q is missing
                    q_tuple = (lats_np, lons_np, pressures_np, None) 

                if 't' in data_for_hour:
                    t_values = data_for_hour['t'].transpose('latitude', 'longitude', 'pressure_level').values.flatten()
                    t_tuple = (lats_np, lons_np, pressures_np, np.nan_to_num(t_values, nan=0.0))
                else:
                    print(f"WARNING: Variable 't' (temperature) not found in {filepath} for hour {target_datetime_utc.hour} UTC. Skipping t calculation for this hour.")
                    # Create a placeholder tuple with None for values if t is missing
                    t_tuple = (lats_np, lons_np, pressures_np, None)
                # ... existing code ...
                print(f"DEBUG APIDataHandler: u_tuple[3] is None? {u_tuple[3] is None}")
                print(f"DEBUG APIDataHandler: v_tuple[3] is None? {v_tuple[3] is None}")
                print(f"DEBUG APIDataHandler: w_tuple[3] is None? {w_tuple[3] is None}")
                #return u_tuple, v_tuple, w_tuple, q_tuple, t_tuple

                return u_tuple, v_tuple, w_tuple, q_tuple, t_tuple

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