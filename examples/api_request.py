#!/usr/bin/env python3
"""
Script to pre-download daily ERA5 data using the CDS API.
"""

import cdsapi
from pathlib import Path
from datetime import datetime, timedelta
import time
import os

# --- Configuration ---
DATA_DIR = Path("./api_files")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Pressure levels for API request
API_PRESSURE_LEVELS = [
    100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800,
    825, 850, 875, 900, 925, 950, 975, 1000
]
# Native ERA5 grid resolution [lat, lon]
ERA5_GRID_RESOLUTION_API = [0.25, 0.25]

# API Client settings
API_RETRY_COUNT = 3
API_RETRY_DELAY_SECONDS = 120 # Increased delay for potentially busy server

# Simulation Scope and Extent
SIMULATION_SCOPE = 'REGIONAL'  # 'GLOBAL' or 'REGIONAL'
# Regional extent: (South_Lat, North_Lat, West_Lon, East_Lon)
# Note: CDS API area is [North, West, South, East]
REGIONAL_EXTENT = (-20, 50, 25, 160) # Example: (min_lat, max_lat, min_lon, max_lon)

# Variables to download - short names used here, mapped to API names later
MODEL_VARIABLES_SHORT = ['u', 'v', 'w', 'q', 't']

# Simulation Date Range (inclusive start, exclusive end)
SIM_START_DATE = datetime(2023, 12, 13)
SIM_END_DATE = datetime(2023, 12, 22) # Data will be downloaded up to 2023-12-21

# --- End Configuration ---

MODEL_TO_API_VAR_MAP = {
    'u': "u_component_of_wind",
    'v': "v_component_of_wind",
    'w': "vertical_velocity",
    'q': "specific_humidity",
    't': "temperature",
    
}

def get_api_variables(short_names: list) -> list:
    """Maps short variable names to API-compatible names."""
    api_vars = []
    for short_name in short_names:
        if short_name in MODEL_TO_API_VAR_MAP:
            api_vars.append(MODEL_TO_API_VAR_MAP[short_name])
        else:
            print(f"Warning: Short variable name '{short_name}' not found in mapping. Skipping.")
    return api_vars

def download_era5_data_for_date(target_date: datetime.date, client: cdsapi.Client):
    """Downloads ERA5 data for a specific date if it doesn't already exist."""
    
    filepath = DATA_DIR / f"api_daily_{target_date.strftime('%Y%m%d')}.nc"

    if filepath.exists():
        print(f"Data for {target_date.strftime('%Y-%m-%d')} already exists: {filepath}")
        return True

    print(f"Attempting to download data for {target_date.strftime('%Y-%m-%d')} to {filepath}...")

    api_variables_to_download = get_api_variables(MODEL_VARIABLES_SHORT)
    if not api_variables_to_download:
        print("Error: No valid API variables specified for download. Aborting.")
        return False

    if SIMULATION_SCOPE.upper() == 'REGIONAL':
        # REGIONAL_EXTENT = (South_Lat, North_Lat, West_Lon, East_Lon)
        # CDS API area = [North, West, South, East]
        area_request = [REGIONAL_EXTENT[1], REGIONAL_EXTENT[2], REGIONAL_EXTENT[0], REGIONAL_EXTENT[3]]
    else: # GLOBAL
        area_request = [90, -180, -90, 180] # Full global coverage

    request = {
        "product_type": "reanalysis",
        "format": "netcdf",
        "variable": api_variables_to_download,
        "pressure_level": [str(p) for p in API_PRESSURE_LEVELS],
        "year": str(target_date.year),
        "month": f"{target_date.month:02d}",
        "day": f"{target_date.day:02d}",
        "time": [f"{h:02d}:00" for h in range(24)], # All 24 hours
        "area": area_request,
        # "grid": f"{ERA5_GRID_RESOLUTION_API[0]}/{ERA5_GRID_RESOLUTION_API[1]}", # Let API use native grid for the area
    }

    try:
        client.retrieve("reanalysis-era5-pressure-levels", request, str(filepath))
        print(f"Successfully downloaded {filepath}")
        return True
    except Exception as e:
        print(f"ERROR downloading data for {target_date.strftime('%Y-%m-%d')}: {e}")
        if filepath.exists(): # Clean up partial file if download failed
            try:
                os.remove(filepath)
                print(f"Removed partial file: {filepath}")
            except OSError as oe:
                print(f"Error removing partial file {filepath}: {oe}")
        return False

def main():
    print("--- ERA5 Data Pre-downloader ---")
    print(f"Output directory: {DATA_DIR.resolve()}")
    print(f"Date range: {SIM_START_DATE.strftime('%Y-%m-%d')} to {(SIM_END_DATE - timedelta(days=1)).strftime('%Y-%m-%d')}")
    print(f"Variables: {MODEL_VARIABLES_SHORT}")
    print(f"Pressure Levels: {API_PRESSURE_LEVELS}")
    print(f"Scope: {SIMULATION_SCOPE}, Extent for Regional: {REGIONAL_EXTENT if SIMULATION_SCOPE.upper() == 'REGIONAL' else 'N/A'}")

    cds_client = cdsapi.Client(retry_max=API_RETRY_COUNT, sleep_max=API_RETRY_DELAY_SECONDS, quiet=False)

    current_date = SIM_START_DATE
    while current_date < SIM_END_DATE:
        download_era5_data_for_date(current_date.date(), cds_client)
        current_date += timedelta(days=1)
        # Optional: Add a small delay between daily requests if needed
        # time.sleep(5) 

    print("--- Pre-download process finished. ---")

if __name__ == "__main__":
    main()