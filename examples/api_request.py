#!/usr/bin/env python3
"""
Script to pre-download daily ERA5 data using the CDS API.

Downloads two categories of data:
  1. Pressure-level variables  → one file per day  (api_daily_YYYYMMDD.nc)
  2. Single-level  variables   → one file per day per variable
                                 (api_daily_YYYYMMDD_<varshortname>_sl.nc)
"""

import cdsapi
from pathlib import Path
from datetime import datetime, timedelta
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("./api_files")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Pressure levels ─────────────────────────────────────────────────────────
API_PRESSURE_LEVELS = sorted([
    50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450,
    500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925,
    950, 975, 1000
])

# Native ERA5 grid resolution [lat, lon]
ERA5_GRID_RESOLUTION_API = [0.25, 0.25]

# ── API client settings ──────────────────────────────────────────────────────
API_RETRY_COUNT        = 3
API_RETRY_DELAY_SECONDS = 120

# ── Simulation scope ─────────────────────────────────────────────────────────
SIMULATION_SCOPE = 'REGIONAL'   # 'GLOBAL' or 'REGIONAL'
# (South_Lat, North_Lat, West_Lon, East_Lon)  →  CDS area: [N, W, S, E]
REGIONAL_EXTENT = (-15, 60, 0, 140)

# ── Date range (inclusive start, exclusive end) ──────────────────────────────
SIM_START_DATE = datetime(2023, 12, 7)
SIM_END_DATE   = datetime(2023, 12, 22)

# ---------------------------------------------------------------------------
# Pressure-level variable definitions
# ---------------------------------------------------------------------------
# Short-name → CDS API long name
PRESSURE_LEVEL_VAR_MAP = {
    # ── Dynamics / thermodynamics ─────────────────────────────────────────
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'w': 'vertical_velocity',
    'q': 'specific_humidity',
    't': 'temperature',
    'z': 'geopotential',               # Geopotential on pressure levels

    # ── Cloud / hydrometeor water content ────────────────────────────────
    #    These collectively describe precipitable water & ice/snow content
    #    throughout the atmospheric column.
    'clwc': 'specific_cloud_liquid_water_content',   # Cloud liquid water
    'ciwc': 'specific_cloud_ice_water_content',      # Cloud ice water
    'crwc': 'specific_rain_water_content',           # Rain water
    'cswc': 'specific_snow_water_content',           # Snow water
    'cc'  : 'fraction_of_cloud_cover',               # Cloud fraction
}

# Which pressure-level short names to actually request
MODEL_VARIABLES_SHORT = ['u', 'v', 'w', 'q', 't', 'z',
                          'clwc', 'ciwc', 'crwc', 'cswc', 'cc']

# ---------------------------------------------------------------------------
# Single-level variable definitions
# ---------------------------------------------------------------------------
# Each entry: short_name → { 'api_name': ..., 'description': ... }
# These will be downloaded from 'reanalysis-era5-single-levels' as
# separate daily files named:  api_daily_YYYYMMDD_<short_name>_sl.nc
SINGLE_LEVEL_VAR_MAP = {
    # ── Precipitation ─────────────────────────────────────────────────────
    'tp'  : {
        'api_name'   : 'total_precipitation',
        'description': 'Total precipitation (liquid + solid)',
    },
    'cp'  : {
        'api_name'   : 'convective_precipitation',
        'description': 'Convective precipitation',
    },
    'lsp' : {
        'api_name'   : 'large_scale_precipitation',
        'description': 'Large-scale (stratiform) precipitation',
    },
    'sf'  : {
        'api_name'   : 'snowfall',
        'description': 'Snowfall (water equivalent)',
    },
    'lsrr': {
        'api_name'   : 'large_scale_rain_rate',
        'description': 'Large-scale rain rate',
    },
    'crr' : {
        'api_name'   : 'convective_rain_rate',
        'description': 'Convective rain rate',
    },

    # ── Column-integrated water / ice ─────────────────────────────────────
    'tcwv': {
        'api_name'   : 'total_column_water_vapour',
        'description': 'Precipitable water (total column water vapour)',
    },
    'tciw': {
        'api_name'   : 'total_column_cloud_ice_water',
        'description': 'Total column cloud ice water',
    },
    'tclw': {
        'api_name'   : 'total_column_cloud_liquid_water',
        'description': 'Total column cloud liquid water',
    },
    'tcrw': {
        'api_name'   : 'total_column_rain_water',
        'description': 'Total column rain water',
    },
    'tcsw': {
        'api_name'   : 'total_column_snow_water',
        'description': 'Total column snow water',
    },

    # ── Surface / near-surface ────────────────────────────────────────────
    'z_sfc': {
        'api_name'   : 'geopotential',
        'description': 'Surface geopotential (orography)',
    },
    'msl'  : {
        'api_name'   : 'mean_sea_level_pressure',
        'description': 'Mean sea-level pressure',
    },
    'sp'   : {
        'api_name'   : 'surface_pressure',
        'description': 'Surface pressure',
    },
    '2t'   : {
        'api_name'   : '2m_temperature',
        'description': '2-metre temperature',
    },
    'sd'   : {
        'api_name'   : 'snow_depth',
        'description': 'Snow depth (water equivalent)',
    },
    'tcc'  : {
        'api_name'   : 'total_cloud_cover',
        'description': 'Total cloud cover',
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_area_request() -> list:
    """Return the CDS API area list [N, W, S, E] based on scope settings."""
    if SIMULATION_SCOPE.upper() == 'REGIONAL':
        # REGIONAL_EXTENT = (South, North, West, East)
        return [REGIONAL_EXTENT[1], REGIONAL_EXTENT[2],
                REGIONAL_EXTENT[0], REGIONAL_EXTENT[3]]
    return [90, -180, -90, 180]   # global


def get_api_variables(short_names: list, var_map: dict) -> list:
    """Map short variable names to their CDS API long names."""
    api_vars = []
    for name in short_names:
        if name in var_map:
            entry = var_map[name]
            # Pressure-level map values are plain strings; single-level are dicts
            api_vars.append(entry if isinstance(entry, str) else entry['api_name'])
        else:
            print(f"  Warning: '{name}' not found in variable map. Skipping.")
    return api_vars


# ---------------------------------------------------------------------------
# Pressure-level downloader  (one file per day, all variables)
# ---------------------------------------------------------------------------

def download_pressure_level_data(target_date: datetime.date,
                                  client: cdsapi.Client) -> bool:
    """
    Download all pressure-level ERA5 variables for *target_date*.
    Output: DATA_DIR/api_daily_YYYYMMDD.nc   (same structure as before)
    """
    filepath = DATA_DIR / f"api_daily_{target_date.strftime('%Y%m%d')}.nc"

    if filepath.exists():
        print(f"  [PL] Already exists: {filepath.name}")
        return True

    print(f"  [PL] Downloading pressure-level data → {filepath.name} …")

    api_vars = get_api_variables(MODEL_VARIABLES_SHORT, PRESSURE_LEVEL_VAR_MAP)
    if not api_vars:
        print("  [PL] ERROR: No valid variables to download. Aborting.")
        return False

    request = {
        "product_type": "reanalysis",
        "format"      : "netcdf",
        "variable"    : api_vars,
        "pressure_level": [str(p) for p in API_PRESSURE_LEVELS],
        "year"        : str(target_date.year),
        "month"       : f"{target_date.month:02d}",
        "day"         : f"{target_date.day:02d}",
        "time"        : [f"{h:02d}:00" for h in range(24)],
        "area"        : _build_area_request(),
    }

    return _retrieve(client, "reanalysis-era5-pressure-levels", request, filepath, "[PL]")


# ---------------------------------------------------------------------------
# Single-level downloader  (one file per day per variable)
# ---------------------------------------------------------------------------

def download_single_level_data(target_date: datetime.date,
                                client: cdsapi.Client) -> dict:
    """
    Download each single-level variable for *target_date* into its own file.
    Output: DATA_DIR/api_daily_YYYYMMDD_<short_name>_sl.nc

    Returns a dict {short_name: True/False} with per-variable success flags.
    """
    results = {}
    area    = _build_area_request()

    for short_name, meta in SINGLE_LEVEL_VAR_MAP.items():
        filepath = DATA_DIR / f"api_daily_{target_date.strftime('%Y%m%d')}_{short_name}_sl.nc"

        if filepath.exists():
            print(f"  [SL] Already exists: {filepath.name}")
            results[short_name] = True
            continue

        print(f"  [SL] Downloading '{short_name}' ({meta['description']}) → {filepath.name} …")

        request = {
            "product_type": "reanalysis",
            "format"      : "netcdf",
            "variable"    : meta['api_name'],
            "year"        : str(target_date.year),
            "month"       : f"{target_date.month:02d}",
            "day"         : f"{target_date.day:02d}",
            "time"        : [f"{h:02d}:00" for h in range(24)],
            "area"        : area,
        }

        results[short_name] = _retrieve(
            client, "reanalysis-era5-single-levels", request, filepath, f"[SL:{short_name}]"
        )

    return results


# ---------------------------------------------------------------------------
# Generic retrieval wrapper with cleanup on failure
# ---------------------------------------------------------------------------

def _retrieve(client: cdsapi.Client, dataset: str,
              request: dict, filepath: Path, tag: str) -> bool:
    """Call client.retrieve(); clean up partial file on failure."""
    try:
        client.retrieve(dataset, request, str(filepath))
        print(f"  {tag} ✓ Saved → {filepath.name}")
        return True
    except Exception as exc:
        print(f"  {tag} ERROR: {exc}")
        if filepath.exists():
            try:
                os.remove(filepath)
                print(f"  {tag} Removed partial file: {filepath.name}")
            except OSError as oe:
                print(f"  {tag} Could not remove partial file: {oe}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  ERA5 Data Pre-downloader")
    print("=" * 60)
    print(f"Output directory : {DATA_DIR.resolve()}")
    print(f"Date range       : {SIM_START_DATE:%Y-%m-%d} → "
          f"{(SIM_END_DATE - timedelta(days=1)):%Y-%m-%d}")
    print(f"Scope            : {SIMULATION_SCOPE}"
          f"{f'  {REGIONAL_EXTENT}' if SIMULATION_SCOPE.upper() == 'REGIONAL' else ''}")
    print()
    print("Pressure-level variables :", MODEL_VARIABLES_SHORT)
    print("Pressure levels          :", API_PRESSURE_LEVELS)
    print()
    print("Single-level variables   :", list(SINGLE_LEVEL_VAR_MAP.keys()))
    print("=" * 60)

    cds_client = cdsapi.Client(
        retry_max=API_RETRY_COUNT,
        sleep_max=API_RETRY_DELAY_SECONDS,
        quiet=False,
    )

    current_date = SIM_START_DATE
    while current_date < SIM_END_DATE:
        date_str = current_date.strftime('%Y-%m-%d')
        print(f"\n{'─' * 60}")
        print(f"  Processing date: {date_str}")
        print(f"{'─' * 60}")

        # 1 ── Pressure-level file  (all variables, one file)
        download_pressure_level_data(current_date.date(), cds_client)

        # 2 ── Single-level files  (one file per variable)
        sl_results = download_single_level_data(current_date.date(), cds_client)

        # Summary for this date
        failed_sl = [k for k, v in sl_results.items() if not v]
        if failed_sl:
            print(f"  !! Single-level variables that FAILED for {date_str}: {failed_sl}")

        current_date += timedelta(days=1)

    print("\n" + "=" * 60)
    print("  Pre-download process finished.")
    print("=" * 60)


if __name__ == "__main__":
    main()
