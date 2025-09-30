"""
Data loading utilities for the hybrid particle tracker
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc
from typing import Tuple, Optional, List


class VelocityDataLoader:
    """Handles loading and preprocessing of velocity field data from CSV files"""
    
    def __init__(self, csv_base_dir: Path, pressure_levels: np.ndarray):
        self.csv_base_dir = Path(csv_base_dir)
        self.pressure_levels = np.sort(pressure_levels)
        
    def load_velocity_field(self, variable: str, hour_index: int, absolute_sim_hour_for_log: Optional[float] = None) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Load velocity field data for a specific variable and hour
        
        Returns:
            Tuple of (lat_coords, lon_coords, pressure_coords, values) or None if failed
        """
        log_hour_str = f"file index {hour_index}"
        if absolute_sim_hour_for_log is not None:
            log_hour_str += f" (abs sim hour {absolute_sim_hour_for_log:.1f})"
        print(f"Loading {variable} data for {log_hour_str}...")

        var_dir = self.csv_base_dir / variable
        all_level_data = []
        unique_lats = None
        unique_lons = None
        
        start_load_time = time.time()
        
        for p_level in self.pressure_levels:
            csv_filename = f"{variable}_{p_level}_{hour_index}.csv"
            file_path = var_dir / csv_filename
            
            if not file_path.exists():
                print(f"ERROR: File not found: {file_path}")
                return None
                
            try:
                # Read only necessary columns
                df = pd.read_csv(file_path, usecols=['Latitude', 'Longitude', variable])
                if df.empty:
                    print(f"WARNING: Empty file encountered: {file_path}")
                    return None
                
                # Check for NaNs specifically in the variable column
                if df[variable].isnull().all(): # If ALL values for the variable are NaN
                    print(f"ERROR: All values for variable '{variable}' are NaN in {file_path}. Treating as load failure.")
                    return None
                elif df[variable].isnull().any(): # If some values are NaN
                    print(f"WARNING: NaN values found in {file_path}. Filling with 0.")
                    df[variable].fillna(0, inplace=True)
                
                # Store data along with pressure level
                df['Pressure'] = p_level
                all_level_data.append(df)
                
                # Get coordinate grids from the first successfully read file
                if unique_lats is None:
                    unique_lats = np.sort(df['Latitude'].unique())  # Sort ascending for C++ compatibility
                    unique_lons = np.sort(df['Longitude'].unique())  # Sort ascending
                    
            except Exception as e:
                print(f"ERROR: Failed to read or process {file_path}: {e}")
                return None
        
        if not all_level_data or unique_lats is None or len(unique_lats) < 2 or len(unique_lons) < 2:
            print(f"ERROR: Could not load sufficient data for {variable} at hour {hour_index}")
            return None
        
        # Combine data from all levels
        full_df = pd.concat(all_level_data, ignore_index=True)
        del all_level_data
        gc.collect()
        
        # Create the 3D grid (lat, lon, pressure)
        velocity_grid = np.full((len(unique_lats), len(unique_lons), len(self.pressure_levels)), np.nan)
        
        try:
            full_df.set_index(['Latitude', 'Longitude', 'Pressure'], inplace=True)
            full_df.sort_index(inplace=True)
            
            # Create lookup dictionaries for faster indexing
            lat_to_idx = {lat: i for i, lat in enumerate(unique_lats)}
            lon_to_idx = {lon: j for j, lon in enumerate(unique_lons)}
            pressure_to_idx = {p: k for k, p in enumerate(self.pressure_levels)}
            
            # Populate grid more efficiently
            for (lat, lon, pressure), row in full_df.iterrows():
                if lat in lat_to_idx and lon in lon_to_idx and pressure in pressure_to_idx:
                    i = lat_to_idx[lat]
                    j = lon_to_idx[lon]
                    k = pressure_to_idx[pressure]
                    velocity_grid[i, j, k] = row[variable]
            
            del full_df
            gc.collect()
            
        except Exception as e:
            print(f"ERROR during grid population: {e}")
            return None
        
        if np.isnan(velocity_grid).any():
            nan_count = np.isnan(velocity_grid).sum()
            print(f"WARNING: Grid for {variable} hour {hour_index} contains {nan_count} NaN values.")
            # Replace NaN with 0 for C++ compatibility
            velocity_grid = np.nan_to_num(velocity_grid, nan=0.0)
        
        load_duration = time.time() - start_load_time
        print(f"Finished loading {variable} for {log_hour_str} in {load_duration:.2f} seconds.")
        
        # Flatten the grid for C++ (C++ expects 1D array in row-major order)
        values_flat = velocity_grid.flatten()

        return unique_lats, unique_lons, self.pressure_levels, values_flat
    
    def load_velocity_fields_for_timestep(self, hour_index: int, absolute_sim_hour_for_log: Optional[float] = None) -> Optional[Tuple]:
        """
        Load all velocity components (u, v, w) for a specific hour
        
        Returns:
            Tuple of (u_data, v_data, w_data) where each is (lats, lons, pressures, values)
        """
        u_data = self.load_velocity_field('u', hour_index, absolute_sim_hour_for_log)
        v_data = self.load_velocity_field('v', hour_index, absolute_sim_hour_for_log)
        w_data = self.load_velocity_field('w', hour_index, absolute_sim_hour_for_log)
        
        if any(data is None for data in [u_data, v_data, w_data]):
            return None
            
        return u_data, v_data, w_data
