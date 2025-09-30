import xarray as xr
import pandas as pd
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
# import time # For timing, if needed

# --- Configuration (remains the same) ---
FILE_CONFIG = {
    'u': {
        'filename': 'u_data.nc',
        'variable_name': 'UGRD_prl',
        'output_folder': 'u',
        'output_column': 'u'
    }
}

MIN_PLEV_HPA = 100
MAX_PLEV_HPA = 1000
G_STANDARD = 9.80665
G_APPROX = 9.81
R_AIR = 287.058
T_ASSUMED_K = 250.0
RHO_ASSUMED_SEA_LEVEL = 1.225
MAX_LONGITUDE_TO_EXPORT = 120.0 # New constant for longitude filtering

# --- Worker Function (Modified for Longitude Filtering) ---
def process_slice_to_csv(component_char, config_dict, time_idx, plev_hpa, input_nc_file_path, var_name_in_nc):
    output_folder_path = config_dict['output_folder']
    output_col_name_csv = config_dict['output_column']

    try:
        with xr.open_dataset(input_nc_file_path) as ds_worker:
            data_var_worker = ds_worker[var_name_in_nc]
            data_slice = data_var_worker.sel(plevel=plev_hpa).isel(time=time_idx)

            # Optional: Explicitly load if Dask issues persist (as discussed before)
            # data_slice = data_slice.load()

            df = data_slice.to_dataframe(name='temp_value_col')
            df.reset_index(inplace=True)
            df.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude'}, inplace=True)

            # --- NEW: Filter DataFrame by Longitude ---
            df = df[df['Longitude'] <= MAX_LONGITUDE_TO_EXPORT]

            # If after filtering, the DataFrame is empty, don't save an empty CSV
            if df.empty:
                return f"Skipped saving: No data for {component_char}, Time_idx:{time_idx}, Plev:{plev_hpa} with Longitude <= {MAX_LONGITUDE_TO_EXPORT}"

            if component_char == 'w':
                w_m_s = df['temp_value_col']
                df[output_col_name_csv] = - (RHO_ASSUMED_SEA_LEVEL * G_APPROX * w_m_s)
                # --- PREVIOUS, MORE ACCURATE METHOD (COMMENTED OUT) ---
                # P_pa = plev_hpa * 100.0
                # denominator = R_AIR * T_ASSUMED_K
                # if denominator == 0:
                #     df[output_col_name_csv] = w_m_s
                # else:
                #     df[output_col_name_csv] = - (P_pa / denominator) * G_STANDARD * w_m_s
            else:
                df.rename(columns={'temp_value_col': output_col_name_csv}, inplace=True)

            columns_to_save = ['Latitude', 'Longitude', output_col_name_csv]
            df_to_save = df[columns_to_save]

            time_step_for_filename = time_idx + 1
            csv_filename = f"{component_char}_{int(plev_hpa)}_{time_step_for_filename}.csv"
            csv_filepath = os.path.join(output_folder_path, csv_filename)

            df_to_save.to_csv(csv_filepath, index=False, na_rep='NaN')
            return f"Successfully processed and saved: {csv_filepath}"

    except Exception as e:
        return f"ERROR processing slice ({component_char}, Time_idx:{time_idx}, Plev:{plev_hpa}): {str(e)}"

# --- Main Script (Parallelized - unchanged) ---
def process_data_parallel():
    num_workers = max(1, os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1)
    print(f"Using {num_workers} worker processes for parallel tasks.")

    for component_char, config in FILE_CONFIG.items():
        input_nc_file = config['filename']
        var_name = config['variable_name']
        output_folder = config['output_folder']

        print(f"\nProcessing component: {component_char.upper()}")
        print(f"Input file: {input_nc_file}, Variable: {var_name}")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")

        try:
            with xr.open_dataset(input_nc_file) as ds_main:
                # Filter pressure levels
                plevels_to_process_da = ds_main.plevel.sel(plevel=slice(MIN_PLEV_HPA, MAX_PLEV_HPA))
                if plevels_to_process_da.size == 0:
                    print(f"  No pressure levels found in the range {MIN_PLEV_HPA}-{MAX_PLEV_HPA} hPa for {component_char}. Skipping.")
                    continue
                plevels_to_process_values = plevels_to_process_da.values.tolist()
                print(f"  Selected pressure levels (hPa) for {component_char}: {plevels_to_process_values}")

                time_coords = ds_main.time.values
                num_time_steps = len(time_coords)

                # --- Modification for Longitude filtering in main metadata check ---
                # Although the actual filtering happens in the worker,
                # we can pre-filter the longitude coordinates in the main dataset
                # if we want to avoid processing slices that will be entirely empty.
                # However, given your structure, the data selection is by time and plevel,
                # and longitude filtering is applied *after* extracting the 2D (lat, lon) slice.
                # An alternative here would be to use ds_main.longitude.sel(longitude=slice(None, MAX_LONGITUDE_TO_EXPORT))
                # when extracting the data_slice in the worker, but filtering the DataFrame is simpler.

        except FileNotFoundError:
            print(f"ERROR: File not found: {input_nc_file}. Skipping this component.")
            continue
        except Exception as e:
            print(f"ERROR: Could not open {input_nc_file} to get metadata: {e}. Skipping this component.")
            continue

        tasks_for_pool = []
        for time_idx in range(num_time_steps):
            for plev_hpa_val in plevels_to_process_values:
                plev_hpa = float(plev_hpa_val)
                task_args = (component_char, config, time_idx, plev_hpa, input_nc_file, var_name)
                tasks_for_pool.append(task_args)
        
        if not tasks_for_pool:
            print(f"  No tasks generated for {component_char.upper()}. Skipping parallel processing for this component.")
            continue

        print(f"  Submitting {len(tasks_for_pool)} tasks for {component_char.upper()} to the process pool...")
        processed_count = 0
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task_details = {executor.submit(process_slice_to_csv, *args): args for args in tasks_for_pool}
            
            for future in as_completed(future_to_task_details):
                task_details_for_log = future_to_task_details[future]
                try:
                    result_message = future.result()
                    processed_count += 1
                    if "ERROR" in result_message:
                        print(f"  Task Result (Error): {result_message}")
                    elif "Skipped saving" in result_message: # Catch our new skip message
                        # print(f"  Task Result (Skipped): {result_message}") # Can be verbose
                        pass # Or just silently acknowledge
                    
                    if processed_count % 50 == 0 or processed_count == len(tasks_for_pool):
                        print(f"  Progress for {component_char.upper()}: {processed_count}/{len(tasks_for_pool)} tasks completed.")
                except Exception as exc:
                    processed_count += 1
                    brief_args = (task_details_for_log[0], task_details_for_log[2], task_details_for_log[3])
                    print(f"  Critical error processing task for args {brief_args}: {exc}")
        
        print(f"Finished processing component: {component_char.upper()}")

    print("\nAll processing complete.")


# --- Dummy File Creation (unchanged) ---
if __name__ == "__main__":
    def create_dummy_nc_file(filename, var_name_nc, dims_order, coords_data_dict):
        if os.path.exists(filename):
            return
        print(f"Creating dummy file: {filename} for testing purposes.")
        data_shape = tuple(len(coords_data_dict[dim_name]) for dim_name in dims_order)
        dummy_array_data = np.random.rand(*data_shape).astype(np.float32)
        # Introduce some NaNs in dummy data for testing longitude filter effect
        # For dummy data, let's make some longitudes > 120 actually NaN
        if 'longitude' in coords_data_dict:
            lon_coord = coords_data_dict['longitude']
            lon_idx = dims_order.index('longitude')
            for i, lon_val in enumerate(lon_coord):
                if lon_val > MAX_LONGITUDE_TO_EXPORT:
                    # Create a slice object for numpy indexing
                    nan_slice = [slice(None)] * len(data_shape)
                    nan_slice[lon_idx] = i
                    dummy_array_data[tuple(nan_slice)] = np.nan


        coords_for_ds = {name: (name, data) for name, data in coords_data_dict.items()}
        data_vars_for_ds = { var_name_nc: (dims_order, dummy_array_data) }
        ds = xr.Dataset(data_vars_for_ds, coords=coords_for_ds)
        ds.attrs['CDI'] = "Climate Data Interface dummy version for testing"
        ds.attrs['Conventions'] = "COARDS_dummy"
        ds.attrs['history'] = "Created by dummy file generator for testing"
        if 'plevel' in ds.coords:
            current_plevels = ds['plevel'].values.tolist()
            plevels_extended = [10.0, 50.0] + current_plevels + [1050.0, 1100.0, 9.999e+20]
            ds = ds.assign_coords(plevel=sorted(list(set(plevels_extended))))
        ds.to_netcdf(filename, format='NETCDF4')
        print(f"Dummy file {filename} created with variable {var_name_nc}.")
        ds.close()

    # Adjusted dummy longitudes to include values > 120 for testing the filter
    times_dummy = pd.to_datetime(['2023-12-15T00:00:00', '2023-12-15T03:00:00']) # Fewer for quicker test
    longitudes_dummy = np.array([30.0, 60.0, 90.0, 119.9, 120.0, 120.1, 125.0]) # Test boundary
    latitudes_dummy = np.linspace(-15, -14, 3)
    plevels_hpa_dummy = [100.0, 200.0, 1000.0] # Fewer for quicker test

    common_coords_data_dummy = {
        'time': times_dummy, 'longitude': longitudes_dummy,
        'latitude': latitudes_dummy, 'plevel': plevels_hpa_dummy
    }
    data_dims_order_dummy = ('time', 'plevel', 'latitude', 'longitude') # As per example

    print("--- Setting up dummy files for testing (if actual files are not present) ---")
    create_dummy_nc_file('u_data.nc', 'UGRD_prl', data_dims_order_dummy, common_coords_data_dummy)
    create_dummy_nc_file('v_data.nc', 'VGRD_prl', data_dims_order_dummy, common_coords_data_dummy)
    create_dummy_nc_file('w_data.nc', 'DZDT_prl', data_dims_order_dummy, common_coords_data_dummy)
    print("--- Dummy file setup attempt complete ---")

    process_data_parallel()