# test_simulation_consistency.py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import time
import os
import pandas as pd # For saving detailed CSV output
import sys
from pathlib import Path # Use Path for consistency

# Add src/python to path for direct imports
# Assumes this script is in the project root directory (parent of 'src')
current_script_dir = Path(__file__).resolve().parent
python_src_dir_path = current_script_dir / "src" / "python"

if str(python_src_dir_path) not in sys.path:
    sys.path.insert(0, str(python_src_dir_path))

try:
    from data_loader import VelocityDataLoader
    # Removed Config import, parameters will be defined locally
    import particle_engine_cpp
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the script is run from a location where 'src' is accessible,")
    print("and the C++ extension 'particle_engine_cpp' is compiled and installed/discoverable.")
    sys.exit(1)

# --- Constants for Scipy RK4 (mirroring C++ rk4_integrator.cpp) ---
R_EARTH_PY = 6371000.0  # meters
MIN_COS_FACTOR_PY = np.cos(np.deg2rad(85.0)) # cos(85 degrees)
LON_DENOM_THRESHOLD_PY = 1e-6 # Threshold for longitude calculation

# --- Helper functions for Scipy-based simulation ---

def get_raw_velocity_scipy(lat, lon, pressure, alpha,
                           u_curr_scipy, v_curr_scipy, w_curr_scipy,
                           u_next_scipy, v_next_scipy, w_next_scipy,
                           bounds):
    """
    Gets temporally interpolated u, v, w (m/s, m/s, Pa/s) using Scipy interpolators.
    Coordinates are clipped to bounds before interpolation.
    bounds: [lat_min, lat_max, lon_min, lon_max, pres_min, pres_max]
    """
    clipped_lat = np.clip(lat, bounds[0], bounds[1])
    clipped_lon = np.clip(lon, bounds[2], bounds[3])
    clipped_pressure = np.clip(pressure, bounds[4], bounds[5])
    
    query_point = (clipped_lat, clipped_lon, clipped_pressure)

    # Handle cases where interpolator returns a 0-D array (fill_value)
    raw_u_c = u_curr_scipy(query_point)
    u_c = raw_u_c[0] if raw_u_c.ndim > 0 else float(raw_u_c)

    raw_v_c = v_curr_scipy(query_point)
    v_c = raw_v_c[0] if raw_v_c.ndim > 0 else float(raw_v_c)

    raw_w_c = w_curr_scipy(query_point)
    w_c = raw_w_c[0] if raw_w_c.ndim > 0 else float(raw_w_c)

    raw_u_n = u_next_scipy(query_point)
    u_n = raw_u_n[0] if raw_u_n.ndim > 0 else float(raw_u_n)

    raw_v_n = v_next_scipy(query_point)
    v_n = raw_v_n[0] if raw_v_n.ndim > 0 else float(raw_v_n)

    raw_w_n = w_next_scipy(query_point)
    w_n = raw_w_n[0] if raw_w_n.ndim > 0 else float(raw_w_n)

    u = (1.0 - alpha) * u_c + alpha * u_n
    v = (1.0 - alpha) * v_c + alpha * v_n
    w_pa_s = (1.0 - alpha) * w_c + alpha * w_n

    # Mimic C++ behavior: if interpolated values are NaN, return 0s.
    # This assumes fill_value in Scipy interpolators might not be 0 or might be NaN itself.
    if np.isnan(u) or np.isnan(v) or np.isnan(w_pa_s):
        return 0.0, 0.0, 0.0
    return u, v, w_pa_s

def get_velocity_derivatives_scipy(lat, lon, pressure, alpha,
                                   u_curr_scipy, v_curr_scipy, w_curr_scipy,
                                   u_next_scipy, v_next_scipy, w_next_scipy,
                                   bounds):
    """
    Calculates dlat/dt (deg/s), dlon/dt (deg/s), dp/dt (hPa/s) using Scipy.
    """
    u, v, w_pa_s = get_raw_velocity_scipy(lat, lon, pressure, alpha,
                                          u_curr_scipy, v_curr_scipy, w_curr_scipy,
                                          u_next_scipy, v_next_scipy, w_next_scipy,
                                          bounds)
    
    # If raw velocities are all zero (e.g., due to NaN handling or actual zero wind)
    if u == 0.0 and v == 0.0 and w_pa_s == 0.0:
        return 0.0, 0.0, 0.0

    lat_rad = np.deg2rad(lat)
    cos_lat = np.cos(lat_rad)
    safe_cos_lat = np.maximum(cos_lat, MIN_COS_FACTOR_PY)
    lon_denom = R_EARTH_PY * safe_cos_lat

    delta_lat_deg_s = v / R_EARTH_PY * 180.0 / np.pi
    delta_lon_deg_s = (u / lon_denom * 180.0 / np.pi) if lon_denom > LON_DENOM_THRESHOLD_PY else 0.0
    delta_p_hpa_s = w_pa_s / 100.0 # Convert Pa/s to hPa/s

    return delta_lat_deg_s, delta_lon_deg_s, delta_p_hpa_s

def rk4_step_scipy(particle_state, dt_seconds, initial_alpha,
                   u_curr_scipy, v_curr_scipy, w_curr_scipy,
                   u_next_scipy, v_next_scipy, w_next_scipy,
                   bounds, data_interval_hours_local):
    """
    Performs one RK4 step for a single particle using Scipy interpolators.
    particle_state: [id, lat, lon, pressure]
    """
    pid, lat, lon, pressure = particle_state
    pos = np.array([lat, lon, pressure], dtype=np.float64)

    # RK4 stages
    alpha_k1 = initial_alpha
    k1 = np.array(get_velocity_derivatives_scipy(pos[0], pos[1], pos[2], alpha_k1,
                                                 u_curr_scipy, v_curr_scipy, w_curr_scipy,
                                                 u_next_scipy, v_next_scipy, w_next_scipy, bounds), dtype=np.float64)

    pos_k2 = pos + 0.5 * dt_seconds * k1
    alpha_k2 = initial_alpha + 0.5 * (dt_seconds / (data_interval_hours_local * 3600.0))
    k2 = np.array(get_velocity_derivatives_scipy(pos_k2[0], pos_k2[1], pos_k2[2], alpha_k2,
                                                 u_curr_scipy, v_curr_scipy, w_curr_scipy,
                                                 u_next_scipy, v_next_scipy, w_next_scipy, bounds), dtype=np.float64)

    pos_k3 = pos + 0.5 * dt_seconds * k2
    alpha_k3 = initial_alpha + 0.5 * (dt_seconds / (data_interval_hours_local * 3600.0)) # Same as alpha_k2
    k3 = np.array(get_velocity_derivatives_scipy(pos_k3[0], pos_k3[1], pos_k3[2], alpha_k3,
                                                 u_curr_scipy, v_curr_scipy, w_curr_scipy,
                                                 u_next_scipy, v_next_scipy, w_next_scipy, bounds), dtype=np.float64)

    pos_k4 = pos + dt_seconds * k3
    alpha_k4 = initial_alpha + 1.0 * (dt_seconds / (data_interval_hours_local * 3600.0))
    k4 = np.array(get_velocity_derivatives_scipy(pos_k4[0], pos_k4[1], pos_k4[2], alpha_k4,
                                                 u_curr_scipy, v_curr_scipy, w_curr_scipy,
                                                 u_next_scipy, v_next_scipy, w_next_scipy, bounds), dtype=np.float64)

    delta_pos = (dt_seconds / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)
    new_pos = pos + delta_pos
    
    # Apply pressure bounds (consistent with C++ RK4Integrator)
    new_pos[2] = np.clip(new_pos[2], bounds[4], bounds[5]) # pressure is at index 2
    
    return np.array([pid, new_pos[0], new_pos[1], new_pos[2]])


def run_simulation_test():
    print("=== Testing Simulation Consistency (C++ vs Scipy) ===")

    # --- Configuration Parameters (defined locally) ---
    DATA_DIR = Path("D:/Manish/ERA_Data/Data/era5_csv_output")
    PRESSURE_LEVELS = np.array([
        200, 250, 300, 350, 400, 450, 500,
        550, 600, 650, 700, 750, 775, 800, 825,
        850, 875, 900, 925, 950, 975, 1000
    ], dtype=int)
    FILL_VALUE = 0.0  # Fill value for interpolators
    DATA_INTERVAL_HOURS = 1.0 # How often new velocity data files are available (e.g., every 1 hour)

    # --- Simulation Parameters ---
    num_target_particles = 50 
    num_timesteps_loop = 5 # Simulate for 5 time steps
    dt_seconds = 30 * 60.0 # Each time step is 30 minutes

    # --- 1. Initialize Particle Engine ---
    cpp_engine = particle_engine_cpp.ParticleEngine(dt_seconds, DATA_INTERVAL_HOURS)
    print(f"✓ C++ ParticleEngine initialized (dt={dt_seconds}s, data_interval={DATA_INTERVAL_HOURS}h)")

    # --- 2. Load Velocity Data ---
    print("Loading velocity data...")
    data_loader = VelocityDataLoader(DATA_DIR, PRESSURE_LEVELS)
    all_u_data, all_v_data, all_w_data = [], [], []

    # Calculate how many hourly data files are needed
    total_simulation_duration_hours = num_timesteps_loop * dt_seconds / 3600.0
    # Number of distinct 1-hour data intervals the simulation spans
    num_distinct_data_intervals_spanned = np.ceil(total_simulation_duration_hours / DATA_INTERVAL_HOURS)
    # We need one more data file than the number of intervals for interpolation at the end
    num_data_files_to_load = int(num_distinct_data_intervals_spanned + 1)

    print(f"Total simulation: {total_simulation_duration_hours}h. Needing {num_data_files_to_load} hourly data files.")

    for hour_file_idx in range(1, num_data_files_to_load + 1):
        print(f"  Loading data for hour_file_idx {hour_file_idx}...")
        start_time = time.time()
        u_data = data_loader.load_velocity_field('u', hour_file_idx)
        v_data = data_loader.load_velocity_field('v', hour_file_idx)
        w_data = data_loader.load_velocity_field('w', hour_file_idx)
        all_u_data.append(u_data)
        all_v_data.append(v_data)
        all_w_data.append(w_data)
        print(f"  Finished loading hour_file_idx {hour_file_idx} in {time.time() - start_time:.2f} seconds.")
    print("✓ Velocity data loaded.")

    lats_coords = all_u_data[0][0]
    lons_coords = all_u_data[0][1]
    press_coords = all_u_data[0][2]
    nlat, nlon, npres = len(lats_coords), len(lons_coords), len(press_coords)
    
    # --- 3. Create Interpolators for all time steps ---
    print("Creating interpolators for all time steps...")
    cpp_interpolators_u, cpp_interpolators_v, cpp_interpolators_w = [], [], []
    scipy_interpolators_u, scipy_interpolators_v, scipy_interpolators_w = [], [], []

    for i in range(num_data_files_to_load): # Iterate up to the number of loaded files
        cpp_u = cpp_engine.create_interpolator(lats_coords.tolist(), lons_coords.tolist(), press_coords.tolist(), all_u_data[i][3].tolist(), FILL_VALUE)
        cpp_v = cpp_engine.create_interpolator(lats_coords.tolist(), lons_coords.tolist(), press_coords.tolist(), all_v_data[i][3].tolist(), FILL_VALUE)
        cpp_w = cpp_engine.create_interpolator(lats_coords.tolist(), lons_coords.tolist(), press_coords.tolist(), all_w_data[i][3].tolist(), FILL_VALUE)
        cpp_interpolators_u.append(cpp_u)
        cpp_interpolators_v.append(cpp_v)
        cpp_interpolators_w.append(cpp_w)

        u_grid = all_u_data[i][3].reshape(nlat, nlon, npres)
        v_grid = all_v_data[i][3].reshape(nlat, nlon, npres)
        w_grid = all_w_data[i][3].reshape(nlat, nlon, npres)

        scipy_u = RegularGridInterpolator((lats_coords, lons_coords, press_coords), u_grid, bounds_error=False, fill_value=FILL_VALUE, method="linear")
        scipy_v = RegularGridInterpolator((lats_coords, lons_coords, press_coords), v_grid, bounds_error=False, fill_value=FILL_VALUE, method="linear")
        scipy_w = RegularGridInterpolator((lats_coords, lons_coords, press_coords), w_grid, bounds_error=False, fill_value=FILL_VALUE, method="linear")
        scipy_interpolators_u.append(scipy_u)
        scipy_interpolators_v.append(scipy_v)
        scipy_interpolators_w.append(scipy_w)
    print("✓ Interpolators created.")

    # --- 4. Initialize Particles ---
    print(f"Initializing {num_target_particles} random particles away from domain edges...")
    
    margin_cells = 3 # Number of grid cells from each edge to exclude for initialization

    if len(lats_coords) > 2 * margin_cells:
        # lats_coords are sorted ascending
        safe_lat_min, safe_lat_max = lats_coords[margin_cells], lats_coords[-(margin_cells + 1)]
    else: 
        safe_lat_min, safe_lat_max = lats_coords.min(), lats_coords.max()

    if len(lons_coords) > 2 * margin_cells:
        # lons_coords are sorted ascending
        safe_lon_min, safe_lon_max = lons_coords[margin_cells], lons_coords[-(margin_cells + 1)]
    else:
        safe_lon_min, safe_lon_max = lons_coords.min(), lons_coords.max()

    if len(press_coords) > 2 * margin_cells:
        # press_coords are sorted ascending
        safe_p_min, safe_p_max = press_coords[margin_cells], press_coords[-(margin_cells + 1)]
    else:
        safe_p_min, safe_p_max = press_coords.min(), press_coords.max()

    # Ensure min < max for random.uniform after applying margin
    if safe_lat_min >= safe_lat_max: safe_lat_max = safe_lat_min + 1e-3 # Add small epsilon if equal
    if safe_lon_min >= safe_lon_max: safe_lon_max = safe_lon_min + 1e-3
    if safe_p_min >= safe_p_max: safe_p_max = safe_p_min + 1e-3

    print(f"  Safe init domain: Lat [{safe_lat_min:.2f}, {safe_lat_max:.2f}], "
          f"Lon [{safe_lon_min:.2f}, {safe_lon_max:.2f}], "
          f"Pres [{safe_p_min:.0f}, {safe_p_max:.0f}]")

    initial_lats = np.random.uniform(safe_lat_min, safe_lat_max, num_target_particles)
    initial_lons = np.random.uniform(safe_lon_min, safe_lon_max, num_target_particles)
    initial_pressures = np.random.uniform(safe_p_min, safe_p_max, num_target_particles)
    particle_ids = np.arange(num_target_particles, dtype=np.float64)

    particles_cpp_list = []
    for i in range(num_target_particles):
        particles_cpp_list.append([particle_ids[i], initial_lats[i], initial_lons[i], initial_pressures[i]])
    particles_scipy_arr = np.array(particles_cpp_list, dtype=np.float64)
    actual_num_particles = len(particles_cpp_list)

    if actual_num_particles == 0:
        print(f"!!! Failed to initialize random particles. Target was {num_target_particles}. !!!")
        return
    print(f"✓ Initialized and selected {actual_num_particles} particles.")

    bounds_from_cpp = np.array(cpp_interpolators_u[0].get_bounds())

    # --- 5. Simulation Loop ---
    all_particle_data_for_csv = [] # To store data for CSV export

    print("\nStarting simulation...")
    for t_step in range(num_timesteps_loop):
        print(f"\n--- Time Step {t_step + 1}/{num_timesteps_loop} ---")

        current_sim_time_start_of_step = t_step * dt_seconds
        # Determine which pair of hourly data files to use
        current_data_slot_index = int(current_sim_time_start_of_step / (DATA_INTERVAL_HOURS * 3600.0))

        u_curr_cpp, v_curr_cpp, w_curr_cpp = cpp_interpolators_u[current_data_slot_index], cpp_interpolators_v[current_data_slot_index], cpp_interpolators_w[current_data_slot_index]
        u_next_cpp, v_next_cpp, w_next_cpp = cpp_interpolators_u[current_data_slot_index+1], cpp_interpolators_v[current_data_slot_index+1], cpp_interpolators_w[current_data_slot_index+1]

        u_curr_scipy, v_curr_scipy, w_curr_scipy = scipy_interpolators_u[current_data_slot_index], scipy_interpolators_v[current_data_slot_index], scipy_interpolators_w[current_data_slot_index]
        u_next_scipy, v_next_scipy, w_next_scipy = scipy_interpolators_u[current_data_slot_index+1], scipy_interpolators_v[current_data_slot_index+1], scipy_interpolators_w[current_data_slot_index+1]

        # Calculate alpha at the beginning of this dt_seconds step
        time_into_current_data_interval = current_sim_time_start_of_step % (DATA_INTERVAL_HOURS * 3600.0)
        alpha_initial_step = time_into_current_data_interval / (DATA_INTERVAL_HOURS * 3600.0)

        # ** C++ Path **
        new_particles_cpp_list = cpp_engine.update_particles(
            particles_cpp_list, alpha_initial_step,
            u_curr_cpp, v_curr_cpp, w_curr_cpp, u_next_cpp, v_next_cpp, w_next_cpp
        )
        new_particles_cpp_arr = np.array(new_particles_cpp_list, dtype=np.float64)

        # ** Scipy Path **
        new_particles_scipy_list = []
        for p_idx in range(actual_num_particles):
            new_p_scipy = rk4_step_scipy(particles_scipy_arr[p_idx], dt_seconds, alpha_initial_step,
                                         u_curr_scipy, v_curr_scipy, w_curr_scipy,
                                         u_next_scipy, v_next_scipy, w_next_scipy,
                                         bounds_from_cpp, DATA_INTERVAL_HOURS)
            new_particles_scipy_list.append(new_p_scipy)
        new_particles_scipy_arr = np.array(new_particles_scipy_list, dtype=np.float64)

        # --- Calculate Velocities (u,v,w) at new positions ---
        alpha_at_new_time = alpha_initial_step + dt_seconds / (DATA_INTERVAL_HOURS * 3600.0)
        
        velocities_cpp_uvw_list = []
        for i in range(actual_num_particles):
            p_coords = (new_particles_cpp_arr[i, 1], new_particles_cpp_arr[i, 2], new_particles_cpp_arr[i, 3])
            # Use get_raw_velocity logic for C++ interpolators
            u_c = u_curr_cpp.interpolate(p_coords[0], p_coords[1], p_coords[2])
            v_c = v_curr_cpp.interpolate(p_coords[0], p_coords[1], p_coords[2])
            w_c = w_curr_cpp.interpolate(p_coords[0], p_coords[1], p_coords[2])
            u_n = u_next_cpp.interpolate(p_coords[0], p_coords[1], p_coords[2])
            v_n = v_next_cpp.interpolate(p_coords[0], p_coords[1], p_coords[2])
            w_n = w_next_cpp.interpolate(p_coords[0], p_coords[1], p_coords[2])
            
            u = (1.0 - alpha_at_new_time) * u_c + alpha_at_new_time * u_n
            v = (1.0 - alpha_at_new_time) * v_c + alpha_at_new_time * v_n
            w_pa_s = (1.0 - alpha_at_new_time) * w_c + alpha_at_new_time * w_n
            if np.isnan(u) or np.isnan(v) or np.isnan(w_pa_s): # Safety, though C++ interpolate handles fill_value
                 u,v,w_pa_s = 0.0,0.0,0.0
            velocities_cpp_uvw_list.append([u, v, w_pa_s])
        velocities_cpp_uvw_arr = np.array(velocities_cpp_uvw_list, dtype=np.float64)

        velocities_scipy_uvw_list = []
        for i in range(actual_num_particles):
            u_s, v_s, w_s_pa_s = get_raw_velocity_scipy(
                new_particles_scipy_arr[i, 1], new_particles_scipy_arr[i, 2], new_particles_scipy_arr[i, 3],
                alpha_at_new_time,
                u_curr_scipy, v_curr_scipy, w_curr_scipy,
                u_next_scipy, v_next_scipy, w_next_scipy,
                bounds_from_cpp
            )
            velocities_scipy_uvw_list.append([u_s, v_s, w_s_pa_s])
        velocities_scipy_uvw_arr = np.array(velocities_scipy_uvw_list, dtype=np.float64)

        # --- Compare Results ---
        pos_diff = new_particles_cpp_arr[:, 1:] - new_particles_scipy_arr[:, 1:]
        abs_pos_diff = np.abs(pos_diff)
        
        print("Position Differences (C++ - Scipy):")
        print(f"  Lat Diff (deg): Max={np.max(abs_pos_diff[:, 0]):.6e}, Mean={np.mean(abs_pos_diff[:, 0]):.6e}")
        print(f"  Lon Diff (deg): Max={np.max(abs_pos_diff[:, 1]):.6e}, Mean={np.mean(abs_pos_diff[:, 1]):.6e}")
        print(f"  Prs Diff (hPa): Max={np.max(abs_pos_diff[:, 2]):.6e}, Mean={np.mean(abs_pos_diff[:, 2]):.6e}")

        # Calculate vel_diff before it's used in the loop below
        vel_diff = velocities_cpp_uvw_arr - velocities_scipy_uvw_arr

        # Store detailed data for CSV
        current_step_data_list = []
        for i in range(actual_num_particles):
            current_step_data_list.append({
                'timestep': t_step + 1,
                'particle_id': int(new_particles_cpp_arr[i, 0]),
                'cpp_lat': new_particles_cpp_arr[i, 1],
                'cpp_lon': new_particles_cpp_arr[i, 2],
                'cpp_pressure': new_particles_cpp_arr[i, 3],
                'scipy_lat': new_particles_scipy_arr[i, 1],
                'scipy_lon': new_particles_scipy_arr[i, 2],
                'scipy_pressure': new_particles_scipy_arr[i, 3],
                'diff_lat': pos_diff[i, 0], # C++ - Scipy
                'diff_lon': pos_diff[i, 1],
                'diff_pressure': pos_diff[i, 2],
                'cpp_U': velocities_cpp_uvw_arr[i, 0],
                'cpp_V': velocities_cpp_uvw_arr[i, 1],
                'cpp_W': velocities_cpp_uvw_arr[i, 2],
                'scipy_U': velocities_scipy_uvw_arr[i, 0],
                'scipy_V': velocities_scipy_uvw_arr[i, 1],
                'scipy_W': velocities_scipy_uvw_arr[i, 2],
                'diff_U': vel_diff[i, 0], # C++ - Scipy
                'diff_V': vel_diff[i, 1],
                'diff_W': vel_diff[i, 2],
            })
        all_particle_data_for_csv.append(pd.DataFrame(current_step_data_list))

        abs_vel_diff = np.abs(vel_diff)

        print("Velocity Differences (C++ - Scipy) at new positions:")
        print(f"  U Diff (m/s):   Max={np.max(abs_vel_diff[:, 0]):.6e}, Mean={np.mean(abs_vel_diff[:, 0]):.6e}")
        print(f"  V Diff (m/s):   Max={np.max(abs_vel_diff[:, 1]):.6e}, Mean={np.mean(abs_vel_diff[:, 1]):.6e}")
        print(f"  W Diff (Pa/s):  Max={np.max(abs_vel_diff[:, 2]):.6e}, Mean={np.mean(abs_vel_diff[:, 2]):.6e}")
        
        particles_cpp_list = new_particles_cpp_list
        particles_scipy_arr = new_particles_scipy_arr
        
        if np.any(np.isnan(new_particles_cpp_arr)) or np.any(np.isnan(new_particles_scipy_arr)):
            print("!!! NaN detected in particle positions. Stopping. !!!")
            break
        if np.any(np.isnan(velocities_cpp_uvw_arr)) or np.any(np.isnan(velocities_scipy_uvw_arr)):
            print("!!! NaN detected in velocities. Stopping. !!!")
            break
            
    print("\n=== Simulation Test Finished ===")
    
    # Save detailed CSV output
    if all_particle_data_for_csv:
        final_df = pd.concat(all_particle_data_for_csv, ignore_index=True)
        output_csv_dir = current_script_dir / "output_data" # Save in an 'output_data' subdirectory
        output_csv_dir.mkdir(parents=True, exist_ok=True)
        output_csv_path = output_csv_dir / "simulation_consistency_details.csv"
        final_df.to_csv(output_csv_path, index=False, float_format='%.8e')
        print(f"Saved detailed particle data to {output_csv_path}")

if __name__ == "__main__":
    run_simulation_test()
