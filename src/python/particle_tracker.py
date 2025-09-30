"""
Main hybrid particle tracker implementation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import time
import gc
from tqdm import tqdm
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Dict, Any
import sys
import xarray as xr
# Import C++ engine (will be available after compilation)
try:
    import particle_engine_cpp
except ImportError:
    print("WARNING: C++ engine not available. Please compile the extension first.")
    particle_engine_cpp = None

try:
    from .data_loader import VelocityDataLoader
    from .visualization import ParticleVisualizer
    from .api_data_handler import APIDataHandler
except ImportError:
    # Fallback for direct execution
    from data_loader import VelocityDataLoader
    from visualization import ParticleVisualizer
    from api_data_handler import APIDataHandler


class HybridParticleTracker:
    """
    High-performance hybrid C++/Python particle tracking system
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the hybrid particle tracker
        
        Args:
            config: Configuration dictionary containing simulation parameters
        """
        self.config = config
        self.validate_config()
        
        # Initialize data handler based on config
        if self.config['data_source'].upper() == "API":
            self.data_handler = APIDataHandler(config)
            # Convert string datetime to datetime object
            try:
                self.simulation_start_datetime_obj = datetime.strptime(
                    config['simulation_start_datetime'], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                raise ValueError("Invalid simulation_start_datetime format. Use YYYY-MM-DD HH:MM:SS")
        elif self.config['data_source'].upper() == "CSV":
            self.data_handler = VelocityDataLoader(
                csv_base_dir=config['csv_base_dir'], pressure_levels=config['pressure_levels'])
        else:
            raise ValueError(f"Unsupported data_source: {self.config['data_source']}")

        self.visualizer = ParticleVisualizer(config['output_dir'])
        
        # Initialize C++ engine if available
        if particle_engine_cpp is not None:
            dt_seconds = config['simulation_step_hours'] * 3600.0
            self.cpp_engine = particle_engine_cpp.ParticleEngine(
                dt_seconds, config['data_interval_hours']
            )
        else:
            self.cpp_engine = None
            print("WARNING: Running in Python-only mode (slower)")
        
        # Simulation state
        self.particles = None
        self.current_step = 0
        self.interpolators = {}
        
        # Setup directories
        self.setup_directories()
    
    def validate_config(self):
        """Validate configuration parameters"""
        required_keys = [
            'csv_base_dir', 'output_dir', 'checkpoint_dir', 'pressure_levels',
            'initialization_lat_range', 'initialization_lon_range', 
            'initialization_pressure_levels', 'initialization_spacing_km',
            'total_simulation_hours', 'data_interval_hours', 'simulation_step_hours',
            'output_interval_hours', 'checkpoint_interval_hours', 'data_source', 'output_format'
            # simulation_start_hour is optional if API mode uses simulation_start_datetime
        ]
        
        if self.config.get('data_source', '').upper() == "API":
            required_keys.extend(['api_data_dir', 'simulation_start_datetime']) 
            # simulation_end_datetime is strongly recommended for API mode but total_simulation_hours can be a fallback
            if 'simulation_end_datetime' not in self.config and 'total_simulation_hours' not in self.config:
                raise ValueError("For API mode, either 'simulation_end_datetime' or 'total_simulation_hours' must be provided.")
        elif self.config.get('data_source', '').upper() == "CSV":
             # csv_base_dir is already in required_keys
            pass # No extra keys for CSV if already covered

        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        output_fmt = self.config['output_format'].upper()
        if output_fmt not in ["CSV", "NETCDF"]:
            raise ValueError(f"Invalid output_format: {self.config['output_format']}. Must be 'CSV' or 'NETCDF'.")
    
    def setup_directories(self):
        """Create necessary directories"""
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)
        Path(self.config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    
    def initialize_particles(self) -> np.ndarray:
        """
        Initialize particles on a regular grid
            
        Returns:
            Array of initialized particles [id, lat, lon, pressure]
        """
        print("Initializing particles...")
        
        lat_range = self.config['initialization_lat_range']
        lon_range = self.config['initialization_lon_range']
        p_levels_config = self.config['initialization_pressure_levels']
        spacing_km = self.config['initialization_spacing_km']
        #print(f"PYTHON: Calling C++ initialize_particles with spacing_km: {spacing_km}") # DEBUG PRINT
        #print(f"PYTHON: Lat range: {lat_range}, Lon range: {lon_range}, Pressure levels: {p_levels_config}") 
               
        if p_levels_config is None or not p_levels_config: # Check if None or empty list
            # Default pressure levels if not specified or empty in config
            p_levels_init = np.arange(1000, 200 - 30, -30) 
        else:
            p_levels_init = np.array(p_levels_config)
        
        if self.cpp_engine is not None:
            # Use C++ engine for initialization
            particles_list = self.cpp_engine.initialize_particles(
                lat_range[0], lat_range[1], lon_range[0], lon_range[1],
                p_levels_init.tolist(), spacing_km
            )
            particles = np.array(particles_list)
        else:
            # Fallback to Python implementation
            particles = self._initialize_particles_python(lat_range, lon_range, p_levels_init, spacing_km)
        
        print(f"Initialized {len(particles)} particles.")
        self.particles = particles
        return particles
    
    def _initialize_particles_python(self, lat_range, lon_range, p_levels_init, spacing_km):
        """Python fallback for particle initialization"""
        lat_start, lat_end = lat_range
        lon_start, lon_end = lon_range
        
        deg_per_km_lat = 1 / 111
        delta_lat = spacing_km * deg_per_km_lat
        
        mid_lat_rad = np.radians((lat_start + lat_end) / 2)
        deg_per_km_lon = 1 / (111 * np.cos(mid_lat_rad))
        delta_lon = spacing_km * deg_per_km_lon
        
        lats = np.arange(lat_start, lat_end + delta_lat, delta_lat)
        lons = np.arange(lon_start, lon_end + delta_lon, delta_lon)
        
        particle_positions = np.array(np.meshgrid(lats, lons, p_levels_init, indexing='ij'))
        particle_positions = particle_positions.T.reshape(-1, 3)
        
        particle_ids = np.arange(1, len(particle_positions) + 1)
        particles = np.hstack((particle_ids.reshape(-1, 1), particle_positions))
        
        return particles
    
    def load_velocity_data(self, absolute_sim_hour: float) -> Optional[Tuple]:
        """
        Load velocity data for a specific absolute simulation hour.
        absolute_sim_hour is hours since the very beginning of the simulation period.
        """
        if self.config['data_source'].upper() == "API":
            target_datetime = self.simulation_start_datetime_obj + timedelta(hours=absolute_sim_hour)
            return self.data_handler.load_fields_for_hour(target_datetime)
        elif self.config['data_source'].upper() == "CSV":
            csv_actual_start_hour = float(self.config.get('simulation_start_hour', 0))
            data_interval_hours = float(self.config['data_interval_hours'])

            # Calculate hours relative to the configured start of CSV data
            hour_relative_to_csv_start = absolute_sim_hour - csv_actual_start_hour

            if hour_relative_to_csv_start < -1e-9: # Allow for small float inaccuracies if absolute_sim_hour is the start
                 print(f"WARNING: Attempting to load data for hour {absolute_sim_hour} which is before CSV start hour {csv_actual_start_hour}. Returning None.")
                 return None

            # Determine the 1-based index for the CSV file
            # Example: if data_interval is 3 hours, and files are _1 (0-2h), _2 (3-5h), _3 (6-8h)
            # hour_relative_to_csv_start = 0.0 -> index_0based = 0 -> file_index = 1
            # hour_relative_to_csv_start = 3.0 -> index_0based = 1 -> file_index = 2
            # hour_relative_to_csv_start = 5.5 -> index_0based = 1 -> file_index = 2
            csv_file_index_0based = int(np.floor(hour_relative_to_csv_start / data_interval_hours + 1e-9)) # Add small epsilon
            csv_file_hour_index = csv_file_index_0based + 1

            
            # VelocityDataLoader only loads u,v,w. We'll return None for q,t for CSV.
            # Pass absolute_sim_hour for logging clarity if desired
            uvw_data = self.data_handler.load_velocity_fields_for_timestep(csv_file_hour_index, absolute_sim_hour_for_log=absolute_sim_hour)
            if uvw_data:
                u_data, v_data, w_data = uvw_data
                return u_data, v_data, w_data, None, None # q_data=None, t_data=None
            else:
                return None
        return None
    
    def create_interpolators(self, velocity_data: Tuple) -> Dict[str, Any]:
        """Create interpolators from velocity data"""
        if velocity_data is None:
            raise ValueError("Cannot create interpolators from None velocity_data.")

        if len(velocity_data) == 5: # u,v,w,q,t (API mode or future CSV with q/t)
            u_data, v_data, w_data, q_data, t_data = velocity_data
        elif len(velocity_data) == 3: # only u,v,w (older CSV mode if load_velocity_fields_for_timestep was called directly)
            u_data, v_data, w_data = velocity_data
            q_data, t_data = None, None
        else:
            raise ValueError("Velocity data tuple has unexpected length.")
        
        # Ensure core data components and their coordinate arrays are present
        if u_data is None or not all(isinstance(arr, np.ndarray) for arr in u_data[:3]):
            raise ValueError("Core u, v, or w data is None, cannot create interpolators.")
        if v_data is None or not all(isinstance(arr, np.ndarray) for arr in v_data[:3]):
            raise ValueError("Core v data or its coordinates are None, cannot create interpolators.")
        if w_data is None or not all(isinstance(arr, np.ndarray) for arr in w_data[:3]):
            raise ValueError("Core w data or its coordinates are None, cannot create interpolators.")

        # Always create scipy interpolators as they work correctly
        from scipy.interpolate import RegularGridInterpolator
        
        # Reshape data for scipy
        # At this point, u_data[0], u_data[1], u_data[2] should be valid numpy arrays
        nlat, nlon, npres = len(u_data[0]), len(u_data[1]), len(u_data[2])

        if u_data[3] is None:
            raise ValueError("u_data values (u_data[3]) are None, cannot reshape for u_grid.")
        u_grid = u_data[3].reshape(nlat, nlon, npres)

        if v_data[3] is None:
            raise ValueError("v_data values (v_data[3]) are None, cannot reshape for v_grid.")
        v_grid = v_data[3].reshape(nlat, nlon, npres)

        if w_data[3] is None:
            raise ValueError("w_data values (w_data[3]) are None, cannot reshape for w_grid.")
        w_grid = w_data[3].reshape(nlat, nlon, npres)

        # Create scipy interpolators (always available as fallback)
        u_interp_scipy = RegularGridInterpolator(
            (u_data[0], u_data[1], u_data[2]), u_grid, bounds_error=False, fill_value=0.0
        )
        v_interp_scipy = RegularGridInterpolator(
            (v_data[0], v_data[1], v_data[2]), v_grid, bounds_error=False, fill_value=0.0
        )
        w_interp_scipy = RegularGridInterpolator(
            (w_data[0], w_data[1], w_data[2]), w_grid, bounds_error=False, fill_value=0.0
        )
        q_interp_scipy = None
        if q_data:
            if q_data[3] is not None: # Check if actual values are present
                q_grid = q_data[3].reshape(nlat, nlon, npres)
                q_interp_scipy = RegularGridInterpolator(
                    (q_data[0], q_data[1], q_data[2]), q_grid, bounds_error=False, fill_value=np.nan
                )

        t_interp_scipy = None
        if t_data:
            if t_data[3] is not None: # Check if actual values are present
                t_grid = t_data[3].reshape(nlat, nlon, npres)
                t_interp_scipy = RegularGridInterpolator(
                    (t_data[0], t_data[1], t_data[2]), t_grid, bounds_error=False, fill_value=np.nan
                )

        result = {
            'u': u_interp_scipy, 'v': v_interp_scipy, 'w': w_interp_scipy,
            'u_scipy': u_interp_scipy, 'v_scipy': v_interp_scipy, 'w_scipy': w_interp_scipy,
            'type': 'scipy'
        }
        
        if q_interp_scipy:
            result['q_scipy'] = q_interp_scipy
        if t_interp_scipy:
            result['t_scipy'] = t_interp_scipy

        # Try to create C++ interpolators if available
        if self.cpp_engine is not None:
            try:
                u_interp_cpp = self.cpp_engine.create_interpolator(
                    u_data[0].tolist(), u_data[1].tolist(), u_data[2].tolist(), u_data[3].tolist()
                )
                v_interp_cpp = self.cpp_engine.create_interpolator(
                    v_data[0].tolist(), v_data[1].tolist(), v_data[2].tolist(), v_data[3].tolist()
                )
                w_interp_cpp = self.cpp_engine.create_interpolator(
                    w_data[0].tolist(), w_data[1].tolist(), w_data[2].tolist(), w_data[3].tolist()
                )
                
                # Test C++ interpolators with a sample point
                test_lat = u_data[0][len(u_data[0])//2]
                test_lon = u_data[1][len(u_data[1])//2]
                test_pressure = u_data[2][len(u_data[2])//2]
                
                test_u = u_interp_cpp.interpolate(test_lat, test_lon, test_pressure)
                test_v = v_interp_cpp.interpolate(test_lat, test_lon, test_pressure)
                
                if not (test_u == 0.0 and test_v == 0.0):
                    # C++ interpolators work, use them for the main interface
                    result.update({
                        'u': u_interp_cpp, 'v': v_interp_cpp, 'w': w_interp_cpp,
                        'u_cpp': u_interp_cpp, 'v_cpp': v_interp_cpp, 'w_cpp': w_interp_cpp,
                        'type': 'cpp'
                    })
                    print("✓ Using C++ interpolators")
                else:
                    print("⚠ C++ interpolators return zeros, using scipy fallback")
                    
            except Exception as e:
                print(f"⚠ Failed to create C++ interpolators: {e}, using scipy fallback")
        
        return result
    
    def update_particles(self, particles: np.ndarray, alpha: float,
                        interp_curr: Dict, interp_next: Dict) -> np.ndarray:
        """Update particle positions using RK4 integration"""
        
        # Check if we have C++ interpolators and C++ engine
        if (self.cpp_engine is not None and
            interp_curr.get('type') == 'cpp' and
            interp_next.get('type') == 'cpp'):
            try:
                # Use C++ engine for fast updates with C++ interpolators
                particles_list = [list(p) for p in particles]
                updated_particles = self.cpp_engine.update_particles(
                    particles_list, alpha,
                    interp_curr['u_cpp'], interp_curr['v_cpp'], interp_curr['w_cpp'],
                    interp_next['u_cpp'], interp_next['v_cpp'], interp_next['w_cpp']
                )
                result = np.array(updated_particles)
                
                # Debug: Check if particles actually moved
                if np.allclose(particles[:5, 1:], result[:5, 1:]):
                    print("WARNING: C++ engine returned unchanged particles - falling back to Python")
                    return self._update_particles_python(particles, alpha, interp_curr, interp_next)
                
                print("✓ Using C++ engine for particle updates")
                return result
            except Exception as e:
                print(f"C++ engine failed: {e}, falling back to Python")
                return self._update_particles_python(particles, alpha, interp_curr, interp_next)
        else:
            # Use Python implementation with scipy interpolators
            print("Using Python fallback for particle updates")
            return self._update_particles_python(particles, alpha, interp_curr, interp_next)
    
    def _update_particles_python(self, particles, alpha, interp_curr, interp_next):
        """Python fallback for particle updates (slower)"""
        print("WARNING: Using Python fallback - performance will be significantly slower")
        
        dt_seconds = self.config['simulation_step_hours'] * 3600.0
        R_earth = 6371000.0  # meters
        min_cos_factor = np.cos(np.radians(85))
        
        # Use scipy interpolators for Python fallback
        u_interp_curr = interp_curr.get('u_scipy', interp_curr['u'])
        v_interp_curr = interp_curr.get('v_scipy', interp_curr['v'])
        w_interp_curr = interp_curr.get('w_scipy', interp_curr['w'])
        u_interp_next = interp_next.get('u_scipy', interp_next['u'])
        v_interp_next = interp_next.get('v_scipy', interp_next['v'])
        w_interp_next = interp_next.get('w_scipy', interp_next['w'])
        
        # Get bounds from scipy interpolator
        if hasattr(u_interp_curr, 'grid'):
            bounds = (
                u_interp_curr.grid[0].min(), u_interp_curr.grid[0].max(),  # lat
                u_interp_curr.grid[1].min(), u_interp_curr.grid[1].max(),  # lon
                u_interp_curr.grid[2].min(), u_interp_curr.grid[2].max()   # pressure
            )
        else:
            # Default bounds
            bounds = (-90, 90, -180, 180, 100, 1100)
        
        def get_velocity(lat, lon, pressure, t_alpha):
            # Clip coordinates to bounds
            clipped_lat = np.clip(lat, bounds[0], bounds[1])
            clipped_lon = np.clip(lon, bounds[2], bounds[3])
            clipped_pressure = np.clip(pressure, bounds[4], bounds[5])
            
            point = np.array([clipped_lat, clipped_lon, clipped_pressure])
            
            # Get velocities from scipy interpolators
            try:
                u_c = u_interp_curr(point)
                v_c = v_interp_curr(point)
                w_c = w_interp_curr(point)
                u_n = u_interp_next(point)
                v_n = v_interp_next(point)
                w_n = w_interp_next(point)
                
                # Handle scalar vs array returns
                if hasattr(u_c, '__len__') and len(u_c) == 1:
                    u_c, v_c, w_c = u_c[0], v_c[0], w_c[0]
                    u_n, v_n, w_n = u_n[0], v_n[0], w_n[0]
                elif hasattr(u_c, '__len__'):
                    u_c, v_c, w_c = float(u_c), float(v_c), float(w_c)
                    u_n, v_n, w_n = float(u_n), float(v_n), float(w_n)
                
                # Temporal interpolation
                u = (1.0 - t_alpha) * u_c + t_alpha * u_n
                v = (1.0 - t_alpha) * v_c + t_alpha * v_n
                w_pa_s = (1.0 - t_alpha) * w_c + t_alpha * w_n
                
                # Check for NaN
                if np.isnan(u) or np.isnan(v) or np.isnan(w_pa_s):
                    return np.array([0.0, 0.0, 0.0])
                
                # Convert to coordinate derivatives
                lat_rad = np.radians(lat)
                cos_lat = np.cos(lat_rad)
                lon_denom = R_earth * max(cos_lat, min_cos_factor)
                
                delta_lat_deg_s = np.degrees(v / R_earth)
                delta_lon_deg_s = np.degrees(u / lon_denom) if lon_denom > 1e-6 else 0.0
                delta_p_hpa_s = w_pa_s / 100.0  # Convert Pa/s to hPa/s
                
                return np.array([delta_lat_deg_s, delta_lon_deg_s, delta_p_hpa_s])
                
            except Exception as e:
                print(f"Velocity interpolation error: {e}")
                return np.array([0.0, 0.0, 0.0])
        
        # Update particles using RK4
        updated_particles = np.zeros_like(particles)
        
        for i, particle in enumerate(particles):
            p_id, lat, lon, pressure = particle
            pos = np.array([lat, lon, pressure])
            
            # RK4 stages
            k1 = get_velocity(pos[0], pos[1], pos[2], alpha)
            
            pos_k2 = pos + 0.5 * dt_seconds * k1
            alpha_k2 = alpha + 0.5 * (dt_seconds / 3600.0 / self.config['data_interval_hours'])
            k2 = get_velocity(pos_k2[0], pos_k2[1], pos_k2[2], alpha_k2)
            
            pos_k3 = pos + 0.5 * dt_seconds * k2
            alpha_k3 = alpha + 0.5 * (dt_seconds / 3600.0 / self.config['data_interval_hours'])
            k3 = get_velocity(pos_k3[0], pos_k3[1], pos_k3[2], alpha_k3)
            
            pos_k4 = pos + dt_seconds * k3
            alpha_k4 = alpha + 1.0 * (dt_seconds / 3600.0 / self.config['data_interval_hours'])
            k4 = get_velocity(pos_k4[0], pos_k4[1], pos_k4[2], alpha_k4)
            
            # Combine RK4 stages
            delta_pos = (dt_seconds / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            new_pos = pos + delta_pos
            
            # Apply global boundaries
            # Latitude clamping
            new_pos[0] = np.clip(new_pos[0], -90.0, 90.0)
            
            # Longitude wrapping to [-180, 180)
            new_pos[1] = (new_pos[1] + 180.0) % 360.0 - 180.0
            # Ensure it's exactly 180 if it was -180 after modulo, or handle edge cases if needed
            if new_pos[1] == -180.0: new_pos[1] = 180.0 
            
            # Pressure clamping (already present)
            new_pos[2] = np.clip(new_pos[2], bounds[4], bounds[5])
            
            updated_particles[i] = [p_id, new_pos[0], new_pos[1], new_pos[2]]
        
        return updated_particles
    
    def save_checkpoint(self, particles: np.ndarray, current_step: int, filename: str = "latest_checkpoint.npz"):
        """Save simulation state to disk"""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        filepath = checkpoint_dir / filename
        
        np.savez(filepath,
                 particles=particles,
                 current_step=current_step,
                 **self.config)
        print(f"Checkpoint saved to {filepath} at step {current_step}")
    
    def load_checkpoint(self, filename: str = "latest_checkpoint.npz") -> Optional[Tuple]:
        """Load saved simulation state"""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        filepath = checkpoint_dir / filename
        
        if not filepath.exists():
            return None
        
        print(f"Loading checkpoint from {filepath}")
        try:
            data = np.load(filepath)
            particles = data['particles']
            current_step = data['current_step'].item()
            return particles, current_step
        except Exception as e:
            print(f"ERROR loading checkpoint {filepath}: {e}")
            return None
    
    def save_particles(self, particles: np.ndarray, time_identifier: Any,
                       q_values: Optional[np.ndarray] = None,
                       t_values: Optional[np.ndarray] = None):
        """Save particle data to CSV or NetCDF format"""
        output_dir = Path(self.config['output_dir'])
        output_format = self.config['output_format'].upper()

        # 'time_identifier' is now correctly used as it's the parameter name
        filename_time_str = f"{time_identifier:04d}" if isinstance(time_identifier, int) else time_identifier.strftime("%Y%m%d_%H%M%S")
        
        if particles.shape[0] == 0:
            print(f"WARNING: No particles to save for time_identifier {filename_time_str} (output format: {output_format}). Skipping file save.")
            return


        if output_format == "CSV":
            filename = output_dir / f'particles_output_{filename_time_str}.csv'
            columns = ['id', 'latitude', 'longitude', 'pressure']
            # Ensure particles array has at least 4 columns before trying to access them
            if particles.shape[1] < 4:
                 raise ValueError(f"Particles array has insufficient columns ({particles.shape[1]}) for standard output.")
            
            data_dict = {
                'id': particles[:, 0],
                'latitude': particles[:, 1],
                'longitude': particles[:, 2],
                'pressure': particles[:, 3]
            }

            if q_values is not None:
                data_dict['specific_humidity'] = q_values
            if t_values is not None:
                data_dict['temperature'] = t_values

            df = pd.DataFrame(data_dict)
            df.to_csv(filename, index=False, float_format='%.5f')
            print(f"Saved particle data to CSV: {filename}")

        elif output_format == "NETCDF":
            filename = output_dir / f'particles_output_{filename_time_str}.nc'
            
            # Ensure data types are NetCDF-friendly and define fill values
            # Using float for _FillValue is standard even if data is float32
            netcdf_fill_value_float = np.nan 

            # Assuming particle IDs can be represented as integers
            particle_ids_coord = particles[:, 0].astype(np.int64)
            lat_data = particles[:, 1].astype(np.float32)
            lon_data = particles[:, 2].astype(np.float32)
            pressure_data = particles[:, 3].astype(np.float32)

            data_vars = {
                'latitude': (('particle_id',), lat_data, {'units': 'degrees_north', 'long_name': 'Latitude'}),
                'longitude': (('particle_id',), lon_data, {'units': 'degrees_east', 'long_name': 'Longitude'}),
                'pressure': (('particle_id',), pressure_data, {'units': 'hPa', 'long_name': 'Pressure Level'}),
            }
            if q_values is not None:
                q_data_typed = q_values.astype(np.float32)
                data_vars['specific_humidity'] = (('particle_id',), q_data_typed, {'units': 'kg kg-1', 'long_name': 'Specific Humidity'})
            if t_values is not None:
                t_data_typed = t_values.astype(np.float32)
                data_vars['temperature'] = (('particle_id',), t_data_typed, {'units': 'K', 'long_name': 'Temperature'})

            coords = {'particle_id': particle_ids_coord}
            if isinstance(time_identifier, datetime):
                # Convert timezone-aware datetime to naive UTC datetime, then to datetime64
                # This makes the timezone explicit (UTC) and then removes it for np.datetime64,
                # which expects naive datetimes or handles them as UTC by convention.
                if time_identifier.tzinfo is not None and time_identifier.tzinfo.utcoffset(time_identifier) is not None:
                    utc_datetime = time_identifier.astimezone(timezone.utc)
                    naive_utc_datetime = utc_datetime.replace(tzinfo=None)
                    time_val = np.datetime64(naive_utc_datetime, 'ns') # Explicitly use nanosecond precision
                else: # If already naive, assume it's UTC as per our internal logic
                    time_val = np.datetime64(time_identifier, 'ns') # Explicitly use nanosecond precision
                
                coords['time'] = (('time',), np.array([time_val])) # Make it a 1D array with a 'time' dimension

            ds = xr.Dataset(data_vars, coords=coords)
            ds.attrs['creation_date'] = datetime.now(timezone.utc).isoformat()
            ds.attrs['source_script'] = __file__

            # Define encoding for variables, especially for _FillValue and dtype
            encoding = {}
            for var_name in ds.data_vars:
                if ds[var_name].dtype == np.float32 or ds[var_name].dtype == np.float64:
                    encoding[var_name] = {'_FillValue': netcdf_fill_value_float, 'dtype': 'float32'}
                else:
                    encoding[var_name] = {'dtype': ds[var_name].dtype.name} # Use existing dtype for non-floats
            
            if 'time' in ds.coords: # Encoding for time coordinate
                encoding['time'] = {'units': 'seconds since 1970-01-01 00:00:00Z', 'calendar': 'proleptic_gregorian', 'dtype': 'int64'}

            try:
                ds.to_netcdf(filename, engine="netcdf4", encoding=encoding)
                print(f"Saved particle data to NetCDF: {filename}")
            except Exception as e_nc_save:
                print(f"ERROR saving NetCDF file {filename}: {e_nc_save}")
                import traceback
                traceback.print_exc()
    
    def run_simulation(self, resume: bool = True):
        """Run the main simulation loop"""
        print("--- Starting Hybrid Particle Simulation ---")
        
        # Try to resume from checkpoint
        sim_step_duration_hours = float(self.config['simulation_step_hours']) # Duration of each simulation step
        total_duration_hours_config = float(self.config['total_simulation_hours']) # Total duration from config

        # Determine effective start time and total duration
        display_start_info: str  # For user messages
        initial_output_naming_hour: int # For naming initial outputs
        if self.config['data_source'].upper() == "API":
            # simulation_start_hour from config is an offset from simulation_start_datetime_obj
            start_offset_hours = float(self.config.get('simulation_start_hour', 0))
            effective_start_datetime = self.simulation_start_datetime_obj + timedelta(hours=start_offset_hours)
            
            if 'simulation_end_datetime' in self.config:
                simulation_end_datetime_obj = datetime.strptime(
                    self.config['simulation_end_datetime'], "%Y-%m-%d %H:%M:%S"
                ).replace(tzinfo=timezone.utc)
                if simulation_end_datetime_obj <= effective_start_datetime:
                    raise ValueError("simulation_end_datetime must be after effective_start_datetime in API mode.")
                total_sim_duration_from_dates = (simulation_end_datetime_obj - effective_start_datetime).total_seconds() / 3600.0
            else:
                total_sim_duration_from_dates = total_duration_hours_config # Fallback to total_simulation_hours
            
            display_start_info = f"datetime {effective_start_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            initial_absolute_step_offset = 0 # Steps are relative to effective_start_datetime
            total_duration_hours = total_sim_duration_from_dates
            initial_output_naming_hour = effective_start_datetime.hour # Hour of the day for initial output
        else: # CSV
            csv_actual_start_hour = float(self.config.get('simulation_start_hour', 0)) # Default to 0 if not set
            display_start_info = f"hour {int(csv_actual_start_hour)}"
            initial_absolute_step_offset = int(csv_actual_start_hour / sim_step_duration_hours)
            total_duration_hours = total_duration_hours_config
            initial_output_naming_hour = int(csv_actual_start_hour)
            effective_start_datetime = None # To be explicit, not used for CSV time logic


        # These will be determined by checkpoint or new simulation settings
        loop_start_step = initial_absolute_step_offset 
        last_saved_output_hour = -float('inf') # Initialize to a very small number
        # last_checkpoint_step is the step number *completed* and saved.
        last_checkpoint_step = initial_absolute_step_offset - 1 

        if resume:
            checkpoint_data = self.load_checkpoint() # Tries to load "latest_checkpoint.npz"
            if checkpoint_data:
                self.particles, saved_completed_step = checkpoint_data
                loop_start_step = saved_completed_step + 1
                last_checkpoint_step = saved_completed_step

                # Determine the absolute hour for which output was last saved
                if saved_completed_step == initial_absolute_step_offset - 1:
                    # Checkpoint is for the state *before* the first step of this configured run.
                    # The "output" corresponding to this is the initial state.
                    if self.config['data_source'].upper() == "API":
                        last_saved_output_hour = effective_start_datetime.timestamp() / 3600.0
                    else: # CSV
                        last_saved_output_hour = float(initial_output_naming_hour) 
                elif saved_completed_step >= 0:
                    # Calculate the absolute time (in hours) at the end of the saved_completed_step
                    hours_elapsed_at_checkpoint_end = (saved_completed_step - initial_absolute_step_offset + 1) * sim_step_duration_hours
                    abs_hour_at_checkpoint_end_val: float
                    if self.config['data_source'].upper() == "API":
                        checkpoint_datetime = effective_start_datetime + timedelta(hours=hours_elapsed_at_checkpoint_end)
                        abs_hour_at_checkpoint_end_val = checkpoint_datetime.timestamp() / 3600.0
                    else: # CSV
                        abs_hour_at_checkpoint_end_val = csv_actual_start_hour + hours_elapsed_at_checkpoint_end
                    # Find the last output interval that was met or passed by this time
                    last_saved_output_hour = (np.floor(abs_hour_at_checkpoint_end_val) // self.config['output_interval_hours']) * self.config['output_interval_hours']
                
                print(f"Resuming from absolute simulation step {loop_start_step}. Last output hour: {last_saved_output_hour}.")
            else:
                print(f"No checkpoint found, starting new simulation from configured {display_start_info}.")
                last_saved_output_hour = (effective_start_datetime.timestamp() / 3600.0) if self.config['data_source'].upper() == "API" else float(initial_output_naming_hour)
        
        # Initialize particles if not loaded from checkpoint
        if self.particles is None:
            self.particles = self.initialize_particles()
            # Initial save corresponds to the effective start hour
            # 'initial_output_naming_hour' is already correctly set for API or CSV
            #self.save_particles(self.particles, initial_output_naming_hour)
            #self.visualizer.plot_particles(self.particles, initial_output_naming_hour, initial_absolute_step_offset)
            initial_time_identifier_for_output: Any
            if self.config['data_source'].upper() == "API":
                initial_time_identifier_for_output = effective_start_datetime
            else: # CSV
                initial_time_identifier_for_output = initial_output_naming_hour
            
            self.save_particles(self.particles, initial_time_identifier_for_output)
            # Pass plot ranges from config
            plot_lat_range_config = self.config.get('plot_lat_range')
            plot_lon_range_config = self.config.get('plot_lon_range')
            self.visualizer.plot_particles(self.particles, initial_time_identifier_for_output, 
                                           initial_absolute_step_offset,
                                           plot_lat_range=plot_lat_range_config,
                                           plot_lon_range=plot_lon_range_config)
            self.save_checkpoint(self.particles, initial_absolute_step_offset - 1) # Checkpoint for state *before* first step
            # last_saved_output_hour and last_checkpoint_step are already set for this new simulation case
            # loop_start_step is already initial_absolute_step_offset
        
        # num_duration_steps is the number of steps for *this specific run*,
        # considering the total_duration_hours from the config.
        num_duration_steps = int(total_duration_hours / sim_step_duration_hours)
        
        # loop_end_step is the absolute step index *after* the last one to be executed for this run.
        loop_end_step = initial_absolute_step_offset + num_duration_steps

        data_interval_steps = int(self.config['data_interval_hours'] / sim_step_duration_hours)
        # output_interval_steps = int(self.config['output_interval_hours'] / sim_step_duration_hours) # Not directly used in loop logic
        checkpoint_interval_steps = int(self.config['checkpoint_interval_hours'] / sim_step_duration_hours)
        
        # Load initial velocity fields
        # These are absolute hours for which data is needed
        # `loop_start_step` is the absolute step index for the simulation.
        # `current_sim_time_at_loop_start` is the absolute simulation hour at the beginning of the first step.
        current_sim_time_at_loop_start = loop_start_step * sim_step_duration_hours

        # Determine the start of the data interval that `current_sim_time_at_loop_start` falls into.
        # This `current_data_interval_start_abs_hour` is an absolute simulation hour.
        current_data_interval_start_abs_hour = np.floor(current_sim_time_at_loop_start / self.config['data_interval_hours']) * self.config['data_interval_hours']

        # These are passed to self.load_velocity_data
        current_data_hour = current_data_interval_start_abs_hour # Absolute simulation hour
        next_data_hour = current_data_hour + self.config['data_interval_hours']
        
        print(f"Loading initial data for hours {current_data_hour} and {next_data_hour}")
        
        curr_data = self.load_velocity_data(current_data_hour)
        next_data = self.load_velocity_data(next_data_hour)
        
        if curr_data is None or next_data is None:
            print("ERROR: Failed to load initial velocity data")
            return
        
        interp_curr = self.create_interpolators(curr_data)
        interp_next = self.create_interpolators(next_data)
        
        # Main simulation loop
        print(f"Starting simulation loop. Absolute steps from {loop_start_step} to {loop_end_step - 1}")
        pbar = tqdm(range(loop_start_step, loop_end_step), desc="Simulating", 
                   initial=loop_start_step, total=loop_end_step)
        
        # `step` is the absolute simulation step index
        for step in pbar:
            # This is hours elapsed *since the effective start of the simulation*
            # (e.g., since effective_start_datetime for API, or since hour 0 for CSV if sim_start_hour_config was 0)
            current_sim_hours_elapsed = (step - initial_absolute_step_offset) * sim_step_duration_hours
            # Update velocity fields when crossing data interval
            if step > 0 and step % data_interval_steps == 0:
                # Shift data
                # The new 'current_data_hour' is the previous 'next_data_hour'
                current_data_hour = next_data_hour 
                # The new 'next_data_hour' is one data interval after that
                next_data_hour = current_data_hour + self.config['data_interval_hours']

                interp_curr = interp_next
                
                # Load new next data
                # Max hour for data loading: start_hour_of_sim + total_duration
                # `max_data_hour_for_sim_run` is the absolute simulation hour marking the end of data needed for this run.
                max_data_hour_for_sim_run: float
                if self.config['data_source'].upper() == "CSV":
                    # For CSV, it's the configured start hour + total duration of this run.
                    max_data_hour_for_sim_run = csv_actual_start_hour + total_duration_hours_config
                else: # API
                    # For API, it's the effective start datetime + total duration of this run.
                    max_data_hour_for_sim_run = (effective_start_datetime + timedelta(hours=total_duration_hours_config)).timestamp() / 3600.0
                if next_data_hour < max_data_hour_for_sim_run: # Use < because next_data_hour is the *start* of an interval
                    next_data = self.load_velocity_data(next_data_hour)
                    if next_data is None:
                        print(f"ERROR: Failed to load data for hour {next_data_hour}")
                        break
                    interp_next = self.create_interpolators(next_data)
                else:
                    # Use current as next for final steps
                    print(f"INFO: Next data hour {next_data_hour} is at or beyond max data hour {max_data_hour_for_sim_run}. Using current data as next.")
                    interp_next = interp_curr
                
                gc.collect()
            
            # Calculate temporal interpolation factor
            # Alpha is relative to the start of the current data interval for interp_curr
            # current_data_hour is the start of the interval (offset for API, abs hour for CSV)
            # current_sim_hours_elapsed is offset from sim start for API, or abs hour for CSV if sim_start_hour_config=0
            time_into_current_data_interval = current_sim_hours_elapsed - current_data_hour
            alpha = np.clip(time_into_current_data_interval / self.config['data_interval_hours'], 0.0, 1.0)
            
            # Update particles
            try:
                self.particles = self.update_particles(self.particles, alpha, interp_curr, interp_next)
            except Exception as e:
                print(f"ERROR during particle update at step {step}: {e}")
                self.save_checkpoint(self.particles, step - 1)
                break
            
            # Output and checkpointing
            # Hours elapsed since the *effective start of the simulation*
            hours_elapsed_since_effective_start = (step - initial_absolute_step_offset + 1) * sim_step_duration_hours

            # Determine current absolute time and the identifier for output naming
            # These are calculated once per step and used for q/t interpolation and output checks.
            
            current_absolute_sim_hour_at_step_end = 0
            time_identifier_for_output: Any # For filenames and plot titles (int for CSV, datetime for API)
            # Calculate alpha for q and t interpolation at the end of the step
            # This alpha is relative to the data interval of interp_curr and interp_next
            # hours_elapsed_since_effective_start is the time at the *end* of the current step
            alpha_for_scalars = np.clip(
                (hours_elapsed_since_effective_start - current_data_hour) / self.config['data_interval_hours'],
                0.0, 1.0
            )

            # Interpolate q and t values
            q_final_values = None
            t_final_values = None

            if interp_curr.get('q_scipy') and interp_next.get('q_scipy'):
                try:
                    q_curr_vals = interp_curr['q_scipy'](self.particles[:, 1:4]) # lat, lon, pressure
                    q_next_vals = interp_next['q_scipy'](self.particles[:, 1:4])
                    q_final_values = (1.0 - alpha_for_scalars) * q_curr_vals + alpha_for_scalars * q_next_vals
                    q_final_values = np.nan_to_num(q_final_values, nan=np.nan) # Keep NaNs if fill_value was NaN
                except Exception as e_q_interp:
                    print(f"WARNING: Error during q interpolation at step {step}: {e_q_interp}. Skipping q values.")
            elif self.config['data_source'].upper() == "API": # Only warn if API mode, as CSV mode doesn't load q/t
                print(f"WARNING: q_scipy interpolator not available at step {step}. Skipping q calculation.")

            if interp_curr.get('t_scipy') and interp_next.get('t_scipy'):
                try:
                    t_curr_vals = interp_curr['t_scipy'](self.particles[:, 1:4])
                    t_next_vals = interp_next['t_scipy'](self.particles[:, 1:4])
                    t_final_values = (1.0 - alpha_for_scalars) * t_curr_vals + alpha_for_scalars * t_next_vals
                    t_final_values = np.nan_to_num(t_final_values, nan=np.nan) # Keep NaNs if fill_value was NaN
                except Exception as e_t_interp:
                    print(f"WARNING: Error during t interpolation at step {step}: {e_t_interp}. Skipping t values.")
            elif self.config['data_source'].upper() == "API":
                print(f"WARNING: t_scipy interpolator not available at step {step}. Skipping t calculation.")

            # Determine time identifier for output naming
            if self.config['data_source'].upper() == "API":
                completed_datetime_obj = effective_start_datetime + timedelta(hours=hours_elapsed_since_effective_start)
                time_identifier_for_output = completed_datetime_obj
                current_absolute_sim_hour_at_step_end = completed_datetime_obj.timestamp() / 3600.0
            else: # CSV
                current_absolute_sim_hour_at_step_end = float(csv_actual_start_hour + hours_elapsed_since_effective_start)
                time_identifier_for_output = int(np.floor(current_absolute_sim_hour_at_step_end))

            # Check if it's time to save output
            if (np.floor(current_absolute_sim_hour_at_step_end) > np.floor(last_saved_output_hour) and 
                int(np.floor(current_absolute_sim_hour_at_step_end)) % self.config['output_interval_hours'] == 0):
                #self.save_particles(self.particles, output_naming_hour) # Use hour of day for API, absolute for CSV for filename
                #self.visualizer.plot_particles(self.particles, output_naming_hour, step + 1) 
                # Pass the appropriate time identifier for naming
                self.save_particles(self.particles, time_identifier_for_output, q_values=q_final_values, t_values=t_final_values) 
                # Plotting with the time identifier for output
                # Pass plot ranges from config
                plot_lat_range_config = self.config.get('plot_lat_range')
                plot_lon_range_config = self.config.get('plot_lon_range')
                self.visualizer.plot_particles(self.particles, time_identifier_for_output, 
                                               step + 1,
                                               plot_lat_range=plot_lat_range_config,
                                               plot_lon_range=plot_lon_range_config)
                last_saved_output_hour = current_absolute_sim_hour_at_step_end # Update with the absolute hour
            
            if (step > last_checkpoint_step and 
                (step + 1) % checkpoint_interval_steps == 0):
                self.save_checkpoint(self.particles, step) # Saves as "latest_checkpoint.npz"
                last_checkpoint_step = step
        
        pbar.close()
        print("--- Simulation Finished ---")
        
        # Save final state
        if 'step' in locals(): # Ensure loop ran at least once
            final_hours_elapsed_since_effective_start = (step - initial_absolute_step_offset + 1) * sim_step_duration_hours
            final_time_identifier_for_output: Any
            # Interpolate q and t for the final state
            final_alpha_for_scalars = np.clip(
                #(final_sim_hours_elapsed - current_data_hour) / self.config['data_interval_hours'], 0.0, 1.0
                (final_hours_elapsed_since_effective_start - current_data_hour) / self.config['data_interval_hours'], 0.0, 1.0
             )
            final_q_values, final_t_values = None, None
            if 'q_scipy' in interp_curr and 'q_scipy' in interp_next and interp_curr['q_scipy'] and interp_next['q_scipy']:
                final_q_values = (1.0 - final_alpha_for_scalars) * interp_curr['q_scipy'](self.particles[:, 1:4]) + \
                                 final_alpha_for_scalars * interp_next['q_scipy'](self.particles[:, 1:4])
                final_q_values = np.nan_to_num(final_q_values, nan=0.0)
            if 't_scipy' in interp_curr and 't_scipy' in interp_next and interp_curr['t_scipy'] and interp_next['t_scipy']:
                final_t_values = (1.0 - final_alpha_for_scalars) * interp_curr['t_scipy'](self.particles[:, 1:4]) + \
                                 final_alpha_for_scalars * interp_next['t_scipy'](self.particles[:, 1:4])
                final_t_values = np.nan_to_num(final_t_values, nan=0.0)

            if self.config['data_source'].upper() == "API":
                final_time_identifier_for_output = (effective_start_datetime + timedelta(hours=final_hours_elapsed_since_effective_start))
            else: # CSV
                final_time_identifier_for_output = int(np.floor(csv_actual_start_hour + final_hours_elapsed_since_effective_start))
            self.save_particles(self.particles, final_time_identifier_for_output, q_values=final_q_values, t_values=final_t_values)
            # Pass plot ranges from config for final plot
            plot_lat_range_config = self.config.get('plot_lat_range')
            plot_lon_range_config = self.config.get('plot_lon_range')
            self.visualizer.plot_particles(self.particles, final_time_identifier_for_output, 
                                           step + 1,
                                           plot_lat_range=plot_lat_range_config,
                                           plot_lon_range=plot_lon_range_config)
            self.save_checkpoint(self.particles, step, f"final_state_step_{step}.npz")
