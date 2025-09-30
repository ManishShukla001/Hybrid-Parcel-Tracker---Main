"""
Visualization utilities for the hybrid particle tracker
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from pathlib import Path
from typing import Optional, Any, Dict


class ParticleVisualizer:
    """Handles plotting and visualization of particle positions"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_particles(self, particles: np.ndarray, time_identifier: Any, step_index: int, 
                      plot_lat_range: Optional[tuple] = None, plot_lon_range: Optional[tuple] = None,
                      save_plot: bool = True, default_lat_range: tuple = (-90, 90), default_lon_range: tuple = (-180, 180)) -> Optional[str]:
        """
        Create and save a plot of particle positions
        
        Args:
            particles: Array of particle data [id, lat, lon, pressure]
            time_identifier: Current simulation time identifier (int hour for CSV, datetime for API)
            step_index: Current step index
            plot_lat_range: Optional latitude range for plot extent (min_lat, max_lat).
            plot_lon_range: Optional longitude range for plot extent (min_lon, max_lon).
            default_lat_range: Default latitude range if plot_lat_range is None.
            default_lon_range: Default longitude range if plot_lon_range is None.
            save_plot: Whether to save the plot to file
            
        Returns:
            Filename of saved plot or None if not saved
        """
        time_str_for_title = f"{time_identifier:04d}" if isinstance(time_identifier, int) else time_identifier.strftime("%Y-%m-%d %H:%M")
        print(f"Plotting particles for time {time_str_for_title}...")
        
        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        # Add basic gridlines for context
        ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        
        # Set extent
        lat_min, lat_max = plot_lat_range if plot_lat_range else default_lat_range
        lon_min, lon_max = plot_lon_range if plot_lon_range else default_lon_range
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Plot particles
        if particles.shape[0] > 0:
            lons = particles[:, 2]
            lats = particles[:, 1]
            ax.scatter(lons, lats, s=1, c='red', alpha=0.5, transform=ccrs.PlateCarree())
        else:
            print("Warning: No particles to plot.")
        
        plt.title(f'Particle Positions at {time_str_for_title}')
        
        filename = None
        if save_plot:
            filename_time_str = f"{time_identifier:04d}" if isinstance(time_identifier, int) else time_identifier.strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f'plot_output_{filename_time_str}.png'
            try:
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"Saved plot: {filename}")
            except Exception as e:
                print(f"ERROR saving plot {filename}: {e}")
                filename = None
        
        plt.close()  # Ensure figure is closed
        return str(filename) if filename else None
    
    def plot_particle_trajectories(self, trajectory_data: Dict[Any, np.ndarray], time_identifier: Any,
                                 plot_lat_range: Optional[tuple] = None, plot_lon_range: Optional[tuple] = None,
                                 max_trajectories: int = 100,
                                 default_lat_range: tuple = (-90, 90), default_lon_range: tuple = (-180, 180)) -> Optional[str]:
        """
        Plot particle trajectories over time
        
        Args:
            trajectory_data: Dictionary with particle IDs as keys and position arrays as values
            time_identifier: Current simulation time identifier
            lat_range: Latitude range for plot extent
            lon_range: Longitude range for plot extent
            plot_lat_range: Optional latitude range for plot extent (min_lat, max_lat).
            plot_lon_range: Optional longitude range for plot extent (min_lon, max_lon).
            max_trajectories: Maximum number of trajectories to plot
            default_lat_range: Default latitude range if plot_lat_range is None.
            default_lon_range: Default longitude range if plot_lon_range is None.
            
        Returns:
            Filename of saved plot or None if failed
        """
        time_str_for_title = f"{time_identifier:04d}" if isinstance(time_identifier, int) else time_identifier.strftime("%Y-%m-%d %H:%M")
        print(f"Plotting particle trajectories for time {time_str_for_title}...")
        
        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Add basic gridlines
        ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        
        # Set extent
        lat_min, lat_max = plot_lat_range if plot_lat_range else default_lat_range
        lon_min, lon_max = plot_lon_range if plot_lon_range else default_lon_range
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Plot trajectories
        particle_ids = list(trajectory_data.keys())[:max_trajectories]
        
        for particle_id in particle_ids:
            positions = trajectory_data[particle_id]
            if len(positions) > 1:
                lats = [pos[1] for pos in positions]
                lons = [pos[2] for pos in positions]
                ax.plot(lons, lats, linewidth=0.5, alpha=0.7, transform=ccrs.PlateCarree())
        
        plt.title(f'Particle Trajectories up to {time_str_for_title}')
        
        filename_time_str = f"{time_identifier:04d}" if isinstance(time_identifier, int) else time_identifier.strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f'trajectories_output_{filename_time_str}.png'
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved trajectory plot: {filename}")
        except Exception as e:
            print(f"ERROR saving trajectory plot {filename}: {e}")
            filename = None
        
        plt.close()
        return str(filename) if filename else None
    
    def plot_pressure_distribution(self, particles: np.ndarray, time_identifier: Any) -> Optional[str]:
        """
        Plot histogram of particle pressure distribution
        
        Args:
            particles: Array of particle data [id, lat, lon, pressure]
            time_identifier: Current simulation time identifier
            
        Returns:
            Filename of saved plot or None if failed
        """
        time_str_for_title = f"{time_identifier:04d}" if isinstance(time_identifier, int) else time_identifier.strftime("%Y-%m-%d %H:%M")
        if particles.shape[0] == 0:
            print(f"Warning: No particles to plot pressure distribution for time {time_str_for_title}.")
            return None
            
        plt.figure(figsize=(10, 6))
        pressures = particles[:, 3]
        
        plt.hist(pressures, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Pressure (hPa)')
        plt.ylabel('Number of Particles')
        plt.title(f'Particle Pressure Distribution at {time_str_for_title}')
        plt.grid(True, alpha=0.3)
        
        filename_time_str = f"{time_identifier:04d}" if isinstance(time_identifier, int) else time_identifier.strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f'pressure_dist_output_{filename_time_str}.png'
        try:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved pressure distribution plot: {filename}")
        except Exception as e:
            print(f"ERROR saving pressure distribution plot {filename}: {e}")
            filename = None
        
        plt.close()
        return str(filename) if filename else None