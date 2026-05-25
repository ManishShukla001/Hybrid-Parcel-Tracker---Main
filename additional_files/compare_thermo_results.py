import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

def compare_thermo_results():
    dir_temp = Path("./examples/hybrid_particle_output_Temp")
    dir_temp1 = Path("./examples/hybrid_particle_output_Temp1")
    
    # Load the final step files (10:00:00)
    file_temp = dir_temp / "particles_output_20250321_100000.nc"
    file_temp1 = dir_temp1 / "particles_output_20250321_100000.nc"
    
    if not file_temp.exists() or not file_temp1.exists():
        print(f"Error: Could not find output files.")
        return
        
    print("Loading datasets...")
    ds_theta = xr.open_dataset(file_temp)
    ds_t = xr.open_dataset(file_temp1)
    
    # Extract data
    pressure = ds_t.pressure.values
    
    if 'delta_theta' not in ds_theta.data_vars:
        print("\n[ERROR] 'delta_theta' is missing from the Potential_TEMPERATURE output file!")
        print("This usually means the simulation fell back to Python mode because the C++ engine was not updated.")
        print("Please run `pip install -e .` to recompile the C++ core and re-run your simulations.")
        return
        
    if 'delta_t' not in ds_t.data_vars:
        print("\n[ERROR] 'delta_t' is missing from the SIMPLE_SUBTRACTION output file!")
        print("This usually means the simulation fell back to Python mode because the C++ engine was not updated.")
        print("Please run `pip install -e .` to recompile the C++ core and re-run your simulations.")
        return

    delta_theta = ds_theta.delta_theta.values
    delta_t = ds_t.delta_t.values
    
    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'Pressure (hPa)': pressure,
        'Delta Theta (K)': delta_theta,
        'Delta T (K)': delta_t
    })
    
    # Drop NaNs
    df = df.dropna()
    
    print(f"Loaded {len(df)} valid particles for comparison.")
    
    # Define pressure bins
    bins = [1000, 900, 800, 700, 600, 500, 400, 300, 200, 100]
    labels = [f"{bins[i]}-{bins[i-1]}" for i in range(1, len(bins))]
    
    df['Pressure Bin'] = pd.cut(df['Pressure (hPa)'], bins=bins[::-1], labels=labels[::-1])
    
    # Set up the plot
    fig = plt.figure(figsize=(18, 12))
    sns.set_theme(style="whitegrid")
    
    # 1. Vertical Profile of Means with Std Dev (Error Bars)
    ax1 = plt.subplot(2, 2, 1)
    
    # Calculate statistics per bin
    stats = df.groupby('Pressure Bin', observed=False).agg({
        'Delta T (K)': ['mean', 'std'],
        'Delta Theta (K)': ['mean', 'std']
    }).reset_index()
    
    # Create bin centers for plotting
    bin_centers = [(bins[i] + bins[i-1])/2 for i in range(1, len(bins))][::-1]
    
    ax1.errorbar(stats['Delta T (K)']['mean'], bin_centers, xerr=stats['Delta T (K)']['std'], 
                 fmt='-o', color='blue', label=r'Simple Subtraction ($\Delta T$)', capsize=5)
    ax1.errorbar(stats['Delta Theta (K)']['mean'], bin_centers, xerr=stats['Delta Theta (K)']['std'], 
                 fmt='-s', color='red', label=r'Potential Temp ($\Delta \theta$)', capsize=5)
    
    ax1.set_ylim(1000, 100) # Invert Y axis for pressure
    ax1.set_xlabel("Temperature Change (K)", fontsize=12)
    ax1.set_ylabel("Pressure Level (hPa)", fontsize=12)
    ax1.set_title("Vertical Profile of Mean Temperature Changes", fontsize=14, fontweight='bold')
    ax1.legend()
    
    # 2. Scatter plot: Delta T vs Delta Theta colored by Pressure
    ax2 = plt.subplot(2, 2, 2)
    scatter = ax2.scatter(df['Delta T (K)'], df['Delta Theta (K)'], 
                          c=df['Pressure (hPa)'], cmap='viridis_r', alpha=0.5, s=10)
    plt.colorbar(scatter, ax=ax2, label="Pressure (hPa)")
    
    # Add a 1:1 reference line
    min_val = min(df['Delta T (K)'].min(), df['Delta Theta (K)'].min())
    max_val = max(df['Delta T (K)'].max(), df['Delta Theta (K)'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='1:1 Line')
    
    ax2.set_xlabel(r"Simple Subtraction $\Delta T$ (K)", fontsize=12)
    ax2.set_ylabel(r"Potential Temp $\Delta \theta$ (K)", fontsize=12)
    ax2.set_title(r"Particle-level Comparison: $\Delta T$ vs $\Delta \theta$", fontsize=14, fontweight='bold')
    ax2.legend()
    
    # 3. Boxplots showing distributions at each pressure bin
    ax3 = plt.subplot(2, 1, 2)
    
    # Melt dataframe for seaborn boxplot
    df_melted = pd.melt(df, id_vars=['Pressure Bin'], 
                        value_vars=['Delta T (K)', 'Delta Theta (K)'],
                        var_name='Method', value_name='Change (K)')
    
    sns.boxplot(data=df_melted, x='Pressure Bin', y='Change (K)', hue='Method', ax=ax3, palette=['blue', 'red'])
    ax3.set_title("Distribution of Temperature Changes by Pressure Level", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Pressure Bin (hPa)", fontsize=12)
    ax3.set_ylabel("Temperature Change (K)", fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    output_path = Path("./thermo_comparison_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved successfully to: {output_path}")

if __name__ == "__main__":
    compare_thermo_results()
