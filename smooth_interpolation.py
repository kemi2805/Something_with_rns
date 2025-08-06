#!/usr/bin/env python3
"""
Smooth Surface Interpolation for Neutron Star Data
=================================================

Create smooth, continuous surfaces from cleaned neutron star data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial import ConvexHull
import matplotlib.patches as patches
from pathlib import Path

def create_smooth_surface(df: pd.DataFrame, eos: str, 
                         method: str = 'rbf',
                         resolution: int = 200) -> dict:
    """
    Create a smooth interpolated surface for neutron star data.
    
    Args:
        df: Cleaned DataFrame
        eos: EOS name
        method: 'rbf', 'linear', or 'cubic'
        resolution: Grid resolution
    
    Returns:
        Dictionary with grid data for plotting
    """
    eos_df = df[df['eos'] == eos].copy()
    
    if len(eos_df) < 10:
        raise ValueError(f"Not enough data points for {eos}")
    
    # Extract data
    rho = eos_df['rho_c'].values
    M = eos_df['M'].values  
    omega = eos_df['Omega'].values
    
    # Create regular grid
    rho_min, rho_max = rho.min(), rho.max()
    M_min, M_max = M.min(), M.max()
    
    # Add some padding
    rho_range = rho_max - rho_min
    M_range = M_max - M_min
    
    rho_grid = np.linspace(rho_min - 0.05*rho_range, rho_max + 0.05*rho_range, resolution)
    M_grid = np.linspace(M_min - 0.05*M_range, M_max + 0.05*M_range, resolution)
    rho_mesh, M_mesh = np.meshgrid(rho_grid, M_grid)
    
    # Prepare data points
    points = np.column_stack([rho, M])
    grid_points = np.column_stack([rho_mesh.ravel(), M_mesh.ravel()])
    
    if method == 'rbf':
        # Radial Basis Function interpolation (smoothest)
        rbf = RBFInterpolator(points, omega, kernel='thin_plate_spline', smoothing=0.1)
        omega_interp = rbf(grid_points).reshape(rho_mesh.shape)
        
    elif method == 'cubic':
        # Cubic interpolation
        omega_interp = griddata(points, omega, (rho_mesh, M_mesh), 
                               method='cubic', fill_value=np.nan)
    else:
        # Linear interpolation (default)
        omega_interp = griddata(points, omega, (rho_mesh, M_mesh), 
                               method='linear', fill_value=np.nan)
    
    # Create mask for valid region (inside convex hull of data)
    try:
        hull = ConvexHull(points)
        # Check which grid points are inside the convex hull
        from scipy.spatial.qhull import QhullError
        from matplotlib.path import Path as MPath
        
        hull_path = MPath(points[hull.vertices])
        mask = hull_path.contains_points(grid_points).reshape(rho_mesh.shape)
        
        # Apply mask
        omega_interp[~mask] = np.nan
        
    except (QhullError, ValueError):
        # If convex hull fails, just mask extreme outliers
        print("Convex hull failed, using simple masking")
        pass
    
    return {
        'rho_mesh': rho_mesh / 1e15,  # Convert to 10^15 g/cm³
        'M_mesh': M_mesh,
        'omega_mesh': omega_interp,
        'rho_data': rho / 1e15,
        'M_data': M,
        'omega_data': omega,
        'eos': eos
    }

def plot_smooth_surfaces(df_cleaned: pd.DataFrame, 
                        output_dir: str = "./plots",
                        method: str = 'rbf',
                        resolution: int = 200):
    """
    Create smooth surface plots for all EOS.
    
    Args:
        df_cleaned: Cleaned DataFrame
        output_dir: Output directory
        method: Interpolation method
        resolution: Grid resolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    eos_list = df_cleaned['eos'].unique()
    n_eos = len(eos_list)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_eos, figsize=(8*n_eos, 6))
    if n_eos == 1:
        axes = [axes]
    
    for i, eos in enumerate(eos_list):
        ax = axes[i]
        
        try:
            # Create smooth surface
            surface_data = create_smooth_surface(df_cleaned, eos, method, resolution)
            
            # Plot smooth contour/surface
            contour = ax.contourf(
                surface_data['rho_mesh'], 
                surface_data['M_mesh'], 
                surface_data['omega_mesh'],
                levels=50,
                cmap='viridis',
                alpha=0.8
            )
            
            # Overlay original data points
            scatter = ax.scatter(
                surface_data['rho_data'],
                surface_data['M_data'],
                c=surface_data['omega_data'],
                cmap='viridis',
                s=2,
                alpha=0.6,
                edgecolors='white',
                linewidth=0.1
            )
            
            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax)
            cbar.set_label('Ω (×10⁴ s⁻¹)', fontsize=10)
            
            # Labels and title
            ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
            ax.set_ylabel('Gravitational Mass M (M☉)')
            max_mass = df_cleaned[df_cleaned['eos'] == eos]['M'].max()
            ax.set_title(f'{eos}\nMax Mass: {max_mass:.3f} M☉')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Failed to create surface for {eos}: {e}")
            ax.text(0.5, 0.5, f'Failed to interpolate\n{eos}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f"smooth_surfaces_{method}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved smooth surface plot to {plot_path}")
    plt.show()

def plot_3d_surface(df_cleaned: pd.DataFrame, eos: str, 
                   output_dir: str = "./plots",
                   method: str = 'rbf'):
    """Create a 3D surface plot."""
    from mpl_toolkits.mplot3d import Axes3D
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create surface data
    surface_data = create_smooth_surface(df_cleaned, eos, method, resolution=100)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(
        surface_data['rho_mesh'],
        surface_data['M_mesh'], 
        surface_data['omega_mesh'],
        cmap='viridis',
        alpha=0.8,
        antialiased=True
    )
    
    # Plot original data points
    ax.scatter(
        surface_data['rho_data'],
        surface_data['M_data'],
        surface_data['omega_data'],
        c=surface_data['omega_data'],
        cmap='viridis',
        s=20,
        alpha=0.9
    )
    
    # Labels
    ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
    ax.set_ylabel('Gravitational Mass M (M☉)')
    ax.set_zlabel('Angular Velocity Ω (×10⁴ s⁻¹)')
    ax.set_title(f'3D Surface: {eos}')
    
    # Colorbar
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
    
    # Save
    plot_path = output_dir / f"3d_surface_{eos}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved 3D surface plot to {plot_path}")
    plt.show()

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create smooth interpolated surfaces")
    parser.add_argument('data_file', help='Path to cleaned neutron star data')
    parser.add_argument('--method', choices=['rbf', 'linear', 'cubic'], default='rbf',
                       help='Interpolation method')
    parser.add_argument('--resolution', type=int, default=200,
                       help='Grid resolution')
    parser.add_argument('--3d', action='store_true', help='Create 3D surface plots')
    parser.add_argument('--output', '-o', default='./plots',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load cleaned data
    df = pd.read_parquet(args.data_file)
    print(f"Loaded {len(df)} cleaned neutron star models")
    
    # Create smooth surface plots
    plot_smooth_surfaces(df, args.output, args.method, args.resolution)
    
    # Create 3D plots if requested
    if getattr(args, '3d'):
        for eos in df['eos'].unique():
            try:
                plot_3d_surface(df, eos, args.output, args.method)
            except Exception as e:
                print(f"Failed to create 3D plot for {eos}: {e}")