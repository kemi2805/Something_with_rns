#!/usr/bin/env python3
"""
Clean Boundary Masking with Direct Function Evaluation
======================================================

Returns boundary functions directly and checks each point individually.
Much safer and more transparent than bulk array operations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator, interp1d, CubicSpline, PchipInterpolator
from scipy.spatial import ConvexHull
import matplotlib.patches as patches
from pathlib import Path

class CleanBoundaryInterpolator:
    """Clean interpolation with direct boundary function evaluation."""
    
    def __init__(self, df: pd.DataFrame, eos: str):
        self.df = df
        self.eos = eos
        self.eos_df = df[df['eos'] == eos].copy()
        
        if len(self.eos_df) < 10:
            raise ValueError(f"Not enough data points for {eos}: {len(self.eos_df)}")
        
        # Scale data for numerical stability
        self.rho_raw = self.eos_df['rho_c'].values
        self.M = self.eos_df['M'].values
        self.omega = self.eos_df['Omega'].values
        
        self.rho = self.rho_raw / 1e15  # Scale to reasonable units
        
        print(f"\n{eos}: {len(self.eos_df)} points")
        print(f"  rho range: {self.rho.min():.3f} - {self.rho.max():.3f} ×10¹⁵ g/cm³")
        print(f"  M range: {self.M.min():.3f} - {self.M.max():.3f} M☉")
        print(f"  Ω range: {self.omega.min():.3f} - {self.omega.max():.3f} ×10⁴ s⁻¹")
        
        # Store boundary functions as instance variables
        self.lower_bound_func = None
        self.upper_bound_func = None
        self.boundary_data = None
        
    def create_boundary_functions(self, boundary_method: str = 'hermite'):
        """
        Create boundary functions that can be called for any density value.
        Returns the functions directly, not interpolated arrays.
        """
        
        print(f"  Creating boundary functions using {boundary_method} interpolation")
        
        # Get unique density values and find min/max mass at each density
        unique_rho = np.unique(self.rho)
        
        lower_boundary_rho = []
        lower_boundary_M = []
        upper_boundary_rho = []
        upper_boundary_M = []
        
        for rho_val in unique_rho:
            # Find all masses at this density
            at_density = self.M[self.rho == rho_val]
            
            if len(at_density) > 0:
                lower_boundary_rho.append(rho_val)
                lower_boundary_M.append(at_density.min())
                
                upper_boundary_rho.append(rho_val)
                upper_boundary_M.append(at_density.max())
        
        # Convert to arrays and sort
        lower_boundary_rho = np.array(lower_boundary_rho)
        lower_boundary_M = np.array(lower_boundary_M)
        upper_boundary_rho = np.array(upper_boundary_rho)
        upper_boundary_M = np.array(upper_boundary_M)
        
        lower_sort = np.argsort(lower_boundary_rho)
        lower_boundary_rho = lower_boundary_rho[lower_sort]
        lower_boundary_M = lower_boundary_M[lower_sort]
        
        upper_sort = np.argsort(upper_boundary_rho)
        upper_boundary_rho = upper_boundary_rho[upper_sort]
        upper_boundary_M = upper_boundary_M[upper_sort]
        
        print(f"  Found {len(lower_boundary_rho)} boundary points")
        
        # Create the actual interpolation functions
        try:
            if boundary_method == 'hermite' and len(lower_boundary_rho) >= 3:
                print("  Using PCHIP Hermite interpolators...")
                lower_func = PchipInterpolator(lower_boundary_rho, lower_boundary_M, extrapolate=True)
                upper_func = PchipInterpolator(upper_boundary_rho, upper_boundary_M, extrapolate=True)
            elif boundary_method == 'cubic' and len(lower_boundary_rho) >= 4:
                print("  Using cubic spline interpolators...")
                lower_func = CubicSpline(lower_boundary_rho, lower_boundary_M, bc_type='natural', extrapolate=True)
                upper_func = CubicSpline(upper_boundary_rho, upper_boundary_M, bc_type='natural', extrapolate=True)
            else:
                print("  Using linear interpolators...")
                lower_func = interp1d(lower_boundary_rho, lower_boundary_M, kind='linear', bounds_error=False, fill_value='extrapolate')
                upper_func = interp1d(upper_boundary_rho, upper_boundary_M, kind='linear', bounds_error=False, fill_value='extrapolate')
        except Exception as e:
            print(f"  Interpolation failed: {e}, using linear")
            lower_func = interp1d(lower_boundary_rho, lower_boundary_M, kind='linear', bounds_error=False, fill_value='extrapolate')
            upper_func = interp1d(upper_boundary_rho, upper_boundary_M, kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Store as instance variables
        self.lower_bound_func = lower_func
        self.upper_bound_func = upper_func
        self.boundary_data = {
            'lower_rho': lower_boundary_rho,
            'lower_M': lower_boundary_M,
            'upper_rho': upper_boundary_rho,
            'upper_M': upper_boundary_M
        }
        
        print(f"  ✓ Boundary functions created successfully")
        return lower_func, upper_func
    
    def is_point_valid(self, rho: float, M: float, margin: float = 0.05) -> bool:
        """
        Check if a single point (rho, M) is within the valid boundaries.
        
        Args:
            rho: Density value (in scaled units)
            M: Mass value  
            margin: Safety margin as fraction of total mass range
            
        Returns:
            bool: True if point is valid, False otherwise
        """
        
        if self.lower_bound_func is None or self.upper_bound_func is None:
            raise ValueError("Boundary functions not created. Call create_boundary_functions() first.")
        
        try:
            # Get boundary values at this density
            M_lower = self.lower_bound_func(rho)
            M_upper = self.upper_bound_func(rho)
            
            # Add safety margin
            M_range = self.M.max() - self.M.min()
            margin_size = margin * M_range
            
            M_lower_margin = M_lower - margin_size
            M_upper_margin = M_upper + margin_size
            
            # Simple check: is mass between boundaries?
            return M_lower_margin <= M <= M_upper_margin
            
        except Exception as e:
            print(f"Warning: Boundary evaluation failed for rho={rho}, M={M}: {e}")
            return False
    
    def create_clean_mask(self, grid_rho: np.ndarray, grid_M: np.ndarray, 
                         margin: float = 0.05, use_convex_hull: bool = False):
        """
        Create mask by checking each grid point individually.
        Much safer than bulk array operations.
        """
        
        if self.lower_bound_func is None or self.upper_bound_func is None:
            print("  Creating boundary functions...")
            self.create_boundary_functions()
        
        print(f"  Checking {grid_rho.size} grid points individually...")
        
        # Get original shape
        original_shape = grid_rho.shape
        
        # Flatten for easier iteration
        rho_flat = grid_rho.ravel()
        M_flat = grid_M.ravel()
        
        # Check each point individually
        mask_flat = np.zeros(len(rho_flat), dtype=bool)
        
        for i in range(len(rho_flat)):
            mask_flat[i] = self.is_point_valid(rho_flat[i], M_flat[i], margin)
            
            # Progress indicator for large grids
            if i % 10000 == 0 and i > 0:
                print(f"    Checked {i}/{len(rho_flat)} points ({i/len(rho_flat)*100:.1f}%)")
        
        # Reshape back to original shape
        physical_mask = mask_flat.reshape(original_shape)
        
        # Apply convex hull if requested
        if use_convex_hull:
            print("  Applying convex hull mask...")
            hull_mask = self.create_convex_hull_mask(grid_rho, grid_M)
            combined_mask = physical_mask & hull_mask
        else:
            combined_mask = physical_mask
        
        valid_points = np.sum(combined_mask)
        total_points = combined_mask.size
        
        print(f"  ✓ Valid points: {valid_points}/{total_points} ({valid_points/total_points*100:.1f}%)")
        
        return combined_mask, physical_mask
    
    def create_convex_hull_mask(self, grid_rho: np.ndarray, grid_M: np.ndarray):
        """Create mask based on convex hull of data points."""
        
        try:
            # Prepare data points
            points = np.column_stack([self.rho, self.M])
            grid_points = np.column_stack([grid_rho.ravel(), grid_M.ravel()])
            
            # Create convex hull
            hull = ConvexHull(points)
            
            # Check which grid points are inside
            from matplotlib.path import Path as MPath
            hull_path = MPath(points[hull.vertices])
            hull_mask = hull_path.contains_points(grid_points).reshape(grid_rho.shape)
            
            return hull_mask
            
        except Exception as e:
            print(f"  Convex hull failed: {e}, using rectangular bounds")
            rho_mask = (grid_rho >= self.rho.min()) & (grid_rho <= self.rho.max())
            M_mask = (grid_M >= self.M.min()) & (grid_M <= self.M.max())
            return rho_mask & M_mask
    
    def interpolate_with_clean_masking(self, method: str = 'linear', resolution: int = 200,
                                     margin: float = 0.05, use_convex_hull: bool = False,
                                     boundary_method: str = 'hermite'):
        """
        Create interpolated surface with clean point-by-point masking.
        """
        
        print(f"Creating clean interpolation with {method} method...")
        
        # Create boundary functions first
        self.create_boundary_functions(boundary_method)
        
        # Create grid
        rho_min, rho_max = self.rho.min(), self.rho.max()
        M_min, M_max = self.M.min(), self.M.max()
        
        # Add padding
        rho_range = rho_max - rho_min
        M_range = M_max - M_min
        
        rho_grid = np.linspace(rho_min - 0.1*rho_range, rho_max + 0.1*rho_range, resolution)
        M_grid = np.linspace(M_min - 0.1*M_range, M_max + 0.1*M_range, resolution)
        rho_mesh, M_mesh = np.meshgrid(rho_grid, M_grid)
        
        # Perform interpolation
        points = np.column_stack([self.rho, self.M])
        grid_points = np.column_stack([rho_mesh.ravel(), M_mesh.ravel()])
        
        if method == 'rbf':
            try:
                print("  Using RBF interpolation...")
                rbf = RBFInterpolator(points, self.omega, 
                                    kernel='thin_plate_spline', 
                                    smoothing=0.1)
                omega_interp = rbf(grid_points).reshape(rho_mesh.shape)
                print("  ✓ RBF interpolation successful")
            except Exception as e:
                print(f"  RBF failed: {e}, falling back to linear")
                omega_interp = griddata(points, self.omega, (rho_mesh, M_mesh),
                                       method='linear', fill_value=np.nan)
        else:
            print(f"  Using {method} interpolation...")
            omega_interp = griddata(points, self.omega, (rho_mesh, M_mesh),
                                   method=method, fill_value=np.nan)
        
        # Apply clean masking
        print("  Applying clean point-by-point masking...")
        combined_mask, physical_mask = self.create_clean_mask(rho_mesh, M_mesh, margin, use_convex_hull)
        
        # Apply mask cleanly
        omega_masked = np.full_like(omega_interp, np.nan)
        valid_interp = ~np.isnan(omega_interp)
        final_mask = combined_mask & valid_interp
        omega_masked[final_mask] = omega_interp[final_mask]
        
        return {
            'rho_mesh': rho_mesh,
            'M_mesh': M_mesh,
            'omega_mesh': omega_masked,
            'omega_raw': omega_interp,
            'physical_mask': physical_mask,
            'combined_mask': combined_mask,
            'rho_data': self.rho,
            'M_data': self.M,
            'omega_data': self.omega,
            'eos': self.eos,
            'lower_bound_func': self.lower_bound_func,
            'upper_bound_func': self.upper_bound_func
        }
    
    def plot_clean_interpolation(self, interpolation_data: dict, 
                               output_dir: str = "./plots",
                               high_res_boundary: int = 2000):
        """Plot the clean interpolation with high-resolution boundaries."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Raw interpolation
        ax = axes[0, 0]
        valid_raw = ~np.isnan(interpolation_data['omega_raw'])
        if np.any(valid_raw):
            contour1 = ax.contourf(interpolation_data['rho_mesh'],
                                  interpolation_data['M_mesh'],
                                  interpolation_data['omega_raw'],
                                  levels=50, cmap='viridis', alpha=0.8)
            plt.colorbar(contour1, ax=ax, label='Ω (×10⁴ s⁻¹)')
        
        ax.scatter(interpolation_data['rho_data'],
                  interpolation_data['M_data'],
                  c=interpolation_data['omega_data'],
                  cmap='viridis', s=3, alpha=0.7,
                  edgecolors='white', linewidth=0.1)
        
        ax.set_title('Raw Interpolation')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Gravitational Mass M (M☉)')
        ax.grid(True, alpha=0.3)
        
        # 2. Clean masked interpolation
        ax = axes[0, 1]
        valid_masked = ~np.isnan(interpolation_data['omega_mesh'])
        if np.any(valid_masked):
            contour2 = ax.contourf(interpolation_data['rho_mesh'],
                                  interpolation_data['M_mesh'],
                                  interpolation_data['omega_mesh'],
                                  levels=50, cmap='viridis', alpha=0.8)
            plt.colorbar(contour2, ax=ax, label='Ω (×10⁴ s⁻¹)')
        
        ax.scatter(interpolation_data['rho_data'],
                  interpolation_data['M_data'],
                  c=interpolation_data['omega_data'],
                  cmap='viridis', s=3, alpha=0.7,
                  edgecolors='white', linewidth=0.1)
        
        ax.set_title('Clean Point-by-Point Masking')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Gravitational Mass M (M☉)')
        ax.grid(True, alpha=0.3)
        
        # 3. High-resolution boundaries
        ax = axes[1, 0]
        
        # Plot data points
        scatter = ax.scatter(interpolation_data['rho_data'],
                           interpolation_data['M_data'],
                           c=interpolation_data['omega_data'],
                           cmap='viridis', s=5, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Ω (×10⁴ s⁻¹)')
        
        # Plot boundary points
        if self.boundary_data:
            ax.scatter(self.boundary_data['lower_rho'], self.boundary_data['lower_M'], 
                      c='red', s=15, marker='v', alpha=0.9, 
                      label='Lower boundary points', zorder=10)
            ax.scatter(self.boundary_data['upper_rho'], self.boundary_data['upper_M'], 
                      c='red', s=15, marker='^', alpha=0.9, 
                      label='Upper boundary points', zorder=10)
        
        # High-resolution boundary curves
        if self.lower_bound_func and self.upper_bound_func:
            rho_fine = np.linspace(self.rho.min(), self.rho.max(), high_res_boundary)
            M_lower_smooth = self.lower_bound_func(rho_fine)
            M_upper_smooth = self.upper_bound_func(rho_fine)
            
            ax.plot(rho_fine, M_lower_smooth, 'r-', linewidth=3, alpha=0.9, 
                   label='Lower boundary')
            ax.plot(rho_fine, M_upper_smooth, 'r-', linewidth=3, alpha=0.9, 
                   label='Upper boundary')
            
            ax.fill_between(rho_fine, M_lower_smooth, M_upper_smooth, 
                           alpha=0.15, color='blue', label='Valid region')
        
        ax.set_title(f'High-Res Boundaries ({high_res_boundary} points)')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Gravitational Mass M (M☉)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # 4. Clean mask visualization
        ax = axes[1, 1]
        
        mask_plot = ax.contourf(interpolation_data['rho_mesh'],
                               interpolation_data['M_mesh'],
                               interpolation_data['combined_mask'].astype(int),
                               levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.6)
        
        ax.scatter(interpolation_data['rho_data'],
                  interpolation_data['M_data'],
                  c='black', s=2, alpha=0.8)
        
        ax.set_title('Clean Point-by-Point Mask')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Gravitational Mass M (M☉)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = output_dir / f"clean_boundary_interpolation_{self.eos}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved clean interpolation plot to {plot_path}")
        plt.show()

# Simple usage function
def process_eos_clean(df: pd.DataFrame, eos: str, 
                     method: str = 'linear',
                     boundary_method: str = 'hermite',
                     margin: float = 0.001,
                     resolution: int = 200,
                     output_dir: str = "./clean_plots"):
    """Process single EOS with clean boundary masking."""
    
    print(f"Processing {eos} with clean boundary masking...")
    
    interpolator = CleanBoundaryInterpolator(df, eos)
    
    # Create interpolation with clean masking
    result = interpolator.interpolate_with_clean_masking(
        method=method,
        resolution=resolution,
        margin=margin,
        use_convex_hull=False,  # Start without convex hull
        boundary_method=boundary_method
    )
    
    # Plot results
    interpolator.plot_clean_interpolation(result, output_dir)
    
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean boundary masking with direct function evaluation")
    parser.add_argument('data_file', help='Path to parquet file')
    parser.add_argument('--eos', help='EOS name to process')
    parser.add_argument('--method', choices=['rbf', 'cubic', 'linear', 'nearest'], default='linear',
                       help='Interpolation method')
    parser.add_argument('--boundary-method', choices=['linear', 'cubic', 'hermite'], 
                       default='hermite', help='Boundary interpolation method')
    parser.add_argument('--margin', type=float, default=0.001, 
                        help='Safety margin outside boundaries')
    parser.add_argument('--resolution', type=int, default=200,
                       help='Grid resolution')
    parser.add_argument('--output', '-o', default='./clean_plots',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_parquet(args.data_file)
    print(f"Loaded {len(df)} neutron star models")
    
    if args.eos:
        if args.eos in df['eos'].unique():
            result = process_eos_clean(df, args.eos, args.method, args.boundary_method, 
                                     args.margin, args.resolution, args.output)
        else:
            print(f"EOS '{args.eos}' not found. Available: {list(df['eos'].unique())}")
    else:
        print("Please specify --eos parameter")
        print(f"Available EOS: {list(df['eos'].unique())}")