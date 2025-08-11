#!/usr/bin/env python3
"""
Super Interpolation with Physical Boundary Masking
==================================================

Combines high-quality smooth interpolation with physical boundary constraints
for neutron star data. Creates smooth surfaces that respect physical limits.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator, interp1d, CubicSpline, PchipInterpolator
from scipy.spatial import ConvexHull
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import matplotlib.patches as patches
from pathlib import Path

class SuperInterpolator:
    """Advanced interpolation with physical boundary masking."""
    
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
        
    def extract_physical_boundaries(self, n_bins: int = 400, smoothing_window: int = 15,
                                  boundary_method: str = 'cubic'):
        """Extract super-smooth physical boundaries from data."""
        
        print(f"  Extracting boundaries with {n_bins} bins, smoothing {smoothing_window}")
        print(f"  Boundary interpolation method: {boundary_method}")
        
        # Create high-resolution density bins
        rho_min, rho_max = self.rho.min(), self.rho.max()
        rho_bins = np.linspace(rho_min, rho_max, n_bins)
        rho_bin_centers = (rho_bins[:-1] + rho_bins[1:]) / 2
        
        lower_boundary_M = []
        upper_boundary_M = []
        valid_rho = []
        
        # For each density bin, find min and max mass
        for i, (rho_low, rho_high) in enumerate(zip(rho_bins[:-1], rho_bins[1:])):
            in_bin = (self.rho >= rho_low) & (self.rho < rho_high)
            
            if np.sum(in_bin) > 0:
                M_in_bin = self.M[in_bin]
                lower_boundary_M.append(M_in_bin.min())
                upper_boundary_M.append(M_in_bin.max())
                valid_rho.append(rho_bin_centers[i])
        
        valid_rho = np.array(valid_rho)
        lower_boundary_M = np.array(lower_boundary_M)
        upper_boundary_M = np.array(upper_boundary_M)
        
        print(f"  Found {len(valid_rho)} valid boundary points")
        
        # Apply smoothing based on method (skip for linear and bilinear)
        if boundary_method not in ['linear', 'bilinear'] and len(valid_rho) > smoothing_window:
            print("  Applying Gaussian smoothing to boundaries...")
            # Use Gaussian filter for smoother results than uniform filter
            sigma = smoothing_window / 3.0  # Convert window to sigma
            lower_boundary_M = gaussian_filter1d(lower_boundary_M, sigma=sigma)
            upper_boundary_M = gaussian_filter1d(upper_boundary_M, sigma=sigma)
        
        # Create interpolators based on specified method
        lower_interp, upper_interp = self._create_boundary_interpolators(
            valid_rho, lower_boundary_M, upper_boundary_M, boundary_method
        )
        
        return lower_interp, upper_interp, valid_rho, lower_boundary_M, upper_boundary_M
    
    def _create_boundary_interpolators(self, valid_rho, lower_boundary_M, upper_boundary_M, method):
        try:
            if method == 'linear':
                print("  Creating linear boundary interpolators...")
                return (
                    interp1d(valid_rho, lower_boundary_M, kind='linear', bounds_error=False, fill_value='extrapolate'),
                    interp1d(valid_rho, upper_boundary_M, kind='linear', bounds_error=False, fill_value='extrapolate')
                )
            elif method == 'bilinear':
                print("  Creating bilinear boundary interpolators...")
                return (
                    interp1d(valid_rho, lower_boundary_M, kind='linear', bounds_error=False,
                             fill_value=(lower_boundary_M[0], lower_boundary_M[-1])),
                    interp1d(valid_rho, upper_boundary_M, kind='linear', bounds_error=False,
                             fill_value=(upper_boundary_M[0], upper_boundary_M[-1]))
                )
            elif method == 'cubic':
                print("  Creating cubic spline boundary interpolators...")
                return (
                    CubicSpline(valid_rho, lower_boundary_M, bc_type='natural', extrapolate=True),
                    CubicSpline(valid_rho, upper_boundary_M, bc_type='natural', extrapolate=True)
                )
            elif method == 'hermite':
                print("  Creating Hermite spline boundary interpolators...")
                return (
                    PchipInterpolator(valid_rho, lower_boundary_M, extrapolate=True),
                    PchipInterpolator(valid_rho, upper_boundary_M, extrapolate=True)
                )
        except Exception as e:
            print(f"  Interpolation method '{method}' failed: {e}")

        print("  Falling back to linear interpolation...")
        return (
            interp1d(valid_rho, lower_boundary_M, kind='linear', bounds_error=False, fill_value='extrapolate'),
            interp1d(valid_rho, upper_boundary_M, kind='linear', bounds_error=False, fill_value='extrapolate')
        )

    
    def create_physical_mask(self, grid_rho: np.ndarray, grid_M: np.ndarray, 
                           margin: float = 0.05, boundary_method: str = 'cubic'):
        """Create mask based on physical boundaries with safety margins."""
        
        lower_interp, upper_interp, _, _, _ = self.extract_physical_boundaries(
            boundary_method=boundary_method)
        
        # Store original shape and flatten for interpolation
        original_shape = grid_rho.shape
        rho_flat = grid_rho.ravel()
        M_flat = grid_M.ravel()
        
        # Evaluate boundaries at grid densities
        M_lower_at_grid = lower_interp(rho_flat)
        M_upper_at_grid = upper_interp(rho_flat)
        
        # Ensure arrays are the right shape
        if np.isscalar(M_lower_at_grid):
            M_lower_at_grid = np.full_like(rho_flat, M_lower_at_grid)
        if np.isscalar(M_upper_at_grid):
            M_upper_at_grid = np.full_like(rho_flat, M_upper_at_grid)
        
        # Add safety margins
        M_range = self.M.max() - self.M.min()
        margin_size = margin * M_range
        
        M_lower_margin = M_lower_at_grid - margin_size
        M_upper_margin = M_upper_at_grid + margin_size
        
        # Create mask: valid if mass is between boundaries
        physical_mask = (M_flat >= M_lower_margin) & (M_flat <= M_upper_margin)
        
        # Reshape back to original grid shape
        physical_mask = physical_mask.reshape(original_shape)
        
        return physical_mask
    
    def create_convex_hull_mask(self, grid_rho: np.ndarray, grid_M: np.ndarray):
        """Create mask based on convex hull of data points."""
        
        try:
            # Prepare data points (scaled)
            points = np.column_stack([self.rho, self.M])
            grid_points = np.column_stack([grid_rho.ravel(), grid_M.ravel()])
            
            # Add small jitter to avoid coplanar points
            jittered_points = points + np.random.normal(0, 1e-10, points.shape)
            hull = ConvexHull(jittered_points)
            
            # Check which grid points are inside the convex hull
            from matplotlib.path import Path as MPath
            hull_path = MPath(jittered_points[hull.vertices])
            hull_mask = hull_path.contains_points(grid_points).reshape(grid_rho.shape)
            
            print(f"  Successfully created convex hull mask")
            return hull_mask
            
        except Exception as e:
            print(f"  Convex hull failed: {e}, using rectangular bounds")
            # Fallback to rectangular mask
            rho_mask = (grid_rho >= self.rho.min()) & (grid_rho <= self.rho.max())
            M_mask = (grid_M >= self.M.min()) & (grid_M <= self.M.max())
            return rho_mask & M_mask
    
    def super_interpolate(self, method: str = 'rbf', resolution: int = 200,
                         margin: float = 0.05, use_convex_hull: bool = True,
                         boundary_method: str = 'cubic'):
        """
        Create super-smooth interpolated surface with physical boundary masking.
        
        Args:
            method: 'rbf', 'cubic', 'linear', 'nearest', or 'multiquadric'
            resolution: Grid resolution
            margin: Safety margin for physical boundaries
            use_convex_hull: Whether to also apply convex hull masking
            boundary_method: 'linear', 'bilinear', 'cubic', or 'hermite' for boundaries
        """
        
        print(f"Creating super interpolation with {method} method...")
        
        # Create high-resolution grid
        rho_min, rho_max = self.rho.min(), self.rho.max()
        M_min, M_max = self.M.min(), self.M.max()
        
        # Add some padding for better interpolation
        rho_range = rho_max - rho_min
        M_range = M_max - M_min
        
        rho_grid = np.linspace(rho_min - 0.1*rho_range, rho_max + 0.1*rho_range, resolution)
        M_grid = np.linspace(M_min - 0.1*M_range, M_max + 0.1*M_range, resolution)
        rho_mesh, M_mesh = np.meshgrid(rho_grid, M_grid)
        
        # Prepare data points (using scaled coordinates)
        points = np.column_stack([self.rho, self.M])
        grid_points = np.column_stack([rho_mesh.ravel(), M_mesh.ravel()])
        
        # Perform high-quality interpolation
        if method == 'rbf':
            try:
                print("  Using RBF interpolation (thin plate spline)...")
                rbf = RBFInterpolator(points, self.omega, 
                                    kernel='thin_plate_spline', 
                                    smoothing=0.1)
                omega_interp = rbf(grid_points).reshape(rho_mesh.shape)
                print("  ✓ RBF interpolation successful")
            except Exception as e:
                print(f"  RBF failed: {e}, falling back to cubic")
                omega_interp = griddata(points, self.omega, (rho_mesh, M_mesh),
                                       method='cubic', fill_value=np.nan)
        
        elif method == 'multiquadric':
            try:
                print("  Using RBF multiquadric...")
                rbf = RBFInterpolator(points, self.omega, 
                                    kernel='multiquadric', 
                                    smoothing=0.1)
                omega_interp = rbf(grid_points).reshape(rho_mesh.shape)
                print("  ✓ RBF multiquadric successful")
            except Exception as e:
                print(f"  RBF multiquadric failed: {e}, falling back to cubic")
                omega_interp = griddata(points, self.omega, (rho_mesh, M_mesh),
                                       method='cubic', fill_value=np.nan)
        
        elif method == 'cubic':
            print("  Using cubic interpolation...")
            omega_interp = griddata(points, self.omega, (rho_mesh, M_mesh),
                                   method='cubic', fill_value=np.nan)
        
        elif method == 'linear':
            print("  Using linear interpolation...")
            omega_interp = griddata(points, self.omega, (rho_mesh, M_mesh),
                                   method='linear', fill_value=np.nan)
        
        elif method == 'nearest':
            print("  Using nearest neighbor interpolation...")
            omega_interp = griddata(points, self.omega, (rho_mesh, M_mesh),
                                   method='nearest', fill_value=np.nan)
        
        else:
            print(f"  Unknown method '{method}', using cubic...")
            omega_interp = griddata(points, self.omega, (rho_mesh, M_mesh),
                                   method='cubic', fill_value=np.nan)
        
        # Create physical boundary mask
        print("  Applying physical boundary mask...")
        physical_mask = self.create_physical_mask(rho_mesh, M_mesh, margin, boundary_method)
        
        # Create convex hull mask if requested
        if use_convex_hull:
            print("  Applying convex hull mask...")
            hull_mask = self.create_convex_hull_mask(rho_mesh, M_mesh)
            combined_mask = physical_mask & hull_mask
        else:
            combined_mask = physical_mask
        
        # Apply masks
        omega_masked = omega_interp.copy()
        omega_masked[~combined_mask] = np.nan
        
        # Count valid points
        valid_before = np.sum(~np.isnan(omega_interp))
        valid_after = np.sum(~np.isnan(omega_masked))
        print(f"  Valid points: {valid_after}/{valid_before} ({valid_after/valid_before*100:.1f}%)")
        
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
            'eos': self.eos
        }
    
    def plot_super_interpolation(self, interpolation_data: dict, 
                               output_dir: str = "./plots",
                               show_boundaries: bool = True):
        """Create comprehensive plots of the super interpolation."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Raw interpolation (top-left)
        ax = axes[0, 0]
        contour1 = ax.contourf(interpolation_data['rho_mesh'],
                              interpolation_data['M_mesh'],
                              interpolation_data['omega_raw'],
                              levels=50, cmap='viridis', alpha=0.8)
        
        ax.scatter(interpolation_data['rho_data'],
                  interpolation_data['M_data'],
                  c=interpolation_data['omega_data'],
                  cmap='viridis', s=3, alpha=0.7,
                  edgecolors='white', linewidth=0.1)
        
        plt.colorbar(contour1, ax=ax, label='Ω (×10⁴ s⁻¹)')
        ax.set_title('Raw Interpolation (Before Masking)')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Gravitational Mass M (M☉)')
        ax.grid(True, alpha=0.3)
        
        # 2. Masked interpolation (top-right)
        ax = axes[0, 1]
        contour2 = ax.contourf(interpolation_data['rho_mesh'],
                              interpolation_data['M_mesh'],
                              interpolation_data['omega_mesh'],
                              levels=50, cmap='viridis', alpha=0.8)
        
        ax.scatter(interpolation_data['rho_data'],
                  interpolation_data['M_data'],
                  c=interpolation_data['omega_data'],
                  cmap='viridis', s=3, alpha=0.7,
                  edgecolors='white', linewidth=0.1)
        
        plt.colorbar(contour2, ax=ax, label='Ω (×10⁴ s⁻¹)')
        ax.set_title('Super Interpolation (After Masking)')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Gravitational Mass M (M☉)')
        ax.grid(True, alpha=0.3)
        
        # 3. Physical boundaries (bottom-left)
        ax = axes[1, 0]
        
        # Show original data
        scatter = ax.scatter(interpolation_data['rho_data'],
                           interpolation_data['M_data'],
                           c=interpolation_data['omega_data'],
                           cmap='viridis', s=5, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Ω (×10⁴ s⁻¹)')
        
        if show_boundaries:
            # Extract and plot boundaries
            lower_interp, upper_interp, rho_boundary, M_lower, M_upper = self.extract_physical_boundaries()
            
            # Fine grid for smooth boundary curves
            rho_fine = np.linspace(self.rho.min(), self.rho.max(), 200)
            M_lower_smooth = lower_interp(rho_fine)
            M_upper_smooth = upper_interp(rho_fine)
            
            # Plot boundary points and curves
            ax.scatter(rho_boundary, M_lower, c='red', s=15, alpha=0.8, 
                      marker='v', label='Lower boundary')
            ax.scatter(rho_boundary, M_upper, c='red', s=15, alpha=0.8, 
                      marker='^', label='Upper boundary')
            
            ax.plot(rho_fine, M_lower_smooth, 'r-', linewidth=2, alpha=0.8)
            ax.plot(rho_fine, M_upper_smooth, 'r-', linewidth=2, alpha=0.8)
            
            # Fill valid region
            ax.fill_between(rho_fine, M_lower_smooth, M_upper_smooth, 
                           alpha=0.1, color='blue', label='Valid region')
            
            ax.legend()
        
        ax.set_title('Physical Boundaries')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Gravitational Mass M (M☉)')
        ax.grid(True, alpha=0.3)
        
        # 4. Mask visualization (bottom-right)
        ax = axes[1, 1]
        
        # Show the combined mask
        mask_plot = ax.contourf(interpolation_data['rho_mesh'],
                               interpolation_data['M_mesh'],
                               interpolation_data['combined_mask'].astype(int),
                               levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.6)
        
        ax.scatter(interpolation_data['rho_data'],
                  interpolation_data['M_data'],
                  c='black', s=2, alpha=0.8)
        
        ax.set_title('Combined Mask (Green=Valid, Red=Masked)')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Gravitational Mass M (M☉)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f"super_interpolation_{self.eos}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved super interpolation plot to {plot_path}")
        plt.show()
    
    def generate_training_data(self, n_points: int = 50000, 
                             method: str = 'rbf',
                             margin: float = 0.05,
                             boundary_method: str = 'cubic'):
        """Generate high-quality training data using super interpolation."""
        
        print(f"Generating {n_points} training points using super interpolation...")
        
        # Create super interpolation
        interp_data = self.super_interpolate(
            method=method, 
            resolution=int(np.sqrt(n_points * 2)),
            margin=margin,
            boundary_method=boundary_method
        )
        
        # Extract valid points
        valid_mask = ~np.isnan(interp_data['omega_mesh'])
        rho_valid = interp_data['rho_mesh'][valid_mask]
        M_valid = interp_data['M_mesh'][valid_mask]
        omega_valid = interp_data['omega_mesh'][valid_mask]
        
        print(f"  Generated {len(rho_valid)} valid interpolated points")
        
        # Subsample if we have too many points
        if len(rho_valid) > n_points:
            idx = np.random.choice(len(rho_valid), n_points, replace=False)
            rho_valid = rho_valid[idx]
            M_valid = M_valid[idx]
            omega_valid = omega_valid[idx]
            print(f"  Subsampled to {n_points} points")
        
        # Clamp omega values to data range for safety
        omega_valid = np.clip(omega_valid, self.omega.min(), self.omega.max())
        
        # Convert back to original units
        rho_valid_raw = rho_valid * 1e15
        
        return {
            'rho_c': rho_valid_raw,
            'M': M_valid,
            'Omega': omega_valid,
            'n_points': len(rho_valid),
            'eos': self.eos
        }

# Function for processing all EOS
def process_all_eos(df: pd.DataFrame,
                    output_dir: str = "./super_interpolation",
                    method: str = 'rbf',
                    boundary_method: str = 'cubic',
                    n_points: int = 50000,
                    margin: float = 0.05,
                    create_plots: bool = True):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    all_training_data = []

    for eos in df['eos'].unique():
        try:
            print(f"\n{'='*60}")
            print(f"Processing {eos} with Super Interpolation")
            print(f"{'='*60}")

            super_interp = SuperInterpolator(df, eos)
            interp_data = super_interp.super_interpolate(
                method=method, margin=margin, boundary_method=boundary_method
            )

            if create_plots:
                super_interp.plot_super_interpolation(interp_data, output_dir)

            training_data = super_interp.generate_training_data(
                n_points=n_points,
                method=method,
                margin=margin,
                boundary_method=boundary_method
            )

            eos_training_df = pd.DataFrame({
                'eos': [eos] * training_data['n_points'],
                'rho_c': training_data['rho_c'],
                'M': training_data['M'],
                'Omega': training_data['Omega']
            })

            all_training_data.append(eos_training_df)

            eos_file = output_dir / f"super_training_{eos}.parquet"
            eos_training_df.to_parquet(eos_file, index=False)
            print(f"Saved training data to {eos_file}")

        except Exception as e:
            print(f"❌ Failed for {eos}: {e}")
            import traceback
            traceback.print_exc()

    if all_training_data:
        combined_df = pd.concat(all_training_data, ignore_index=True)
        combined_file = output_dir / "super_training_combined.parquet"
        combined_df.to_parquet(combined_file, index=False)

        print(f"\n{'='*60}")
        print("SUPER INTERPOLATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total training points: {len(combined_df):,}")
        print(f"Original data points: {len(df):,}")
        print(f"Augmentation factor: {len(combined_df) / len(df):.1f}x")
        print(f"Main interpolation method: {method}")
        print(f"Boundary interpolation method: {boundary_method}")
        print(f"Physical boundary margin: {margin * 100:.1f}%")

        return combined_df
    else:
        raise ValueError("Failed to generate any training data")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Super interpolation with physical boundary masking")
    parser.add_argument('data_file', help='Path to parquet file')
    parser.add_argument('--method', choices=['rbf', 'multiquadric', 'cubic', 'linear', 'nearest'], default='rbf',
                       help='Interpolation method (rbf recommended)')
    parser.add_argument('--boundary-method', choices=['linear', 'bilinear', 'cubic', 'hermite'], 
                       default='cubic', help='Boundary interpolation method (default: cubic)')
    parser.add_argument('--n-points', type=int, default=50000,
                       help='Training points per EOS')
    parser.add_argument('--margin', type=float, default=0.05, 
                        help='Safety margin outside boundaries (fraction)')
    parser.add_argument('--output', '-o', default='./super_interpolation',
                       help='Output directory')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip creating plots')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_parquet(args.data_file)
    print(f"Loaded {len(df)} neutron star models")
    
    # Process all EOS
    training_df = process_all_eos(
        df, args.output, args.method, args.boundary_method,
        args.n_points, args.margin, not args.no_plots
    )