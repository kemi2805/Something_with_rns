#!/usr/bin/env python3
"""
Simple Local Interpolation with Physical Boundary Masking
=========================================================

Uses each data point effectively with local 2+2 neighbor interpolation
for creating smooth boundaries that respect physical limits.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RBFInterpolator, interp1d, CubicSpline, PchipInterpolator
from scipy.spatial import ConvexHull
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
import matplotlib.patches as patches
from pathlib import Path

class SimpleLocalInterpolator:
    """Simple local interpolation using 2+2 neighbors for each point."""
    
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
        
    def extract_simple_local_boundaries(self, boundary_method: str = 'hermite'):
        """
        Extract TRUE data envelope boundaries using actual data distribution.
        For each density, find the actual min/max mass values present in the data.
        """
        
        print(f"  Extracting TRUE envelope boundaries from data")
        print(f"  Boundary interpolation method: {boundary_method}")
        
        # Get unique density values and find min/max mass at each density
        unique_rho = np.unique(self.rho)
        
        lower_boundary_rho = []
        lower_boundary_M = []
        upper_boundary_rho = []
        upper_boundary_M = []
        
        print(f"  Processing {len(unique_rho)} unique density values")
        
        for rho_val in unique_rho:
            # Find all masses at this density
            at_density = self.M[self.rho == rho_val]
            
            if len(at_density) > 0:
                # True boundaries are simply min/max at each density
                lower_boundary_rho.append(rho_val)
                lower_boundary_M.append(at_density.min())
                
                upper_boundary_rho.append(rho_val)
                upper_boundary_M.append(at_density.max())
        
        # Convert to arrays
        lower_boundary_rho = np.array(lower_boundary_rho)
        lower_boundary_M = np.array(lower_boundary_M)
        upper_boundary_rho = np.array(upper_boundary_rho)
        upper_boundary_M = np.array(upper_boundary_M)
        
        # Sort by density (should already be sorted, but ensure it)
        lower_sort = np.argsort(lower_boundary_rho)
        lower_boundary_rho = lower_boundary_rho[lower_sort]
        lower_boundary_M = lower_boundary_M[lower_sort]
        
        upper_sort = np.argsort(upper_boundary_rho)
        upper_boundary_rho = upper_boundary_rho[upper_sort]
        upper_boundary_M = upper_boundary_M[upper_sort]
        
        print(f"  Lower boundary: {len(lower_boundary_rho)} points")
        print(f"  Upper boundary: {len(upper_boundary_rho)} points")
        print(f"  Lower M range: {lower_boundary_M.min():.3f} - {lower_boundary_M.max():.3f}")
        print(f"  Upper M range: {upper_boundary_M.min():.3f} - {upper_boundary_M.max():.3f}")
        
        # Create interpolators
        lower_interp, upper_interp = self._create_boundary_interpolators(
            lower_boundary_rho, lower_boundary_M, 
            upper_boundary_rho, upper_boundary_M, 
            boundary_method
        )
        
        return lower_interp, upper_interp, lower_boundary_rho, lower_boundary_M, upper_boundary_M
    
    def _create_boundary_interpolators(self, lower_rho, lower_M, upper_rho, upper_M, method):
        """Create smart interpolators for lower and upper boundaries with derivative preservation."""
        
        # Ensure we have enough points for the requested method
        min_points_needed = {'linear': 2, 'bilinear': 2, 'cubic': 4, 'hermite': 3}
        min_lower = min_points_needed.get(method, 2)
        min_upper = min_points_needed.get(method, 2)
        
        print(f"  Creating {method} interpolators with {len(lower_rho)} lower, {len(upper_rho)} upper points")
        
        try:
            if method == 'linear':
                print("  Using linear boundary interpolators...")
                return (
                    interp1d(lower_rho, lower_M, kind='linear', bounds_error=False, fill_value='extrapolate'),
                    interp1d(upper_rho, upper_M, kind='linear', bounds_error=False, fill_value='extrapolate')
                )
                
            elif method == 'bilinear':
                print("  Using bilinear boundary interpolators...")
                return (
                    interp1d(lower_rho, lower_M, kind='linear', bounds_error=False,
                             fill_value=(lower_M[0], lower_M[-1])),
                    interp1d(upper_rho, upper_M, kind='linear', bounds_error=False,
                             fill_value=(upper_M[0], upper_M[-1]))
                )
                
            elif method == 'cubic' and len(lower_rho) >= 4 and len(upper_rho) >= 4:
                print("  Using cubic spline boundary interpolators with natural boundary conditions...")
                return (
                    CubicSpline(lower_rho, lower_M, bc_type='natural', extrapolate=True),
                    CubicSpline(upper_rho, upper_M, bc_type='natural', extrapolate=True)
                )
                
            elif method == 'hermite' and len(lower_rho) >= 3 and len(upper_rho) >= 3:
                print("  Using PCHIP Hermite interpolators (derivative-preserving)...")
                # PCHIP is the best Hermite interpolator - preserves monotonicity and derivatives
                return (
                    PchipInterpolator(lower_rho, lower_M, extrapolate=True),
                    PchipInterpolator(upper_rho, upper_M, extrapolate=True)
                )
                
            else:
                print(f"  Insufficient points for {method} (need {min_lower}/{min_upper}), trying alternatives...")
                
                # Smart fallback hierarchy
                if len(lower_rho) >= 3 and len(upper_rho) >= 3:
                    print("  Falling back to PCHIP Hermite interpolators...")
                    return (
                        PchipInterpolator(lower_rho, lower_M, extrapolate=True),
                        PchipInterpolator(upper_rho, upper_M, extrapolate=True)
                    )
                elif len(lower_rho) >= 4 and len(upper_rho) >= 4:
                    print("  Falling back to cubic spline...")
                    return (
                        CubicSpline(lower_rho, lower_M, bc_type='natural', extrapolate=True),
                        CubicSpline(upper_rho, upper_M, bc_type='natural', extrapolate=True)
                    )
                else:
                    print("  Falling back to quadratic interpolation...")
                    return (
                        interp1d(lower_rho, lower_M, kind='quadratic', bounds_error=False, fill_value='extrapolate'),
                        interp1d(upper_rho, upper_M, kind='quadratic', bounds_error=False, fill_value='extrapolate')
                    )
                    
        except Exception as e:
            print(f"  Primary interpolation method failed: {e}")
            
            # Final fallback - always works
            print("  Using final fallback: linear with extrapolation...")
            return (
                interp1d(lower_rho, lower_M, kind='linear', bounds_error=False, fill_value='extrapolate'),
                interp1d(upper_rho, upper_M, kind='linear', bounds_error=False, fill_value='extrapolate')
            )
    
    def create_physical_mask(self, grid_rho: np.ndarray, grid_M: np.ndarray, 
                           margin: float = 0.05, boundary_method: str = 'hermite',
                           check_method: str = 'simple'):
        """
        Create mask based on physical boundaries with different checking methods.
        
        Args:
            check_method: 'simple', 'gradient_aware', or 'distance_based'
        """
        
        lower_interp, upper_interp, _, _, _ = self.extract_simple_local_boundaries(
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
        
        print(f"  Using {check_method} boundary checking method")
        
        if check_method == 'simple':
            # SIMPLE CHECK: Just margins
            M_range = self.M.max() - self.M.min()
            margin_size = margin * M_range
            
            M_lower_margin = M_lower_at_grid - margin_size
            M_upper_margin = M_upper_at_grid + margin_size
            
            # Simple check: is point between boundaries?
            physical_mask = (M_flat >= M_lower_margin) & (M_flat <= M_upper_margin)
            
        elif check_method == 'gradient_aware':
            # GRADIENT-AWARE CHECK: Adaptive margins based on boundary slope
            print("    Computing boundary gradients for adaptive margins...")
            
            # Compute gradients of boundaries
            drho = np.diff(rho_flat)
            if len(drho) > 0:
                avg_drho = np.mean(np.abs(drho[drho != 0]))
            else:
                avg_drho = (self.rho.max() - self.rho.min()) / len(self.rho)
            
            # Estimate gradients by finite differences
            rho_test = rho_flat + avg_drho * 0.01  # Small offset
            M_lower_test = lower_interp(rho_test)
            M_upper_test = upper_interp(rho_test)
            
            # Compute slopes
            lower_slope = np.abs(M_lower_test - M_lower_at_grid) / (avg_drho * 0.01)
            upper_slope = np.abs(M_upper_test - M_upper_at_grid) / (avg_drho * 0.01)
            
            # Adaptive margin based on local slope
            base_margin = margin * (self.M.max() - self.M.min())
            lower_adaptive_margin = base_margin * (1 + lower_slope * 0.1)
            upper_adaptive_margin = base_margin * (1 + upper_slope * 0.1)
            
            M_lower_margin = M_lower_at_grid - lower_adaptive_margin
            M_upper_margin = M_upper_at_grid + upper_adaptive_margin
            
            physical_mask = (M_flat >= M_lower_margin) & (M_flat <= M_upper_margin)
            
        elif check_method == 'distance_based':
            # DISTANCE-BASED CHECK: Use distance from nearest data point
            print("    Computing distance-based validity...")
            
            # For each grid point, find distance to nearest actual data point
            from scipy.spatial.distance import cdist
            
            # Normalize coordinates for distance calculation
            rho_norm = (rho_flat - self.rho.min()) / (self.rho.max() - self.rho.min())
            M_norm = (M_flat - self.M.min()) / (self.M.max() - self.M.min())
            grid_points_norm = np.column_stack([rho_norm, M_norm])
            
            data_rho_norm = (self.rho - self.rho.min()) / (self.rho.max() - self.rho.min())
            data_M_norm = (self.M - self.M.min()) / (self.M.max() - self.M.min())
            data_points_norm = np.column_stack([data_rho_norm, data_M_norm])
            
            # Compute distances (this might be slow for large grids)
            distances = cdist(grid_points_norm, data_points_norm)
            min_distances = np.min(distances, axis=1)
            
            # Points are valid if they're close to data AND within boundaries
            distance_threshold = margin * 2  # Adjust this
            distance_mask = min_distances <= distance_threshold
            
            # Still apply boundary check
            M_range = self.M.max() - self.M.min()
            margin_size = margin * M_range
            M_lower_margin = M_lower_at_grid - margin_size
            M_upper_margin = M_upper_at_grid + margin_size
            boundary_mask = (M_flat >= M_lower_margin) & (M_flat <= M_upper_margin)
            
            # Combine both checks
            physical_mask = distance_mask & boundary_mask
            
        else:
            # Fallback to simple
            print(f"    Unknown check method '{check_method}', using simple")
            M_range = self.M.max() - self.M.min()
            margin_size = margin * M_range
            M_lower_margin = M_lower_at_grid - margin_size
            M_upper_margin = M_upper_at_grid + margin_size
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
    
    def simple_interpolate(self, method: str = 'rbf', resolution: int = 1000,
                          margin: float = 0.05, use_convex_hull: bool = True,
                          boundary_method: str = 'hermite', check_method: str = 'simple'):
        """
        Create smooth interpolated surface with simple local boundary masking.
        
        Args:
            method: 'rbf', 'cubic', 'linear', 'nearest', or 'multiquadric'
            resolution: Grid resolution
            margin: Safety margin for physical boundaries
            use_convex_hull: Whether to also apply convex hull masking
            boundary_method: 'linear', 'bilinear', 'cubic', or 'hermite' for boundaries
        """
        
        print(f"Creating simple local interpolation with {method} method...")
        
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
        
        # Create physical boundary mask using specified checking method
        print("  Applying physical boundary mask...")
        physical_mask = self.create_physical_mask(rho_mesh, M_mesh, margin, boundary_method, check_method)
        
        # Create convex hull mask if requested
        if use_convex_hull:
            print("  Applying convex hull mask...")
            hull_mask = self.create_convex_hull_mask(rho_mesh, M_mesh)
            combined_mask = physical_mask & hull_mask
        else:
            combined_mask = physical_mask
        
        # Apply masks with better NaN handling
        omega_masked = omega_interp.copy()
        
        # First, ensure we have valid interpolation results
        valid_interp_mask = ~np.isnan(omega_interp)
        
        # Combine with physical masks
        final_mask = combined_mask & valid_interp_mask
        
        # Apply mask more cleanly
        omega_masked = np.full_like(omega_interp, np.nan)
        omega_masked[final_mask] = omega_interp[final_mask]
        
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
    
    def plot_simple_interpolation(self, interpolation_data: dict, 
                                 output_dir: str = "./plots",
                                 show_boundaries: bool = True):
        """Create comprehensive plots of the simple local interpolation."""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Raw interpolation (top-left)
        ax = axes[0, 0]
        
        # Only plot where we have valid data
        valid_raw = ~np.isnan(interpolation_data['omega_raw'])
        if np.any(valid_raw):
            contour1 = ax.contourf(interpolation_data['rho_mesh'],
                                  interpolation_data['M_mesh'],
                                  interpolation_data['omega_raw'],
                                  levels=50, cmap='viridis', alpha=0.8)
            plt.colorbar(contour1, ax=ax, label='Ω (×10⁴ s⁻¹)')
        else:
            ax.text(0.5, 0.5, 'No valid raw interpolation', transform=ax.transAxes, ha='center')
        
        ax.scatter(interpolation_data['rho_data'],
                  interpolation_data['M_data'],
                  c=interpolation_data['omega_data'],
                  cmap='viridis', s=3, alpha=0.7,
                  edgecolors='white', linewidth=0.1)
        
        ax.set_title('Raw Interpolation (Before Masking)')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Gravitational Mass M (M☉)')
        ax.grid(True, alpha=0.3)
        
        # 2. Masked interpolation (top-right)
        ax = axes[0, 1]
        
        # Only plot where we have valid masked data
        valid_masked = ~np.isnan(interpolation_data['omega_mesh'])
        if np.any(valid_masked):
            contour2 = ax.contourf(interpolation_data['rho_mesh'],
                                  interpolation_data['M_mesh'],
                                  interpolation_data['omega_mesh'],
                                  levels=50, cmap='viridis', alpha=0.8)
            plt.colorbar(contour2, ax=ax, label='Ω (×10⁴ s⁻¹)')
        else:
            ax.text(0.5, 0.5, 'No valid masked interpolation', transform=ax.transAxes, ha='center')
        
        ax.scatter(interpolation_data['rho_data'],
                  interpolation_data['M_data'],
                  c=interpolation_data['omega_data'],
                  cmap='viridis', s=3, alpha=0.7,
                  edgecolors='white', linewidth=0.1)
        
        ax.set_title('Simple Local Interpolation (After Masking)')
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
            # Extract and plot boundaries using simple local method
            lower_interp, upper_interp, rho_lower, M_lower, M_upper = self.extract_simple_local_boundaries()
            
            # Plot boundary points
            ax.scatter(rho_lower, M_lower, c='red', s=20, alpha=0.8, 
                      marker='v', label='Lower boundary points', zorder=10)
            
            # For upper boundary, we need the rho values too
            _, _, rho_upper, _, M_upper = self.extract_simple_local_boundaries()
            ax.scatter(rho_upper, M_upper, c='red', s=20, alpha=0.8, 
                      marker='^', label='Upper boundary points', zorder=10)
            
            # Fine grid for smooth boundary curves
            rho_fine = np.linspace(self.rho.min(), self.rho.max(), 200)
            M_lower_smooth = lower_interp(rho_fine)
            M_upper_smooth = upper_interp(rho_fine)
            
            ax.plot(rho_fine, M_lower_smooth, 'r-', linewidth=2, alpha=0.8, label='Lower boundary')
            ax.plot(rho_fine, M_upper_smooth, 'r-', linewidth=2, alpha=0.8, label='Upper boundary')
            
            # Fill valid region
            ax.fill_between(rho_fine, M_lower_smooth, M_upper_smooth, 
                           alpha=0.1, color='blue', label='Valid region')
            
            ax.legend()
        
        ax.set_title('TRUE Data Envelope Boundaries')
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
        plot_path = output_dir / f"simple_local_interpolation_{self.eos}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved simple local interpolation plot to {plot_path}")
        plt.show()
    
    def generate_training_data(self, n_points: int = 50000, 
                             method: str = 'rbf',
                             margin: float = 0.05,
                             boundary_method: str = 'hermite',
                             check_method: str = 'simple'):
        """Generate high-quality training data using simple local interpolation."""
        
        print(f"Generating {n_points} training points using simple local interpolation...")
        
        # Create simple interpolation
        interp_data = self.simple_interpolate(
            method=method, 
            resolution=int(np.sqrt(n_points * 2)),
            margin=margin,
            boundary_method=boundary_method,
            check_method=check_method
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
                    output_dir: str = "./simple_local_interpolation",
                    method: str = 'rbf',
                    boundary_method: str = 'hermite',
                    n_points: int = 50000,
                    margin: float = 0.05,
                    create_plots: bool = True,
                    use_convex_hull: bool = True,
                    check_method: str = 'simple'):
    """Process all EOS using simple local interpolation."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    all_training_data = []

    for eos in df['eos'].unique():
        try:
            print(f"\n{'='*60}")
            print(f"Processing {eos} with Simple Local Interpolation")
            print(f"{'='*60}")

            simple_interp = SimpleLocalInterpolator(df, eos)
            interp_data = simple_interp.simple_interpolate(
                method=method, margin=margin, boundary_method=boundary_method,
                use_convex_hull=use_convex_hull, check_method=check_method
            )

            if create_plots:
                simple_interp.plot_simple_interpolation(interp_data, output_dir)

            training_data = simple_interp.generate_training_data(
                n_points=n_points,
                method=method,
                margin=margin,
                boundary_method=boundary_method,
                check_method=check_method
            )

            eos_training_df = pd.DataFrame({
                'eos': [eos] * training_data['n_points'],
                'rho_c': training_data['rho_c'],
                'M': training_data['M'],
                'Omega': training_data['Omega']
            })

            all_training_data.append(eos_training_df)

            eos_file = output_dir / f"simple_local_training_{eos}.parquet"
            eos_training_df.to_parquet(eos_file, index=False)
            print(f"Saved training data to {eos_file}")

        except Exception as e:
            print(f"❌ Failed for {eos}: {e}")
            import traceback
            traceback.print_exc()

    if all_training_data:
        combined_df = pd.concat(all_training_data, ignore_index=True)
        combined_file = output_dir / "simple_local_training_combined.parquet"
        combined_df.to_parquet(combined_file, index=False)

        print(f"\n{'='*60}")
        print("SIMPLE LOCAL INTERPOLATION COMPLETE")
        print(f"{'='*60}")
        print(f"Total training points: {len(combined_df):,}")
        print(f"Original data points: {len(df):,}")
        print(f"Augmentation factor: {len(combined_df) / len(df):.1f}x")
        print(f"Main interpolation method: {method}")
        print(f"Boundary interpolation method: {boundary_method}")
        print(f"Physical boundary margin: {margin * 100:.1f}%")
        print(f"Local neighborhood: 2+2 neighbors per point")

        return combined_df
    else:
        raise ValueError("Failed to generate any training data")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple local interpolation with 2+2 neighbor boundaries")
    parser.add_argument('data_file', help='Path to parquet file')
    parser.add_argument('--method', choices=['rbf', 'multiquadric', 'cubic', 'linear', 'nearest'], default='rbf',
                       help='Interpolation method (rbf recommended)')
    parser.add_argument('--boundary-method', choices=['linear', 'bilinear', 'cubic', 'hermite'], 
                       default='hermite', help='Boundary interpolation method (default: hermite for smooth curves)')
    parser.add_argument('--n-points', type=int, default=50000,
                       help='Training points per EOS')
    parser.add_argument('--margin', type=float, default=0.05, 
                        help='Safety margin outside boundaries (fraction)')
    parser.add_argument('--output', '-o', default='./simple_local_interpolation',
                       help='Output directory')
    parser.add_argument('--no-plots', action='store_true', 
                       help='Skip creating plots')
    parser.add_argument('--no-convex-hull', action='store_true', 
                       help='Disable convex hull masking (use only physical boundaries)')
    parser.add_argument('--check-method', choices=['simple', 'gradient_aware', 'distance_based'], 
                       default='simple', help='Boundary checking method (default: simple)')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_parquet(args.data_file)
    print(f"Loaded {len(df)} neutron star models")
    
    # Process all EOS
    training_df = process_all_eos(
        df, args.output, args.method, args.boundary_method,
        args.n_points, args.margin, not args.no_plots,
        not args.no_convex_hull, args.check_method
    )