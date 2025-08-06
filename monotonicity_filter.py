#!/usr/bin/env python3
"""
Monotonicity-Based Fragment Filter
=================================

Filter neutron star data by checking monotonicity along physical sequences:
- Along constant r_ratio: Mass vs rho_c should be smooth
- Along constant rho_c: Mass vs r_ratio should be smooth

Points that violate monotonicity in BOTH directions get flagged for removal.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Set, Tuple
from pathlib import Path

class MonotonicityFilter:
    """Filter based on monotonicity violations along physical sequences."""
    
    def __init__(self, tolerance_factor: float = 2.0):
        """
        Args:
            tolerance_factor: How strict the monotonicity check is
                            Larger values = more permissive
        """
        self.tolerance_factor = tolerance_factor
        self.flagged_points = set()
    
    def flag_violations_along_r_ratio(self, df: pd.DataFrame, eos: str) -> Set[int]:
        """
        Walk along constant r_ratio lines and flag monotonicity violations.
        
        For each r_ratio, Mass should smoothly vary with rho_c (typically increase then decrease).
        """
        eos_df = df[df['eos'] == eos].copy()
        flagged = set()
        
        print(f"Checking r_ratio sequences for {eos}...")
        
        for r_ratio in sorted(eos_df['r_ratio'].unique(), reverse=True):
            r_ratio_group = eos_df[eos_df['r_ratio'] == r_ratio].copy()
            
            if len(r_ratio_group) < 3:
                continue  # Need at least 3 points to check monotonicity
            
            # Sort by rho_c
            r_ratio_group = r_ratio_group.sort_values('rho_c').reset_index(drop=True)
            
            rho_vals = r_ratio_group['rho_c'].values
            M_vals = r_ratio_group['M'].values
            original_indices = r_ratio_group.index.values  # Original DataFrame indices
            
            # Look for obvious outliers using simpler method
            if len(M_vals) >= 3:
                # Calculate local deviations
                for i in range(1, len(M_vals) - 1):
                    current_M = M_vals[i]
                    prev_M = M_vals[i-1]
                    next_M = M_vals[i+1]
                    
                    # Expected value based on neighbors
                    expected_M = (prev_M + next_M) / 2
                    deviation = abs(current_M - expected_M)
                    
                    # Calculate local scale
                    local_scale = abs(next_M - prev_M) / 2 if abs(next_M - prev_M) > 0 else 0.1
                    
                    # Flag points that deviate too much from smooth interpolation
                    if deviation > local_scale * self.tolerance_factor:
                        flagged.add(original_indices[i])
                        print(f"  r_ratio={r_ratio:.3f}: Flagged rho_c={rho_vals[i]:.2e}, M={M_vals[i]:.3f} (local deviation)")
                
                # Also check for mass reversals (mass should generally increase then decrease)
                mass_diffs = np.diff(M_vals)
                
                # Find where mass changes direction unexpectedly
                for i in range(1, len(mass_diffs)):
                    if abs(mass_diffs[i]) > abs(mass_diffs[i-1]) * 3:  # Sudden large change
                        flagged.add(original_indices[i])
                        print(f"  r_ratio={r_ratio:.3f}: Flagged rho_c={rho_vals[i]:.2e}, M={M_vals[i]:.3f} (sudden change)")

                # Find where mass decreases with increasing rotation
                for i in range(1, len(M_vals)):
                    if M_vals[i] < M_vals[i-1]:
                        flagged.add(original_indices[i])
                        print(f"  r_ratio={r_ratio:.3f}: Flagged rho_c={rho_vals[i]:.2e}, M={M_vals[i]:.3f} (mass decrease)")
        
        return flagged
    
    def flag_violations_along_rho_c(self, df: pd.DataFrame, eos: str) -> Set[int]:
        """
        Walk along constant rho_c lines and flag monotonicity violations.
        
        For each rho_c, Mass typically increases with rotation (decreasing r_ratio) up to a limit.
        """
        eos_df = df[df['eos'] == eos].copy()
        flagged = set()
        
        print(f"Checking rho_c sequences for {eos}...")
        
        # Group by rho_c (with tolerance for numerical precision)
        rho_values = sorted(eos_df['rho_c'].unique())
        
        for rho_c in rho_values:
            rho_group = eos_df[eos_df['rho_c'] == rho_c].copy()
            
            if len(rho_group) < 3:
                continue
            
            # Sort by r_ratio (decreasing = increasing rotation)
            rho_group = rho_group.sort_values('r_ratio', ascending=False).reset_index(drop=True)
            
            r_ratio_vals = rho_group['r_ratio'].values
            M_vals = rho_group['M'].values
            Omega_vals = rho_group['Omega'].values
            original_indices = rho_group.index.values  # Original DataFrame indices
            
            # Check for local outliers along the sequence
            if len(M_vals) >= 3:
                for i in range(1, len(M_vals) - 1):
                    current_M = M_vals[i]
                    prev_M = M_vals[i-1]
                    next_M = M_vals[i+1]
                    
                    # Expected value based on neighbors
                    expected_M = (prev_M + next_M) / 2
                    deviation = abs(current_M - expected_M)
                    
                    # Calculate local scale
                    local_scale = abs(next_M - prev_M) / 2 if abs(next_M - prev_M) > 0 else 0.1
                    
                    # Flag points that deviate too much
                    if deviation > local_scale * self.tolerance_factor:
                        flagged.add(original_indices[i])
                        print(f"  rho_c={rho_c:.2e}: Flagged r_ratio={r_ratio_vals[i]:.3f}, M={M_vals[i]:.3f} (mass anomaly)")
                    
                    # Also check for physically impossible Omega values
                    if Omega_vals[i] < 0 or (r_ratio_vals[i] < 0.99 and Omega_vals[i] == 0):
                        flagged.add(original_indices[i])
                        print(f"  rho_c={rho_c:.2e}: Flagged r_ratio={r_ratio_vals[i]:.3f} (bad Omega={Omega_vals[i]:.3f})")
                    
                    # Find where mass decreases with increasing rotation
                    if i > 0 and M_vals[i] < M_vals[i-1]:
                        flagged.add(original_indices[i])
                        print(f"  rho_c={rho_c:.2e}: Flagged r_ratio={r_ratio_vals[i]:.3f}, M={M_vals[i]:.3f} (mass decrease)")
        
        return flagged
    
    def iterative_filter(self, df: pd.DataFrame, n_iterations: int = 3) -> pd.DataFrame:
        """
        Apply the monotonicity filter iteratively.
        
        Points must be flagged in BOTH r_ratio AND rho_c directions to be removed.
        """
        current_df = df.copy()
        
        print(f"Starting iterative monotonicity filtering ({n_iterations} iterations)")
        print(f"Initial data: {len(current_df)} points")
        
        for iteration in range(n_iterations):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration + 1}/{n_iterations}")
            print(f"{'='*60}")
            print(f"Current DataFrame has {len(current_df)} points with index range {current_df.index.min()}-{current_df.index.max()}")
            
            all_flagged_r_ratio = set()
            all_flagged_rho_c = set()
            
            for eos in current_df['eos'].unique():
                print(f"\nProcessing {eos} (iteration {iteration + 1})...")
                eos_count = len(current_df[current_df['eos'] == eos])
                print(f"Current points for {eos}: {eos_count}")
                
                # Flag violations in both directions
                r_ratio_flagged = self.flag_violations_along_r_ratio(current_df, eos)
                rho_c_flagged = self.flag_violations_along_rho_c(current_df, eos)
                
                all_flagged_r_ratio.update(r_ratio_flagged)
                all_flagged_rho_c.update(rho_c_flagged)
                
                print(f"  Flagged in r_ratio direction: {len(r_ratio_flagged)} -> {sorted(list(r_ratio_flagged))[:5]}{'...' if len(r_ratio_flagged) > 5 else ''}")
                print(f"  Flagged in rho_c direction: {len(rho_c_flagged)} -> {sorted(list(rho_c_flagged))[:5]}{'...' if len(rho_c_flagged) > 5 else ''}")
            
            # Remove points flagged in BOTH directions
            doubly_flagged = all_flagged_r_ratio.intersection(all_flagged_rho_c)
            
            print(f"\nFlagged in r_ratio direction: {len(all_flagged_r_ratio)} total")
            print(f"Flagged in rho_c direction: {len(all_flagged_rho_c)} total")
            print(f"Flagged in BOTH directions: {len(doubly_flagged)} -> {sorted(list(doubly_flagged))[:10]}{'...' if len(doubly_flagged) > 10 else ''}")
            
            if len(doubly_flagged) == 0:
                print(f"\nNo points flagged in both directions. Stopping at iteration {iteration + 1}")
                break
            
            print(f"\nRemoving {len(doubly_flagged)} points flagged in both directions")
            
            # Verify the indices exist before dropping
            valid_indices = [idx for idx in doubly_flagged if idx in current_df.index]
            invalid_indices = doubly_flagged - set(valid_indices)
            
            if invalid_indices:
                print(f"WARNING: {len(invalid_indices)} flagged indices not found in current DataFrame: {sorted(list(invalid_indices))[:5]}")
            
            # Remove the flagged points
            current_df = current_df.drop(valid_indices)
            
            print(f"Successfully removed {len(valid_indices)} points")
            print(f"Points remaining: {len(current_df)}")
            
            # DO NOT reset index - keep original indices for next iteration
        
        print(f"\n{'='*60}")
        print("FILTERING COMPLETE")
        print(f"{'='*60}")
        print(f"Original points: {len(df)}")
        print(f"Final points: {len(current_df)}")
        print(f"Removed: {len(df) - len(current_df)}")
        print(f"Retention rate: {len(current_df)/len(df)*100:.1f}%")
        
        # Reset index only at the very end for clean output
        return current_df.reset_index(drop=True)

def plot_filtering_progress(df_original: pd.DataFrame, 
                          df_filtered: pd.DataFrame,
                          output_dir: str = "./plots"):
    """Plot before and after the monotonicity filtering."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    eos_list = df_original['eos'].unique()
    n_eos = len(eos_list)
    
    fig, axes = plt.subplots(2, n_eos, figsize=(7*n_eos, 10))
    if n_eos == 1:
        axes = axes.reshape(-1, 1)
    
    for i, eos in enumerate(eos_list):
        # Original
        ax = axes[0, i]
        orig_eos = df_original[df_original['eos'] == eos]
        scatter = ax.scatter(orig_eos['rho_c']/1e15, orig_eos['M'], 
                           c=orig_eos['Omega'], cmap='viridis', s=4, alpha=0.7)
        ax.set_title(f'Original: {eos}\n({len(orig_eos)} points)')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Mass M (M☉)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Ω (×10⁴ s⁻¹)')
        
        # Filtered
        ax = axes[1, i]
        filt_eos = df_filtered[df_filtered['eos'] == eos]
        if len(filt_eos) > 0:
            scatter = ax.scatter(filt_eos['rho_c']/1e15, filt_eos['M'], 
                               c=filt_eos['Omega'], cmap='viridis', s=4, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Ω (×10⁴ s⁻¹)')
        ax.set_title(f'Monotonicity Filtered: {eos}\n({len(filt_eos)} points)')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Mass M (M☉)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / "monotonicity_filtering_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {plot_path}")
    plt.show()

def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Filter neutron star data using monotonicity checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic filtering
  python monotonicity_filter.py cleaned_data.parquet
  
  # More aggressive filtering
  python monotonicity_filter.py cleaned_data.parquet --tolerance 1.5 --iterations 4
  
  # Save to custom file with plots
  python monotonicity_filter.py cleaned_data.parquet --output mono_filtered.parquet --plot
        """
    )
    
    parser.add_argument('data_file', 
                       help='Path to neutron star parquet file')
    parser.add_argument('--output', '-o', default='./monotonicity_filtered.parquet',
                       help='Output file for filtered data')
    parser.add_argument('--tolerance', type=float, default=2.0,
                       help='Tolerance factor for monotonicity (higher = more permissive)')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of filtering iterations')
    parser.add_argument('--plot', action='store_true',
                       help='Generate before/after comparison plots')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_parquet(args.data_file)
    print(f"Loaded {len(df):,} neutron star models")
    
    # Create filter and apply
    filter_obj = MonotonicityFilter(tolerance_factor=args.tolerance)
    df_filtered = filter_obj.iterative_filter(df, n_iterations=args.iterations)
    
    # Save filtered data
    df_filtered.to_parquet(args.output, index=False)
    print(f"\nSaved filtered data to {args.output}")
    
    # Create comparison plots if requested
    if args.plot:
        print("\nGenerating comparison plots...")
        plot_filtering_progress(df, df_filtered)

if __name__ == "__main__":
    main()