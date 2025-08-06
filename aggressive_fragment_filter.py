#!/usr/bin/env python3
"""
Aggressive Fragment Filter
=========================

More aggressive filtering to remove all problematic fragments
that cause interpolation issues.
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from pathlib import Path

class AggressiveFragmentFilter:
    """More aggressive fragment filtering for smooth interpolation."""
    
    def __init__(self):
        self.name = "AggressiveFragmentFilter"
    
    def detect_sequence_breaks(self, df: pd.DataFrame, eos: str) -> List[int]:
        """
        Detect breaks in rotation sequences that shouldn't exist.
        
        For each r_ratio, the rho_c should form a continuous sequence.
        """
        eos_df = df[df['eos'] == eos].copy()
        remove_indices = []
        
        print(f"Analyzing sequence breaks for {eos}...")
        
        # Group by r_ratio
        for r_ratio, group in eos_df.groupby('r_ratio'):
            if len(group) < 5:  # Skip small groups
                remove_indices.extend(group.index.tolist())
                continue
            
            # Sort by rho_c
            group_sorted = group.sort_values('rho_c')
            
            # Check for large gaps in rho_c
            rho_vals = group_sorted['rho_c'].values
            rho_diffs = np.diff(rho_vals)
            
            # Find unusually large gaps
            median_diff = np.median(rho_diffs)
            large_gap_threshold = median_diff * 10  # 10x the median gap
            
            large_gaps = rho_diffs > large_gap_threshold
            
            if np.any(large_gaps):
                # Remove points after large gaps (likely fragments)
                gap_indices = np.where(large_gaps)[0]
                for gap_idx in gap_indices:
                    # Remove points after the gap
                    points_to_remove = group_sorted.iloc[gap_idx+1:].index.tolist()
                    remove_indices.extend(points_to_remove)
                    print(f"  r_ratio={r_ratio:.3f}: Removing {len(points_to_remove)} points after large gap")
        
        return remove_indices
    
    def detect_mass_anomalies(self, df: pd.DataFrame, eos: str) -> List[int]:
        """
        Detect mass values that are inconsistent with physics.
        
        Mass should generally increase with rotation up to a maximum,
        then potentially decrease at very high rotation.
        """
        eos_df = df[df['eos'] == eos].copy()
        remove_indices = []
        
        print(f"Analyzing mass anomalies for {eos}...")
        
        # For each rho_c value, check mass behavior across rotation
        rho_values = eos_df['rho_c'].unique()
        
        for rho_c in rho_values:
            rho_group = eos_df[eos_df['rho_c'] == rho_c]
            if len(rho_group) < 3:
                continue
            
            # Sort by r_ratio (decreasing rotation)
            rho_group_sorted = rho_group.sort_values('r_ratio', ascending=False)
            
            # Check for non-physical mass jumps
            mass_vals = rho_group_sorted['M'].values
            mass_diffs = np.diff(mass_vals)
            
            # Mass shouldn't have huge jumps between adjacent rotation rates
            mass_std = np.std(mass_vals)
            jump_threshold = mass_std * 3  # 3 sigma jumps
            
            large_jumps = np.abs(mass_diffs) > jump_threshold
            
            if np.any(large_jumps):
                jump_indices = np.where(large_jumps)[0]
                for jump_idx in jump_indices:
                    # Remove the point after the jump (usually the problematic one)
                    problematic_idx = rho_group_sorted.index[jump_idx + 1]
                    remove_indices.append(problematic_idx)
        
        print(f"  Found {len(remove_indices)} mass anomalies")
        return remove_indices
    
    def detect_sparse_regions(self, df: pd.DataFrame, eos: str, 
                             min_neighbors: int = 10,
                             neighbor_radius: float = 0.05) -> List[int]:
        """
        Remove points in very sparse regions of parameter space.
        """
        eos_df = df[df['eos'] == eos].copy()
        
        if len(eos_df) < min_neighbors * 2:
            return []
        
        print(f"Analyzing sparse regions for {eos}...")
        
        # Normalize coordinates
        rho_norm = (eos_df['rho_c'] - eos_df['rho_c'].min()) / (eos_df['rho_c'].max() - eos_df['rho_c'].min())
        M_norm = (eos_df['M'] - eos_df['M'].min()) / (eos_df['M'].max() - eos_df['M'].min())
        
        # Build KD-tree
        points = np.column_stack([rho_norm, M_norm])
        tree = cKDTree(points)
        
        # Count neighbors within radius for each point
        neighbor_counts = []
        for point in points:
            neighbors = tree.query_ball_point(point, neighbor_radius)
            neighbor_counts.append(len(neighbors) - 1)  # Exclude self
        
        neighbor_counts = np.array(neighbor_counts)
        
        # Remove points with too few neighbors
        sparse_mask = neighbor_counts < min_neighbors
        remove_indices = eos_df.index[sparse_mask].tolist()
        
        print(f"  Found {sum(sparse_mask)} points in sparse regions")
        return remove_indices
    
    def detect_outlier_omega(self, df: pd.DataFrame, eos: str) -> List[int]:
        """
        Remove points with physically impossible Omega values.
        """
        eos_df = df[df['eos'] == eos].copy()
        remove_indices = []
        
        print(f"Analyzing Omega outliers for {eos}...")
        
        # Group by r_ratio and check for Omega consistency
        for r_ratio, group in eos_df.groupby('r_ratio'):
            if len(group) < 5:
                continue
            
            omega_vals = group['Omega'].values
            
            # Remove points with Omega = 0 when r_ratio < 1 (should be rotating)
            if r_ratio < 0.99:  # Rotating stars
                zero_omega_mask = np.abs(omega_vals) < 1e-6
                if np.any(zero_omega_mask):
                    zero_indices = group.index[zero_omega_mask].tolist()
                    remove_indices.extend(zero_indices)
                    print(f"  r_ratio={r_ratio:.3f}: Removing {sum(zero_omega_mask)} points with Omega≈0")
            
            # Remove points with unreasonably high Omega
            omega_median = np.median(omega_vals[omega_vals > 0])
            if omega_median > 0:
                high_omega_threshold = omega_median * 5  # 5x median
                high_omega_mask = omega_vals > high_omega_threshold
                if np.any(high_omega_mask):
                    high_indices = group.index[high_omega_mask].tolist()
                    remove_indices.extend(high_indices)
                    print(f"  r_ratio={r_ratio:.3f}: Removing {sum(high_omega_mask)} points with very high Omega")
        
        return remove_indices

def aggressive_clean(df: pd.DataFrame, 
                    min_neighbors: int = 15,
                    neighbor_radius: float = 0.03,
                    min_sequence_length: int = 20) -> pd.DataFrame:
    """
    Aggressively clean neutron star data for smooth interpolation.
    
    Args:
        df: Input DataFrame
        min_neighbors: Minimum neighbors in sparse region detection
        neighbor_radius: Radius for neighbor counting
        min_sequence_length: Minimum points per r_ratio sequence
    
    Returns:
        Aggressively cleaned DataFrame
    """
    print("Starting aggressive fragment removal...")
    
    filter_obj = AggressiveFragmentFilter()
    all_remove_indices = []
    
    # Process each EOS separately
    for eos in df['eos'].unique():
        print(f"\n{'='*50}")
        print(f"Processing {eos}")
        print(f"{'='*50}")
        
        eos_df = df[df['eos'] == eos]
        print(f"Starting with {len(eos_df)} points")
        
        eos_remove_indices = []
        
        # 1. Remove broken sequences
        sequence_remove = filter_obj.detect_sequence_breaks(df, eos)
        eos_remove_indices.extend(sequence_remove)
        
        # 2. Remove mass anomalies  
        mass_remove = filter_obj.detect_mass_anomalies(df, eos)
        eos_remove_indices.extend(mass_remove)
        
        # 3. Remove sparse regions
        sparse_remove = filter_obj.detect_sparse_regions(df, eos, min_neighbors, neighbor_radius)
        eos_remove_indices.extend(sparse_remove)
        
        # 4. Remove Omega outliers
        omega_remove = filter_obj.detect_outlier_omega(df, eos)
        eos_remove_indices.extend(omega_remove)
        
        # 5. Remove short r_ratio sequences
        r_ratio_counts = eos_df.groupby('r_ratio').size()
        short_sequences = r_ratio_counts[r_ratio_counts < min_sequence_length].index
        for r_ratio in short_sequences:
            short_indices = eos_df[eos_df['r_ratio'] == r_ratio].index.tolist()
            eos_remove_indices.extend(short_indices)
            print(f"Removing short sequence r_ratio={r_ratio:.3f} ({len(short_indices)} points)")
        
        # Combine and deduplicate
        eos_remove_indices = list(set(eos_remove_indices))
        all_remove_indices.extend(eos_remove_indices)
        
        remaining_points = len(eos_df) - len(eos_remove_indices)
        print(f"Total removed for {eos}: {len(eos_remove_indices)}")
        print(f"Remaining points: {remaining_points}")
        print(f"Retention rate: {remaining_points/len(eos_df)*100:.1f}%")
    
    # Remove all flagged indices
    all_remove_indices = list(set(all_remove_indices))
    cleaned_df = df.drop(all_remove_indices).reset_index(drop=True)
    
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Original points: {len(df):,}")
    print(f"Removed points: {len(all_remove_indices):,}")
    print(f"Final clean points: {len(cleaned_df):,}")
    print(f"Overall retention rate: {len(cleaned_df)/len(df)*100:.1f}%")
    
    return cleaned_df

def test_interpolation_compatibility(df: pd.DataFrame) -> Dict[str, bool]:
    """
    Test if each EOS can be interpolated without errors.
    
    Returns:
        Dict mapping EOS names to interpolation success
    """
    results = {}
    
    for eos in df['eos'].unique():
        eos_df = df[df['eos'] == eos]
        
        if len(eos_df) < 10:
            results[eos] = False
            print(f"{eos}: Too few points ({len(eos_df)})")
            continue
        
        try:
            # Test simple linear interpolation
            from scipy.interpolate import griddata
            
            points = eos_df[['rho_c', 'M']].values
            values = eos_df['Omega'].values
            
            # Create small test grid
            rho_min, rho_max = eos_df['rho_c'].min(), eos_df['rho_c'].max()
            M_min, M_max = eos_df['M'].min(), eos_df['M'].max()
            
            rho_test = np.linspace(rho_min, rho_max, 10)
            M_test = np.linspace(M_min, M_max, 10)
            rho_mesh, M_mesh = np.meshgrid(rho_test, M_test)
            
            # Try interpolation
            omega_interp = griddata(points, values, (rho_mesh, M_mesh), method='linear')
            
            results[eos] = True
            print(f"{eos}: Interpolation compatible ✓ ({len(eos_df)} points)")
            
        except Exception as e:
            results[eos] = False
            print(f"{eos}: Interpolation failed ✗ - {str(e)}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggressively clean neutron star data")
    parser.add_argument('data_file', help='Path to neutron star data file')
    parser.add_argument('--output', '-o', default='./ultra_cleaned_data.parquet',
                       help='Output file')
    parser.add_argument('--min-neighbors', type=int, default=15,
                       help='Minimum neighbors for sparse detection')
    parser.add_argument('--test-interpolation', action='store_true',
                       help='Test interpolation compatibility')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_parquet(args.data_file)
    print(f"Loaded {len(df):,} neutron star models")
    
    # Aggressive cleaning
    df_ultra_clean = aggressive_clean(df, min_neighbors=args.min_neighbors)
    
    # Test interpolation if requested
    if args.test_interpolation:
        print(f"\n{'='*50}")
        print("TESTING INTERPOLATION COMPATIBILITY")
        print(f"{'='*50}")
        compatibility = test_interpolation_compatibility(df_ultra_clean)
        
        all_compatible = all(compatibility.values())
        print(f"\nAll EOS interpolation compatible: {'✓' if all_compatible else '✗'}")
    
    # Save ultra-cleaned data
    df_ultra_clean.to_parquet(args.output, index=False)
    print(f"\nSaved ultra-clean data to {args.output}")
    
    # Quick comparison plot
    print("\nGenerating comparison plot...")
    
    fig, axes = plt.subplots(2, len(df['eos'].unique()), figsize=(7*len(df['eos'].unique()), 10))
    if len(df['eos'].unique()) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, eos in enumerate(df['eos'].unique()):
        # Original
        ax = axes[0, i]
        orig_eos = df[df['eos'] == eos]
        ax.scatter(orig_eos['rho_c']/1e15, orig_eos['M'], c=orig_eos['Omega'], 
                  cmap='viridis', s=4, alpha=0.7)
        ax.set_title(f'Original {eos}\n({len(orig_eos)} points)')
        ax.grid(True, alpha=0.3)
        
        # Ultra-cleaned
        ax = axes[1, i]
        clean_eos = df_ultra_clean[df_ultra_clean['eos'] == eos]
        if len(clean_eos) > 0:
            scatter = ax.scatter(clean_eos['rho_c']/1e15, clean_eos['M'], c=clean_eos['Omega'], 
                               cmap='viridis', s=4, alpha=0.7)
            ax.set_title(f'Ultra-Clean {eos}\n({len(clean_eos)} points)')
        else:
            ax.set_title(f'Ultra-Clean {eos}\n(0 points - EOS removed)')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Mass M (M☉)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./aggressive_cleaning_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved comparison plot to ./aggressive_cleaning_comparison.png")