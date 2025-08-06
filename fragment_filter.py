#!/usr/bin/env python3
"""
Fragment Detection and Removal for Neutron Star Data
===================================================

Detect and remove isolated fragments in the rho_c vs M space.
"""

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from typing import List, Tuple

class FragmentFilter:
    """Filter to detect and remove isolated data fragments."""
    
    def __init__(self):
        self.name = "FragmentFilter"
    
    def detect_fragments(self, df: pd.DataFrame, 
                        eos: str = None,
                        min_cluster_size: int = 50,
                        eps: float = 0.05) -> List[int]:
        """
        Detect fragments using density-based clustering.
        
        Args:
            df: DataFrame with neutron star data
            eos: Specific EOS to analyze (None for all)
            min_cluster_size: Minimum points in a valid cluster
            eps: DBSCAN epsilon parameter (distance threshold)
        
        Returns:
            List of indices to remove
        """
        if eos:
            work_df = df[df['eos'] == eos].copy()
        else:
            work_df = df.copy()
        
        if len(work_df) < min_cluster_size:
            return []
        
        # Normalize coordinates for clustering
        rho_norm = (work_df['rho_c'] - work_df['rho_c'].min()) / (work_df['rho_c'].max() - work_df['rho_c'].min())
        M_norm = (work_df['M'] - work_df['M'].min()) / (work_df['M'].max() - work_df['M'].min())
        
        # Prepare data for clustering
        X = np.column_stack([rho_norm, M_norm])
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=10).fit(X)
        labels = clustering.labels_
        
        # Find the largest cluster (main sequence)
        unique_labels = np.unique(labels)
        cluster_sizes = [(label, np.sum(labels == label)) for label in unique_labels if label != -1]
        
        if not cluster_sizes:
            return []  # No clusters found
        
        # Keep only the largest clusters
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Determine which clusters to keep (main + significant secondary clusters)
        keep_labels = []
        total_points = len(work_df)
        
        for label, size in cluster_sizes:
            if size >= min_cluster_size or size >= 0.1 * total_points:
                keep_labels.append(label)
        
        # Mark points to remove
        remove_mask = ~np.isin(labels, keep_labels)
        remove_indices = work_df.index[remove_mask].tolist()
        
        print(f"Found {len(unique_labels)} clusters for {eos if eos else 'all EOS'}")
        print(f"Keeping {len(keep_labels)} main clusters, removing {sum(remove_mask)} fragment points")
        
        return remove_indices
    
    def filter_by_connectivity(self, df: pd.DataFrame, 
                              eos: str = None,
                              distance_threshold: float = 0.05) -> List[int]:
        """
        Remove points that are far from their nearest neighbors.
        
        Args:
            df: DataFrame with neutron star data
            eos: Specific EOS to analyze
            distance_threshold: Maximum normalized distance to nearest neighbor
        
        Returns:
            List of indices to remove
        """
        if eos:
            work_df = df[df['eos'] == eos].copy()
        else:
            work_df = df.copy()
        
        if len(work_df) < 10:
            return []
        
        # Normalize coordinates
        rho_norm = (work_df['rho_c'] - work_df['rho_c'].min()) / (work_df['rho_c'].max() - work_df['rho_c'].min())
        M_norm = (work_df['M'] - work_df['M'].min()) / (work_df['M'].max() - work_df['M'].min())
        
        # Build KD-tree for efficient neighbor search
        points = np.column_stack([rho_norm, M_norm])
        tree = cKDTree(points)
        
        # Find distances to 5th nearest neighbor for each point
        distances, _ = tree.query(points, k=6)  # k=6 because first is the point itself
        avg_distances = np.mean(distances[:, 1:], axis=1)  # Average of 2nd-6th nearest
        
        # Mark points with unusually large distances as fragments
        distance_threshold_abs = np.percentile(avg_distances, 90) * 2  # Adaptive threshold
        fragment_mask = avg_distances > distance_threshold_abs
        
        remove_indices = work_df.index[fragment_mask].tolist()
        
        print(f"Connectivity filter: removing {sum(fragment_mask)} isolated points for {eos if eos else 'all EOS'}")
        
        return remove_indices

def clean_neutron_star_data(df: pd.DataFrame, 
                           min_cluster_size: int = 50,
                           dbscan_eps: float = 0.05,
                           connectivity_threshold: float = 0.05) -> pd.DataFrame:
    """
    Comprehensive cleaning of neutron star data to remove fragments.
    
    Args:
        df: Input DataFrame
        min_cluster_size: Minimum size for valid clusters
        dbscan_eps: DBSCAN epsilon parameter
        connectivity_threshold: Distance threshold for connectivity filter
    
    Returns:
        Cleaned DataFrame
    """
    print("Starting fragment detection and removal...")
    
    fragment_filter = FragmentFilter()
    all_remove_indices = []
    
    # Process each EOS separately
    for eos in df['eos'].unique():
        print(f"\nProcessing {eos}...")
        
        # Apply DBSCAN clustering filter
        dbscan_remove = fragment_filter.detect_fragments(
            df, eos, min_cluster_size, dbscan_eps
        )
        
        # Apply connectivity filter
        connectivity_remove = fragment_filter.filter_by_connectivity(
            df, eos, connectivity_threshold
        )
        
        # Combine removal lists
        eos_remove = list(set(dbscan_remove + connectivity_remove))
        all_remove_indices.extend(eos_remove)
    
    # Remove fragments
    all_remove_indices = list(set(all_remove_indices))
    cleaned_df = df.drop(all_remove_indices).reset_index(drop=True)
    
    print(f"\nFragment removal summary:")
    print(f"Original points: {len(df):,}")
    print(f"Removed fragments: {len(all_remove_indices):,}")
    print(f"Final clean points: {len(cleaned_df):,}")
    print(f"Retention rate: {len(cleaned_df)/len(df)*100:.1f}%")
    
    return cleaned_df

def interpolate_surface(df: pd.DataFrame, eos: str, 
                       resolution: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create smooth interpolated surface from cleaned data.
    
    Args:
        df: Cleaned DataFrame
        eos: EOS name to interpolate
        resolution: Grid resolution for interpolation
    
    Returns:
        rho_grid, M_grid, omega_grid for plotting
    """
    eos_df = df[df['eos'] == eos]
    
    # Create regular grid
    rho_min, rho_max = eos_df['rho_c'].min(), eos_df['rho_c'].max()
    M_min, M_max = eos_df['M'].min(), eos_df['M'].max()
    
    rho_grid = np.linspace(rho_min, rho_max, resolution)
    M_grid = np.linspace(M_min, M_max, resolution)
    rho_mesh, M_mesh = np.meshgrid(rho_grid, M_grid)
    
    # Interpolate Omega values
    points = eos_df[['rho_c', 'M']].values
    values = eos_df['Omega'].values
    
    # Use linear interpolation with nearest-neighbor extrapolation
    omega_interp = griddata(
        points, values, 
        (rho_mesh, M_mesh), 
        method='linear', 
        fill_value=0
    )
    
    return rho_mesh / 1e15, M_mesh, omega_interp

def plot_before_after_cleaning(df_original: pd.DataFrame, 
                              df_cleaned: pd.DataFrame,
                              output_dir: str = "./plots"):
    """Plot comparison before and after fragment removal."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for i, (eos, df_orig, df_clean) in enumerate(zip(
        df_original['eos'].unique(),
        [df_original[df_original['eos'] == eos] for eos in df_original['eos'].unique()],
        [df_cleaned[df_cleaned['eos'] == eos] for eos in df_cleaned['eos'].unique()]
    )):
        if i >= 2:
            break
            
        # Original data
        ax = axes[0, i]
        scatter = ax.scatter(df_orig['rho_c']/1e15, df_orig['M'], 
                           c=df_orig['Omega'], cmap='viridis', s=8, alpha=0.7)
        ax.set_title(f'Original: {eos}\n({len(df_orig)} points)')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Mass M (M☉)')
        plt.colorbar(scatter, ax=ax)
        
        # Cleaned data
        ax = axes[1, i]
        scatter = ax.scatter(df_clean['rho_c']/1e15, df_clean['M'], 
                           c=df_clean['Omega'], cmap='viridis', s=8, alpha=0.7)
        ax.set_title(f'Cleaned: {eos}\n({len(df_clean)} points)')
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Mass M (M☉)')
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / "fragment_cleaning_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean neutron star data fragments")
    parser.add_argument('data_file', help='Path to neutron star parquet file')
    parser.add_argument('--output', '-o', default='./cleaned_data.parquet',
                       help='Output file for cleaned data')
    parser.add_argument('--min-cluster-size', type=int, default=50,
                       help='Minimum cluster size to keep')
    parser.add_argument('--plot', action='store_true',
                       help='Show before/after plots')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_parquet(args.data_file)
    print(f"Loaded {len(df)} neutron star models")
    
    # Clean data
    df_cleaned = clean_neutron_star_data(
        df, 
        min_cluster_size=args.min_cluster_size
    )
    
    # Save cleaned data
    df_cleaned.to_parquet(args.output, index=False)
    print(f"Saved cleaned data to {args.output}")
    
    # Plot comparison if requested
    if args.plot:
        plot_before_after_cleaning(df, df_cleaned)