#!/usr/bin/env python3
"""
Neutron Star Visualization Script
=================================

Plot rho_c vs M colored by Omega (angular velocity)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

def plot_neutron_stars(parquet_file, output_dir="./plots", figsize=(12, 8)):
    """
    Create beautiful plots of neutron star models.
    
    Args:
        parquet_file: Path to the neutron star data file
        output_dir: Directory to save plots
        figsize: Figure size (width, height)
    """
    # Read the data
    print(f"Loading data from {parquet_file}...")
    df = pd.read_parquet(parquet_file)
    print(f"Loaded {len(df):,} neutron star models")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up the plot style
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 11
    
    # Create the main plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with Omega as color
    scatter = ax.scatter(
        df['rho_c'] / 1e15,  # Convert to 10^15 g/cm³ for readability
        df['M'], 
        c=df['Omega'], 
        cmap='viridis',
        alpha=0.7,
        s=8,  # Point size
        edgecolors='none'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Angular Velocity Ω (×10⁴ s⁻¹)', fontsize=12)
    
    # Labels and title
    ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)', fontsize=12)
    ax.set_ylabel('Gravitational Mass M (M☉)', fontsize=12)
    ax.set_title('Neutron Star Mass vs Central Density\n(Colored by Angular Velocity)', 
                fontsize=14, pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save the plot
    main_plot_path = output_dir / "neutron_stars_main.png"
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved main plot to {main_plot_path}")
    
    # Show the plot
    plt.show()
    
    # Create separate plots for each EOS
    if 'eos' in df.columns and df['eos'].nunique() > 1:
        plot_by_eos(df, output_dir)
    
    # Create rotation sequence plot
    plot_rotation_sequences(df, output_dir)

def plot_by_eos(df, output_dir):
    """Create separate plots for each EOS."""
    print("\nCreating per-EOS plots...")
    
    eos_list = df['eos'].unique()
    n_eos = len(eos_list)
    
    # Create subplots
    if n_eos <= 2:
        fig, axes = plt.subplots(1, n_eos, figsize=(6*n_eos, 6))
        if n_eos == 1:
            axes = [axes]
    else:
        ncols = min(3, n_eos)
        nrows = int(np.ceil(n_eos / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
        axes = axes.flatten() if n_eos > 1 else [axes]
    
    for i, eos in enumerate(eos_list):
        if i >= len(axes):
            break
            
        eos_df = df[df['eos'] == eos]
        ax = axes[i]
        
        scatter = ax.scatter(
            eos_df['rho_c'] / 1e15,
            eos_df['M'],
            c=eos_df['Omega'],
            cmap='viridis',
            alpha=0.7,
            s=10,
            edgecolors='none'
        )
        
        ax.set_xlabel('Central Density ρc (×10¹⁵ g/cm³)')
        ax.set_ylabel('Gravitational Mass M (M☉)')
        ax.set_title(f'EOS: {eos}\nMax Mass: {eos_df["M"].max():.3f} M☉')
        ax.grid(True, alpha=0.3)
        
        # Add colorbar to each subplot
        plt.colorbar(scatter, ax=ax, label='Ω (×10⁴ s⁻¹)')
    
    # Remove empty subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    eos_plot_path = output_dir / "neutron_stars_by_eos.png"
    plt.savefig(eos_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved EOS comparison plot to {eos_plot_path}")
    plt.show()

def plot_rotation_sequences(df, output_dir):
    """Plot mass vs rotation frequency."""
    print("\nCreating rotation sequence plot...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot for each EOS
    for eos in df['eos'].unique():
        eos_df = df[df['eos'] == eos]
        
        # Plot M vs Omega
        ax.scatter(
            eos_df['Omega'],
            eos_df['M'],
            label=f'{eos}',
            alpha=0.7,
            s=8
        )
    
    ax.set_xlabel('Angular Velocity Ω (×10⁴ s⁻¹)')
    ax.set_ylabel('Gravitational Mass M (M☉)')
    ax.set_title('Neutron Star Mass vs Angular Velocity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    rotation_plot_path = output_dir / "mass_vs_rotation.png"
    plt.savefig(rotation_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved rotation plot to {rotation_plot_path}")
    plt.show()

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Plot neutron star models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot the main results file
  python plot_stars.py results/neutron_stars_20250806_105158.parquet
  
  # Specify custom output directory
  python plot_stars.py results/neutron_stars_20250806_105158.parquet --output plots/
  
  # Custom figure size
  python plot_stars.py results/neutron_stars_20250806_105158.parquet --figsize 15 10
        """
    )
    
    parser.add_argument('data_file', 
                       help='Path to neutron star parquet file')
    parser.add_argument('--output', '-o', default='./plots',
                       help='Output directory for plots (default: ./plots)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[12, 8],
                       help='Figure size as width height (default: 12 8)')
    
    args = parser.parse_args()
    
    # Check if file exists
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"Error: File {data_file} not found!")
        print("Available files in results directory:")
        results_dir = Path("results")
        if results_dir.exists():
            for f in results_dir.glob("*.parquet"):
                print(f"  {f}")
        return 1
    
    # Create plots
    try:
        plot_neutron_stars(
            data_file, 
            output_dir=args.output,
            figsize=tuple(args.figsize)
        )
        print(f"\n🎉 Plotting completed successfully!")
        print(f"Check the {args.output} directory for your plots.")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())