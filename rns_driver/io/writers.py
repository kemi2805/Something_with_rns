# rns_driver/io/writers.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import logging
import h5py


class DataWriter:
    """Write neutron star data in various formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def write_parquet(self,
                     df: pd.DataFrame,
                     filepath: Path,
                     compression: str = 'snappy') -> None:
        """Write DataFrame to parquet file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(filepath, compression=compression, index=False)
        self.logger.info(f"Wrote {len(df)} rows to {filepath}")
    
    def write_hdf5(self,
                  df: pd.DataFrame,
                  filepath: Path,
                  key: str = 'neutron_stars') -> None:
        """Write DataFrame to HDF5 file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_hdf(filepath, key=key, mode='w', complevel=9)
        self.logger.info(f"Wrote {len(df)} rows to {filepath} (key={key})")
    
    def write_csv(self,
                 df: pd.DataFrame,
                 filepath: Path,
                 **kwargs) -> None:
        """Write DataFrame to CSV file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False, **kwargs)
        self.logger.info(f"Wrote {len(df)} rows to {filepath}")
    
    def append_parquet(self,
                      df: pd.DataFrame,
                      filepath: Path) -> None:
        """Append to existing parquet file."""
        if filepath.exists():
            existing = pd.read_parquet(filepath)
            combined = pd.concat([existing, df], ignore_index=True)
            self.write_parquet(combined, filepath)
        else:
            self.write_parquet(df, filepath)
    
    def write_summary(self,
                     df: pd.DataFrame,
                     filepath: Path) -> None:
        """Write summary statistics to text file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("Neutron Star Catalog Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write(f"Total models: {len(df)}\n")
            f.write(f"Number of EOS: {df['eos'].nunique()}\n\n")
            
            # Per-EOS statistics
            for eos in df['eos'].unique():
                eos_df = df[df['eos'] == eos]
                f.write(f"\nEOS: {eos}\n")
                f.write("-" * 30 + "\n")
                f.write(f"  Models: {len(eos_df)}\n")
                f.write(f"  Max mass: {eos_df['M'].max():.3f} M☉\n")
                f.write(f"  Max radius: {eos_df['R'].max():.1f} km\n")
                f.write(f"  Max frequency: {eos_df['Omega'].max():.0f} × 10⁴ s⁻¹\n")