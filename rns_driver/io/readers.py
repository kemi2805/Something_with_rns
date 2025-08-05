# rns_driver/io/readers.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging


class EOSReader:
    """Read various EOS file formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def read_rns_format(self, filepath: Path) -> pd.DataFrame:
        """
        Read RNS format EOS file.
        
        Format:
        - First line: number of points
        - Remaining lines: energy_density pressure enthalpy baryon_density
        
        Returns:
            DataFrame with columns: rho, P, h, n_b
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        n_points = int(lines[0].strip())
        
        data = []
        for i in range(1, min(n_points + 1, len(lines))):
            values = [float(x) for x in lines[i].split()]
            if len(values) >= 4:
                data.append(values[:4])
        
        df = pd.DataFrame(data, columns=['rho', 'P', 'h', 'n_b'])
        
        # Validate
        if len(df) != n_points:
            self.logger.warning(
                f"Expected {n_points} points but got {len(df)} in {filepath}"
            )
        
        return df
    
    def read_compose_format(self, filepath: Path) -> pd.DataFrame:
        """Read CompOSE format EOS file."""
        # Implement CompOSE format reader
        pass
    
    def convert_units(self, 
                     df: pd.DataFrame,
                     from_units: Dict[str, str],
                     to_units: Dict[str, str]) -> pd.DataFrame:
        """Convert units in EOS DataFrame."""
        # Conversion factors
        conversions = {
            ('MeV/fm3', 'g/cm3'): 1.7827e12,
            ('MeV/fm3', 'dyne/cm2'): 1.6022e33,
            ('fm-3', 'cm-3'): 1e39,
        }
        
        df_converted = df.copy()
        
        for col in df.columns:
            if col in from_units and col in to_units:
                key = (from_units[col], to_units[col])
                if key in conversions:
                    df_converted[col] *= conversions[key]
        
        return df_converted


class ParquetReader:
    """Read and combine parquet files."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def read_directory(self, 
                      directory: Path,
                      pattern: str = "*.parquet") -> pd.DataFrame:
        """Read all parquet files in a directory."""
        files = list(directory.glob(pattern))
        
        if not files:
            self.logger.warning(f"No files matching {pattern} in {directory}")
            return pd.DataFrame()
        
        dfs = []
        for file in files:
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                self.logger.error(f"Failed to read {file}: {e}")
        
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            self.logger.info(f"Combined {len(files)} files: {len(combined)} rows")
            return combined
        else:
            return pd.DataFrame()
    
    def read_chunked(self,
                    filepath: Path,
                    chunk_size: int = 10000) -> pd.DataFrame:
        """Read large parquet file in chunks."""
        # Parquet files are columnar, so we read all at once
        # But we can process in chunks if needed
        df = pd.read_parquet(filepath)
        return df