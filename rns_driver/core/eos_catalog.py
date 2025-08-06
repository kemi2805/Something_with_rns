# rns_driver/core/eos_catalog.py
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports]
import logging

from .eos_collection import EOSCollection
from ..solvers.rns_solver import RNSSolver
from ..solvers.optimization import find_tov_configuration
from ..config.settings import RNSConfig
from ..filters.composite_filters import create_default_filter_pipeline


class EOSCatalog:
    """Manages collections of EOS and neutron star models."""
    
    def __init__(self, config: RNSConfig):
        self.config = config
        self.solver = RNSSolver(config)
        self.collections: Dict[str, EOSCollection] = {}
        self.logger = logging.getLogger(__name__)
    
    def process_eos_directory(self, 
                            eos_dir: Path,
                            filter_pipeline=None) -> pd.DataFrame:
        """
        Process all EOS files in a directory.
        
        Args:
            eos_dir: Directory containing EOS files
            filter_pipeline: Optional filter pipeline to apply
        
        Returns:
            Combined DataFrame with all results
        """
        eos_files = list(eos_dir.glob("*.rns"))
        if not eos_files:
            self.logger.warning(f"No .rns files found in {eos_dir}")
            return pd.DataFrame()
        
        self.logger.info(f"Found {len(eos_files)} EOS files")
        
        # Process each EOS
        all_results = []
        for eos_file in eos_files:
            try:
                collection = self._process_single_eos(eos_file)
                
                # Apply filters if provided
                if filter_pipeline and not collection.df.empty:
                    initial_count = len(collection.df)
                    collection.df = filter_pipeline.filter(collection.df)
                    self.logger.info(
                        f"Filtered {eos_file.stem}: {initial_count} -> {len(collection.df)} models"
                    )
                
                if not collection.df.empty:
                    all_results.append(collection.df)
                    self.collections[eos_file.stem] = collection
                    
            except Exception as e:
                self.logger.error(f"Failed to process {eos_file}: {e}")
        
        # Combine results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            self.logger.info(f"Total models: {len(combined_df)}")
            return combined_df
        else:
            return pd.DataFrame()
    
    def _process_single_eos(self, eos_path: Path) -> EOSCollection:
        """Process a single EOS file."""
        self.logger.info(f"Processing {eos_path.stem}")
        
        # Check EOS file format
        self._validate_eos_file(eos_path)
        
        # Find TOV configuration
        rho_tov = find_tov_configuration(
            eos_path,
            rho_bounds=(5e14, 8e15),
            config=self.config
        )
        
        self.logger.info(f"TOV density for {eos_path.stem}: {rho_tov:.3e} g/cm³")
        
        # Create collection and compute sequences
        collection = EOSCollection(eos_path.stem)
        collection.traverse_r_ratio(
            rho_tov,
            self.solver,
            eos_path,
            initial_r_ratio_step=0.01,
        )
        
        return collection
    
    def _validate_eos_file(self, eos_path: Path) -> None:
        """Validate and potentially fix EOS file format."""
        with open(eos_path, 'r') as f:
            lines = f.readlines()
        
        # Check number of points
        n_points = int(lines[0].strip())
        actual_points = len(lines) - 1
        
        if n_points != actual_points:
            self.logger.warning(
                f"EOS file {eos_path.stem} claims {n_points} points but has {actual_points}"
            )
        
        # Check if we need to reduce to 200 points
        if actual_points > 200:
            self._reduce_eos_points(eos_path, actual_points)
    
    def _reduce_eos_points(self, eos_path: Path, n_points: int) -> None:
        """Reduce EOS to maximum 200 points."""
        self.logger.info(f"Reducing {eos_path.stem} from {n_points} to 200 points")
        
        with open(eos_path, 'r') as f:
            lines = f.readlines()
        
        # Calculate stride
        stride = int(np.ceil(n_points / 200))
        
        # Keep header and select points
        new_lines = [f"200\n"]
        for i in range(1, n_points + 1, stride):
            if i < len(lines):
                new_lines.append(lines[i])
        
        # Ensure we have exactly 200 points
        new_lines = new_lines[:201]  # 1 header + 200 data
        new_lines[0] = f"{len(new_lines) - 1}\n"
        
        # Write back
        with open(eos_path, 'w') as f:
            f.writelines(new_lines)
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """Get summary statistics for all processed EOS."""
        summaries = []
        
        for eos_name, collection in self.collections.items():
            if collection.df.empty:
                continue
            
            summary = {
                'eos': eos_name,
                'n_models': len(collection.df),
                'tov_mass': collection.tov_mass,
                'tov_radius': collection.tov_radius,
                'max_mass': collection.df['M'].max(),
                'max_spin_freq': collection.max_spin_freq,
                'min_radius': collection.df['R'].min(),
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)