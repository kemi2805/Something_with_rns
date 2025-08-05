# rns_driver/parallel/strategies.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from ..core.eos_collection import EOSCollection
from ..config.settings import RNSConfig


class ProcessingStrategy(ABC):
    """Abstract base class for EOS processing strategies."""
    
    @abstractmethod
    def process(self, eos_path: Path, config: RNSConfig) -> pd.DataFrame:
        """Process a single EOS file."""
        pass


class StandardProcessingStrategy(ProcessingStrategy):
    """Standard processing: find TOV, then compute sequences."""
    
    def process(self, eos_path: Path, config: RNSConfig) -> pd.DataFrame:
        """Process EOS file using standard approach."""
        from ..core.eos_catalog import EOSCatalog
        
        catalog = EOSCatalog(config)
        collection = catalog._process_single_eos(eos_path)
        
        return collection.df


class AdaptiveProcessingStrategy(ProcessingStrategy):
    """Adaptive processing with dynamic parameter adjustment."""
    
    def __init__(self, 
                 initial_rho_range: tuple = (5e14, 8e15),
                 initial_r_ratio_step: float = 0.01):
        self.initial_rho_range = initial_rho_range
        self.initial_r_ratio_step = initial_r_ratio_step
    
    def process(self, eos_path: Path, config: RNSConfig) -> pd.DataFrame:
        """Process with adaptive parameter refinement."""
        from ..core.eos_catalog import EOSCatalog
        from ..solvers.optimization import find_tov_configuration
        
        catalog = EOSCatalog(config)
        
        # First pass: coarse search for TOV
        rho_tov = find_tov_configuration(
            eos_path, 
            self.initial_rho_range,
            config
        )
        
        # Adaptive refinement based on EOS stiffness
        collection = EOSCollection(str(eos_path))
        
        # Sample to determine EOS characteristics
        test_star = catalog.solver.solve(eos_path, rho_tov, {'static': True})
        if test_star and test_star.M > 2.0:
            # Stiff EOS - use finer steps
            r_ratio_step = self.initial_r_ratio_step / 2
        else:
            r_ratio_step = self.initial_r_ratio_step
        
        # Process with adapted parameters
        collection.traverse_r_ratio(
            rho_tov, 
            r_ratio_step,
            initial_stepsize_rho_c=1e13
        )
        
        return collection.df