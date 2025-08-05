# rns_driver/core/eos_collection.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

from .neutron_star import NeutronStar
from ..solvers.rns_solver import RNSSolver
from ..config.settings import RNSConfig


class EOSCollection:
    """Collection of neutron star models for a single EOS."""
    
    def __init__(self, eos_name: str):
        self.eos_name = eos_name
        self.df = pd.DataFrame()
        self.logger = logging.getLogger(__name__)
        
        # Metadata
        self.tov_mass: Optional[float] = None
        self.tov_radius: Optional[float] = None
        self.max_spin_freq: Optional[float] = None
    
    def add_star(self, star: NeutronStar) -> None:
        """Add a neutron star to the collection."""
        star_df = pd.DataFrame([star.to_dict()])
        
        if self.df.empty:
            self.df = star_df
        else:
            self.df = pd.concat([self.df, star_df], ignore_index=True)
    
    def get_sequence(self,
                    rho_c_start: float,
                    constraint: Dict[str, Any],
                    solver: RNSSolver,
                    eos_path: Path,
                    stepsize: float = 1e13,
                    max_invalid: int = 3) -> pd.DataFrame:
        """
        Generate a sequence of models with fixed constraint.
        
        Args:
            rho_c_start: Starting central density
            constraint: Fixed constraint (e.g., {'r_ratio': 0.8})
            solver: RNS solver instance
            eos_path: Path to EOS file
            stepsize: Initial step size for rho_c
            max_invalid: Stop after this many invalid models
        
        Returns:
            DataFrame with the sequence
        """
        rho_c = rho_c_start
        invalid_count = 0
        sequence_stars = []
        
        # Adaptive step size
        current_stepsize = stepsize
        previous_M = None
        
        while invalid_count < max_invalid and rho_c > 1e13:
            star = solver.solve(eos_path, rho_c, constraint)
            
            if star is None or not star.is_valid:
                invalid_count += 1
                rho_c -= current_stepsize
                continue
            
            invalid_count = 0
            sequence_stars.append(star)
            self.add_star(star)
            
            # Adapt step size based on mass gradient
            if previous_M is not None:
                gradient = abs(star.M - previous_M) / current_stepsize
                if gradient > 0:
                    # Smaller steps where mass changes rapidly
                    current_stepsize = min(stepsize, 0.1 / gradient)
            
            previous_M = star.M
            rho_c -= current_stepsize
        
        if sequence_stars:
            return pd.DataFrame([s.to_dict() for s in sequence_stars])
        else:
            return pd.DataFrame()
    
    def traverse_r_ratio(self,
                        rho_tov: float,
                        solver: RNSSolver,
                        eos_path: Path,
                        initial_r_ratio_step: float = 0.01,
                        min_r_ratio: float = 0.5) -> None:
        """
        Traverse different rotation rates from static to Keplerian.
        
        Args:
            rho_tov: TOV central density
            solver: RNS solver instance
            eos_path: Path to EOS file
            initial_r_ratio_step: Initial step for r_ratio
            min_r_ratio: Minimum r_ratio to consider
        """
        r_ratio = 1.0  # Start with non-rotating
        r_ratio_step = initial_r_ratio_step
        
        while r_ratio >= min_r_ratio:
            self.logger.info(f"Processing r_ratio = {r_ratio:.3f}")
            
            # Get sequence for this r_ratio
            sequence = self.get_sequence(
                rho_tov,
                {'r_ratio': r_ratio},
                solver,
                eos_path
            )
            
            if sequence.empty:
                # Try smaller step
                r_ratio_step /= 2
                if r_ratio_step < 1e-4:
                    break
                r_ratio += r_ratio_step  # Go back
                continue
            
            # Check for Keplerian limit
            if self._check_keplerian_limit(sequence, solver, eos_path):
                self.logger.info(f"Reached Keplerian limit at r_ratio = {r_ratio}")
                break
            
            r_ratio -= r_ratio_step
        
        # Update metadata
        self._update_metadata()
    
    def _check_keplerian_limit(self, 
                              sequence: pd.DataFrame,
                              solver: RNSSolver,
                              eos_path: Path) -> bool:
        """Check if sequence is near Keplerian limit."""
        if sequence.empty:
            return False
        
        # Check the maximum mass configuration
        max_mass_idx = sequence['M'].idxmax()
        max_mass_star = sequence.iloc[max_mass_idx]
        
        # Get Keplerian frequency for this configuration
        kep_star = solver.solve(
            eos_path, 
            max_mass_star['rho_c'],
            {'kepler': True}
        )
        
        if kep_star is None:
            return False
        
        # Check if we're close to Keplerian
        return max_mass_star['Omega'] > 0.95 * kep_star.Omega
    
    def _update_metadata(self) -> None:
        """Update collection metadata."""
        if self.df.empty:
            return
        
        # Find TOV configuration
        static_df = self.df[self.df['r_ratio'] == 1.0]
        if not static_df.empty:
            tov_idx = static_df['M'].idxmax()
            self.tov_mass = static_df.iloc[tov_idx]['M']
            self.tov_radius = static_df.iloc[tov_idx]['R']
        
        # Find maximum spin
        if 'Omega' in self.df.columns:
            self.max_spin_freq = self.df['Omega'].max()
    
    def get_mass_shedding_sequence(self) -> pd.DataFrame:
        """Get the mass-shedding (Keplerian) sequence."""
        if self.df.empty:
            return pd.DataFrame()
        
        # For each r_ratio, find the maximum mass configuration
        kep_sequence = []
        
        for r_ratio, group in self.df.groupby('r_ratio'):
            if len(group) > 0:
                max_idx = group['M'].idxmax()
                kep_sequence.append(group.loc[max_idx])
        
        return pd.DataFrame(kep_sequence)
    
    def filter_by_central_density(self, 
                                 min_rho: float = -1, 
                                 max_rho: float = -1) -> pd.DataFrame:
        """Filter models by central density range."""
        df = self.df
        
        if min_rho > 0:
            df = df[df['rho_c'] >= min_rho]
        if max_rho > 0:
            df = df[df['rho_c'] <= max_rho]
        
        return df