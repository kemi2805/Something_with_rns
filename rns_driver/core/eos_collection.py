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
                    max_invalid: int = 3,
                    use_adaptive_step: bool = True,
                    min_rho_c: float = 1e13) -> pd.DataFrame:
        """
        Generate a sequence of models with fixed constraint.
        
        Args:
            rho_c_start: Starting central density
            constraint: Fixed constraint (e.g., {'r_ratio': 0.8})
            solver: RNS solver instance
            eos_path: Path to EOS file
            stepsize: Initial step size for rho_c (or fixed step if not adaptive)
            max_invalid: Stop after this many invalid models
            use_adaptive_step: If True, use adaptive stepping. If False, use fixed stepsize
            min_rho_c: Minimum central density to consider
        
        Returns:
            DataFrame with the sequence
        """
        rho_c = rho_c_start
        invalid_count = 0
        sequence_stars = []
        
        # Step size management
        current_stepsize = stepsize
        previous_M = None
        
        self.logger.debug(f"Starting sequence with rho_c={rho_c:.3e}, stepsize={stepsize:.3e}, "
                         f"adaptive={'ON' if use_adaptive_step else 'OFF'}")
        
        while invalid_count < max_invalid and rho_c > min_rho_c:
            star = solver.solve(eos_path, rho_c, constraint)
            
            if star is None or not star.is_valid:
                invalid_count += 1
                rho_c -= current_stepsize
                continue
            
            invalid_count = 0
            sequence_stars.append(star)
            self.add_star(star)
            
            # Handle step size
            if use_adaptive_step and previous_M is not None:
                # Adaptive step size based on mass gradient
                gradient = abs(star.M - previous_M) / current_stepsize
                if gradient > 0:
                    # Smaller steps where mass changes rapidly
                    new_stepsize = min(stepsize, 0.1 / gradient)
                    # Add some bounds to prevent too small/large steps
                    new_stepsize = max(stepsize / 100, min(stepsize * 10, new_stepsize))
                    current_stepsize = new_stepsize
                    self.logger.debug(f"Adaptive step: gradient={gradient:.3e}, new_step={current_stepsize:.3e}")
            else:
                # Fixed step size
                current_stepsize = stepsize
            
            previous_M = star.M
            rho_c -= current_stepsize
        
        if sequence_stars:
            return pd.DataFrame([s.to_dict() for s in sequence_stars])
        else:
            return pd.DataFrame()
    
    def get_mesh_grid(self,
                     rho_c_values: np.ndarray,
                     r_ratio_values: np.ndarray,
                     solver: RNSSolver,
                     eos_path: Path,
                     parallel: bool = False) -> pd.DataFrame:
        """
        Generate a regular mesh grid of neutron star models.
        
        Args:
            rho_c_values: Array of central density values
            r_ratio_values: Array of r_ratio values
            solver: RNS solver instance
            eos_path: Path to EOS file
            parallel: If True, use parallel processing (requires setup)
        
        Returns:
            DataFrame with all computed models in mesh grid
        """
        mesh_stars = []
        total_points = len(rho_c_values) * len(r_ratio_values)
        computed = 0
        
        self.logger.info(f"Computing mesh grid: {len(rho_c_values)} rho_c × {len(r_ratio_values)} r_ratio "
                        f"= {total_points} points")
        
        for r_ratio in r_ratio_values:
            for rho_c in rho_c_values:
                # Handle special cases
                if r_ratio == 1.0:
                    constraint = {'static': True}
                else:
                    constraint = {'r_ratio': r_ratio}
                
                star = solver.solve(eos_path, rho_c, constraint)
                
                if star and star.is_valid:
                    mesh_stars.append(star)
                    self.add_star(star)
                else:
                    # Add placeholder for invalid configurations
                    self.logger.debug(f"Invalid configuration at rho_c={rho_c:.3e}, r_ratio={r_ratio:.3f}")
                
                computed += 1
                if computed % 100 == 0:
                    self.logger.info(f"Progress: {computed}/{total_points} points computed")
        
        if mesh_stars:
            return pd.DataFrame([s.to_dict() for s in mesh_stars])
        else:
            return pd.DataFrame()
    
    def traverse_r_ratio(self,
                        rho_tov: float,
                        solver: RNSSolver,
                        eos_path: Path,
                        initial_r_ratio_step: float = 0.01,
                        min_r_ratio: float = 0.5,
                        use_adaptive_step: bool = True,
                        rho_c_stepsize: float = 1e13,
                        rho_c_range: Optional[tuple] = None) -> None:
        """
        Traverse different rotation rates from static to Keplerian.
        
        Args:
            rho_tov: TOV central density
            solver: RNS solver instance
            eos_path: Path to EOS file
            initial_r_ratio_step: Initial step for r_ratio
            min_r_ratio: Minimum r_ratio to consider
            use_adaptive_step: If True, use adaptive stepping for rho_c
            rho_c_stepsize: Step size for rho_c (fixed or initial if adaptive)
            rho_c_range: Optional (min, max) range for rho_c values to explore
        """
        r_ratio = 1.0  # Start with non-rotating
        r_ratio_step = initial_r_ratio_step
        
        # Determine rho_c range to explore
        if rho_c_range:
            min_rho_c, max_rho_c = rho_c_range
        else:
            # Default: explore from TOV down to 10% of TOV density
            max_rho_c = rho_tov * 1.2  # Go slightly above TOV
            min_rho_c = rho_tov * 0.1
        
        while r_ratio >= min_r_ratio:
            self.logger.info(f"Processing r_ratio = {r_ratio:.3f}")
            
            # Get sequence for this r_ratio
            sequence = self.get_sequence(
                max_rho_c,
                {'r_ratio': r_ratio} if r_ratio < 1.0 else {'static': True},
                solver,
                eos_path,
                stepsize=rho_c_stepsize,
                use_adaptive_step=use_adaptive_step,
                min_rho_c=min_rho_c
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
    
    def generate_uniform_mesh(self,
                            solver: RNSSolver,
                            eos_path: Path,
                            n_rho_c: int = 50,
                            n_r_ratio: int = 20,
                            rho_c_range: Optional[tuple] = None,
                            r_ratio_range: tuple = (0.5, 1.0)) -> pd.DataFrame:
        """
        Generate a uniform mesh of neutron star models.
        
        Args:
            solver: RNS solver instance
            eos_path: Path to EOS file
            n_rho_c: Number of rho_c points
            n_r_ratio: Number of r_ratio points
            rho_c_range: (min, max) range for rho_c. If None, auto-determine
            r_ratio_range: (min, max) range for r_ratio
        
        Returns:
            DataFrame with mesh grid models
        """
        # Auto-determine rho_c range if not provided
        if rho_c_range is None:
            # Find TOV configuration first
            from ..solvers.optimization import find_tov_configuration
            rho_tov = find_tov_configuration(eos_path, (5e14, 8e15))
            rho_c_range = (rho_tov * 0.1, rho_tov * 1.2)
        
        # Create uniform grids
        rho_c_values = np.linspace(rho_c_range[0], rho_c_range[1], n_rho_c)
        r_ratio_values = np.linspace(r_ratio_range[0], r_ratio_range[1], n_r_ratio)
        
        self.logger.info(f"Generating uniform mesh: "
                        f"rho_c=[{rho_c_range[0]:.2e}, {rho_c_range[1]:.2e}] with {n_rho_c} points, "
                        f"r_ratio=[{r_ratio_range[0]:.2f}, {r_ratio_range[1]:.2f}] with {n_r_ratio} points")
        
        return self.get_mesh_grid(rho_c_values, r_ratio_values, solver, eos_path)
    
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