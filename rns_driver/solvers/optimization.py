# rns_driver/solvers/optimization.py
import numpy as np
from scipy.optimize import minimize_scalar, brent
from typing import Tuple, Optional, Callable
from pathlib import Path
import logging

from ..core.neutron_star import NeutronStar
from .rns_solver import RNSSolver
from ..config.settings import RNSConfig



def find_tov_configuration(eos_path: Path,
                          rho_bounds: Tuple[float, float] = (5e14, 8e15),
                          config: Optional[RNSConfig] = None) -> float:
    """
    Find the TOV (maximum mass) configuration for an EOS.
    
    Args:
        eos_path: Path to EOS file
        rho_bounds: Search bounds for central density
        config: RNS configuration
    
    Returns:
        Central density at TOV
    """
    if config is None:
        config = RNSConfig()
    
    solver = RNSSolver(config)
    logger = logging.getLogger(__name__)
    
    # History for polynomial fitting fallback
    history = []
    
    def objective(rho_c):
        """Negative mass for minimization."""
        star = solver.solve(eos_path, rho_c, {'static': True})
        
        if star is None or not star.is_valid:
            return 1e10  # Large penalty
        
        history.append((rho_c, star.M))
        return -star.M
    
    # Try Brent's method first
    try:
        result = minimize_scalar(
            objective,
            bracket=rho_bounds,
            method='brent',
            options={'xtol': 1e-14}
        )
        
        if result.success:
            return result.x
            
    except Exception as e:
        logger.warning(f"Brent optimization failed: {e}")
    
    # Fallback: polynomial fitting
    if len(history) >= 5:
        rho_vals, mass_vals = zip(*sorted(history))
        
        # Fit polynomial and find maximum
        coeffs = np.polyfit(rho_vals, mass_vals, 3)
        poly = np.poly1d(coeffs)
        
        # Find critical points
        crit_points = poly.deriv().r
        real_crits = [p.real for p in crit_points if p.imag == 0 and rho_bounds[0] < p.real < rho_bounds[1]]
        
        if real_crits:
            # Check which is maximum
            max_rho = max(real_crits, key=lambda x: poly(x))
            return max_rho
    
    # Last resort: grid search
    logger.warning("Using grid search for TOV")
    rho_grid = np.linspace(rho_bounds[0], rho_bounds[1], 20)
    masses = []
    
    for rho in rho_grid:
        star = solver.solve(eos_path, rho, {'static': True})
        if star and star.is_valid:
            masses.append(star.M)
        else:
            masses.append(0)
    
    max_idx = np.argmax(masses)
    return rho_grid[max_idx]


def find_mass_shedding_limit(eos_path: Path,
                           rho_c: float,
                           config: Optional[RNSConfig] = None) -> Optional[NeutronStar]:
    """
    Find the mass-shedding (Keplerian) configuration.
    
    Args:
        eos_path: Path to EOS file
        rho_c: Central density
        config: RNS configuration
    
    Returns:
        Keplerian neutron star or None
    """
    if config is None:
        config = RNSConfig()
    
    solver = RNSSolver(config)
    return solver.solve(eos_path, rho_c, {'kepler': True})


def binary_search_parameter(eos_path: Path,
                          rho_c: float,
                          target_param: str,
                          target_value: float,
                          param_bounds: Tuple[float, float],
                          config: Optional[RNSConfig] = None,
                          tolerance: float = 1e-4) -> Optional[NeutronStar]:
    """
    Binary search for a specific parameter value.
    
    Args:
        eos_path: Path to EOS file
        rho_c: Central density
        target_param: Parameter to match ('M', 'Omega', etc.)
        target_value: Desired value
        param_bounds: Search bounds for the parameter
        config: RNS configuration
        tolerance: Relative tolerance
    
    Returns:
        Neutron star with target parameter or None
    """
    if config is None:
        config = RNSConfig()
    
    solver = RNSSolver(config)
    logger = logging.getLogger(__name__)
    
    low, high = param_bounds
    
    for iteration in range(50):  # Max iterations
        mid = (low + high) / 2
        
        # Construct constraint
        if target_param == 'r_ratio':
            constraint = {'r_ratio': mid}
        elif target_param == 'Omega':
            constraint = {'Omega': mid}
        else:
            logger.error(f"Unsupported parameter: {target_param}")
            return None
        
        star = solver.solve(eos_path, rho_c, constraint)
        
        if star is None or not star.is_valid:
            # Adjust bounds
            if target_param == 'r_ratio':
                low = mid  # Invalid means we went too low
            else:
                high = mid
            continue
        
        # Check convergence
        actual_value = getattr(star, target_param)
        relative_error = abs(actual_value - target_value) / target_value
        
        if relative_error < tolerance:
            return star
        
        # Adjust bounds
        if actual_value < target_value:
            low = mid
        else:
            high = mid
    
    logger.warning(f"Failed to converge for {target_param}={target_value}")
    return None