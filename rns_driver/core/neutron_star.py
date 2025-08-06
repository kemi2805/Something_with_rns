# rns_driver/core/neutron_star.py
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import numpy as np # type: ignore
from pathlib import Path


@dataclass
class NeutronStar:
    """Represents a single neutron star model."""
    
    # Required parameters
    eos: str
    rho_c: float  # Central energy density [g/cm³]
    
    # Computed properties
    M: float = -1.0  # Gravitational mass [M☉]
    M_0: float = -1.0  # Rest mass [M☉]
    R: float = -1.0  # Radius at equator [km]
    Omega: float = -1.0  # Angular velocity [10⁴ s⁻¹]
    Omega_p: float = -1.0  # Keplerian angular velocity [10⁴ s⁻¹]
    T_W: float = -1.0  # T/W ratio
    J: float = -1.0  # Angular momentum [GM☉²/c]
    I: float = -1.0  # Moment of inertia
    Phi_2: float = -1.0  # Quadrupole moment
    h_plus: float = -1.0  # Height of last stable co-rotating orbit
    h_minus: float = -1.0  # Height of last stable counter-rotating orbit
    Z_p: float = -1.0  # Polar redshift
    Z_b: float = -1.0  # Backward equatorial redshift
    Z_f: float = -1.0  # Forward equatorial redshift
    omega_c_over_Omega: float = -1.0  # ωc/Ω ratio
    r_e: float = -1.0  # Coordinate equatorial radius
    r_ratio: float = -1.0  # Axes ratio (polar/equatorial)
    
    # Additional computed properties
    Omega_pa: float = -1.0
    Omega_plus: float = -1.0
    u_phi: float = -1.0
    
    # Metadata
    computation_time: Optional[float] = None
    convergence_iterations: Optional[int] = None
    
    def __post_init__(self):
        """Validate the neutron star parameters."""
        if self.rho_c <= 0:
            raise ValueError(f"Central density must be positive, got {self.rho_c}")
    
    @property
    def is_valid(self) -> bool:
        """
        Check if the neutron star model is physically valid.
        Accept -1 as "undefined" rather than invalid.
        """
        # Essential parameters that must be positive (not -1)
        if self.M <= 0 or self.R <= 0:
            return False
        
        # Reasonable physical bounds
        if self.M > 5.0:  # Unreasonably high mass
            return False
        
        if self.R > 50.0:  # Unreasonably large radius
            return False
        
        # For rotating stars, check r_ratio if it's defined
        if self.is_rotating and self.r_ratio > 0:
            if self.r_ratio > 1.0:
                return False
        
        return True
    
    @property
    def is_rotating(self) -> bool:
        """Check if the star is rotating."""
        return self.Omega > 0
    
    @property
    def rotation_parameter(self) -> float:
        """Dimensionless rotation parameter Ω/Ωk."""
        if self.Omega_p > 0:
            return self.Omega / self.Omega_p
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NeutronStar':
        """Create from dictionary."""
        return cls(**data)