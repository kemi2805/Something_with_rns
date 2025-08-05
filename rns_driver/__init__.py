"""
RNS Driver - Modern neutron star modeling system
===============================================

A high-performance Python framework for computing neutron star models 
using the RNS (Rapidly rotating Neutron Stars) code.

Main modules:
- core: Core data structures and catalog management
- solvers: RNS solver interface and optimization
- filters: Data filtering and outlier detection
- parallel: Parallel processing framework
- io: Input/output handlers
- analysis: Analysis tools and interpolation
- config: Configuration management
"""

__version__ = "2.0.0"
__author__ = "Your Name"

# Import main classes for convenience
from .core.neutron_star import NeutronStar
from .core.eos_collection import EOSCollection
from .core.eos_catalog import EOSCatalog
from .config.settings import RNSConfig

__all__ = [
    "NeutronStar",
    "EOSCollection", 
    "EOSCatalog",
    "RNSConfig",
]