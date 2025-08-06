# rns_driver/config/settings.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import yaml  # type: ignore
import os


@dataclass
class RNSConfig:
    """Configuration for RNS calculations."""
    rns_executable: Path = Path("/home/miler/codes/Something_with_rns/source/rns.v1.1d/rns")
    eos_directory: Path = Path("/home/miler/codes/Something_with_rns/EOS/106")
    output_directory: Path = Path("./output")
    
    # Computation parameters
    timeout: float = 60.0
    tolerance: float = 1e-4
    accuracy: float = 1e-5
    relaxation_factor: float = 1.0
    
    # Parallelization
    max_workers: Optional[int] = None  # None means use all available cores
    chunk_size: int = 10
    
    # Filtering parameters
    outlier_threshold: float = 3.0
    neighbor_tolerance: float = 1e15
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    
    @classmethod
    def from_yaml(cls, path: Path) -> 'RNSConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)