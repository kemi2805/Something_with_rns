# rns_driver/solvers/rns_solver.py
import subprocess
import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import re

from ..core.neutron_star import NeutronStar
from ..config.settings import RNSConfig



class RNSSolver:
    """Interface to the RNS C executable."""
    
    def __init__(self, config: RNSConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Verify executable exists
        if not self.config.rns_executable.exists():
            raise FileNotFoundError(f"RNS executable not found: {self.config.rns_executable}")
    
    def solve(self, 
              eos_path: Path,
              rho_c: float,
              constraint: Optional[Dict[str, Any]] = None) -> Optional[NeutronStar]:
        """
        Solve for a neutron star model with given parameters.
        
        Args:
            eos_path: Path to EOS file
            rho_c: Central energy density
            constraint: Optional constraint dict with one of:
                - {'r_ratio': float}
                - {'M': float}
                - {'M_0': float}
                - {'Omega': float}
                - {'J': float}
                - {'static': True}
                - {'kepler': True}
        
        Returns:
            NeutronStar object or None if failed
        """
        cmd = self._build_command(eos_path, rho_c, constraint)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            if result.returncode != 0:
                self.logger.error(f"RNS failed with code {result.returncode}: {result.stderr}")
                return None
            
            return self._parse_output(result.stdout, eos_path, rho_c)
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"RNS timeout after {self.config.timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"RNS solver error: {e}")
            return None
    
    def _build_command(self, 
                       eos_path: Path,
                       rho_c: float,
                       constraint: Optional[Dict[str, Any]] = None) -> List[str]:
        """Build the RNS command line."""
        cmd = [
            str(self.config.rns_executable),
            "-f", str(eos_path),
            "-p", "2"  # Compact output format
        ]
        
        if constraint is None:
            # Default to static model
            cmd.extend(["-t", "static", "-e", str(rho_c)])
        elif 'r_ratio' in constraint:
            cmd.extend(["-t", "model", "-e", str(rho_c), "-r", str(constraint['r_ratio'])])
        elif 'M' in constraint:
            cmd.extend(["-t", "gmass", "-e", str(rho_c), "-m", str(constraint['M'])])
        elif 'M_0' in constraint:
            cmd.extend(["-t", "rmass", "-e", str(rho_c), "-z", str(constraint['M_0'])])
        elif 'Omega' in constraint:
            cmd.extend(["-t", "omega", "-e", str(rho_c), "-o", str(constraint['Omega'])])
        elif 'J' in constraint:
            cmd.extend(["-t", "jmoment", "-e", str(rho_c), "-j", str(constraint['J'])])
        elif constraint.get('static'):
            cmd.extend(["-t", "static", "-e", str(rho_c)])
        elif constraint.get('kepler'):
            cmd.extend(["-t", "kepler", "-e", str(rho_c), "-b", str(self.config.tolerance)])
        else:
            raise ValueError(f"Unknown constraint: {constraint}")
        
        # Add optional parameters
        if self.config.tolerance != 1e-4:
            cmd.extend(["-b", str(self.config.tolerance)])
        if self.config.accuracy != 1e-5:
            cmd.extend(["-a", str(self.config.accuracy)])
        if self.config.relaxation_factor != 1.0:
            cmd.extend(["-c", str(self.config.relaxation_factor)])
        
        return cmd
    
    def _parse_output(self, output: str, eos_path: Path, rho_c: float) -> Optional[NeutronStar]:
        """Parse RNS output into NeutronStar object."""
        lines = output.strip().split('\n')
        if not lines:
            return None
        
        # Find the data line (last line with numbers or dashes)
        data_line = None
        for line in reversed(lines):
            # Look for lines with numbers or dashes
            if any(char.isdigit() or char == '-' for char in line):
                data_line = line
                break
        
        if not data_line:
            return None
        
        try:
            # Split and clean the values
            raw_values = data_line.split()
            values = []
            
            for val in raw_values:
                val = val.strip()
                if val == '---' or val == '-' or val == 'nan' or val == 'inf':
                    # RNS couldn't compute this value
                    self.logger.warning(f"RNS returned invalid value: '{val}' for {eos_path.name}")
                    return None
                try:
                    float_val = float(val)
                    if np.isnan(float_val) or np.isinf(float_val):
                        self.logger.warning(f"Invalid numerical value: {float_val}")
                        return None
                    values.append(float_val)
                except ValueError:
                    self.logger.warning(f"Could not parse value: '{val}'")
                    return None
            
            if len(values) < 10:  # Need at least basic parameters
                self.logger.warning(f"Too few valid values: {len(values)}")
                return None
            
            # Determine if static or rotating based on number of values and content
            if len(values) >= 16 and abs(values[3]) < 1e-10:  # Omega ~ 0 means static
                # Static star - reorder values appropriately
                return NeutronStar(
                    eos=eos_path.stem,
                    rho_c=rho_c,
                    M=values[0],           # Mass
                    M_0=values[1],         # Rest mass  
                    R=values[2],           # Radius
                    Omega=0.0,             # Angular velocity (static)
                    Omega_p=0.0,           # Keplerian frequency
                    T_W=0.0,               # T/W ratio
                    J=0.0,                 # Angular momentum
                    I=0.0,                 # Moment of inertia
                    Phi_2=values[8] if len(values) > 8 else 0.0,     # Quadrupole moment
                    h_plus=values[9] if len(values) > 9 else 0.0,    # ISCO height +
                    h_minus=values[10] if len(values) > 10 else 0.0, # ISCO height -
                    Z_p=values[11] if len(values) > 11 else 0.0,     # Polar redshift
                    Z_b=values[12] if len(values) > 12 else 0.0,     # Backward redshift
                    Z_f=values[13] if len(values) > 13 else 0.0,     # Forward redshift
                    omega_c_over_Omega=values[14] if len(values) > 14 else 0.0,
                    r_e=values[15] if len(values) > 15 else values[2], # Coordinate radius
                    r_ratio=1.0            # Static star
                )
                
            elif len(values) >= 16:  # Rotating star
                return NeutronStar(
                    eos=eos_path.stem,
                    rho_c=rho_c,
                    M=values[0],           # Mass
                    M_0=values[1],         # Rest mass
                    R=values[2],           # Radius
                    Omega=values[3],       # Angular velocity
                    Omega_p=values[4],     # Keplerian frequency
                    T_W=values[5],         # T/W ratio
                    J=values[6],           # Angular momentum
                    I=values[7],           # Moment of inertia
                    Phi_2=values[8],       # Quadrupole moment
                    h_plus=values[9],      # ISCO height +
                    h_minus=values[10],    # ISCO height -
                    Z_p=values[11],        # Polar redshift
                    Z_b=values[12],        # Backward redshift
                    Z_f=values[13],        # Forward redshift
                    omega_c_over_Omega=values[14],
                    r_e=values[15],        # Coordinate radius
                    r_ratio=values[16] if len(values) > 16 else 0.8  # Default rotating
                )
            else:
                self.logger.error(f"Unexpected number of values: {len(values)}")
                return None
                
        except (ValueError, IndexError) as e:
            self.logger.error(f"Failed to parse RNS output: {e}")
            self.logger.debug(f"Raw output: {output}")
            return None