# rns_driver/solvers/rns_solver.py
import subprocess
import logging
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
        
        # Find the data line (last line with numbers)
        data_line = None
        for line in reversed(lines):
            if any(char.isdigit() for char in line):
                data_line = line
                break
        
        if not data_line:
            return None
        
        try:
            values = [float(x) for x in data_line.split()]
            
            # Check for NaN values
            if any(np.isnan(v) or np.isinf(v) for v in values):
                self.logger.warning("NaN or Inf values in RNS output")
                return None
            
            # Determine if static or rotating based on number of values
            if len(values) == 17:  # Static star
                return NeutronStar(
                    eos=eos_path.stem,
                    rho_c=rho_c,
                    M=values[0],
                    M_0=values[1],
                    R=values[2],
                    Omega=0.0,
                    Omega_p=0.0,
                    T_W=0.0,
                    J=0.0,
                    I=0.0,
                    Phi_2=values[3],
                    h_plus=values[4],
                    h_minus=values[5],
                    Z_p=values[6],
                    Z_b=values[7],
                    Z_f=values[8],
                    omega_c_over_Omega=values[9],
                    r_e=values[10],
                    r_ratio=1.0
                )
            elif len(values) >= 20:  # Rotating star
                return NeutronStar(
                    eos=eos_path.stem,
                    rho_c=rho_c,
                    M=values[0],
                    M_0=values[1],
                    R=values[2],
                    Omega=values[3],
                    Omega_p=values[4],
                    T_W=values[5],
                    J=values[6],
                    I=values[7],
                    Phi_2=values[8],
                    h_plus=values[9],
                    h_minus=values[10],
                    Z_p=values[11],
                    Z_b=values[12],
                    Z_f=values[13],
                    omega_c_over_Omega=values[14],
                    r_e=values[15],
                    r_ratio=values[16] if len(values) > 16 else 1.0
                )
            else:
                self.logger.error(f"Unexpected number of values: {len(values)}")
                return None
                
        except (ValueError, IndexError) as e:
            self.logger.error(f"Failed to parse RNS output: {e}")
            return None