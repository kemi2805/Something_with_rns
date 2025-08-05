# rns_driver/io/converters.py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class EOSConverter:
    """Convert between different EOS formats."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def pressure_to_enthalpy(self, 
                           pressure: np.ndarray,
                           energy_density: np.ndarray) -> np.ndarray:
        """
        Calculate enthalpy from pressure and energy density.
        
        h = ∫ c² dP / (ε + P)
        """
        c_squared = 8.98755e20  # cm²/s²
        
        # Numerical integration
        h = np.zeros_like(pressure)
        
        for i in range(1, len(pressure)):
            dp = pressure[i] - pressure[i-1]
            avg_denominator = 0.5 * (
                energy_density[i] + pressure[i] + 
                energy_density[i-1] + pressure[i-1]
            )
            
            if avg_denominator > 0:
                h[i] = h[i-1] + c_squared * dp / avg_denominator
        
        return h
    
    def convert_to_rns_format(self,
                            input_file: Path,
                            output_file: Path,
                            input_format: str = 'two_column') -> None:
        """
        Convert EOS file to RNS format.
        
        Args:
            input_file: Input EOS file
            output_file: Output RNS format file
            input_format: 'two_column' (P vs ε) or other formats
        """
        if input_format == 'two_column':
            # Read pressure and energy density
            data = np.loadtxt(input_file)
            energy_density = data[:, 0]  # g/cm³
            pressure = data[:, 1]  # dyne/cm²
            
            # Calculate enthalpy
            enthalpy = self.pressure_to_enthalpy(pressure, energy_density)
            
            # Estimate baryon density (rough approximation)
            m_b = 1.66e-24  # Baryon mass in g
            baryon_density = energy_density / (m_b * 939.56)  # cm⁻³
            
            # Write RNS format
            with open(output_file, 'w') as f:
                f.write(f"{len(energy_density)}\n")
                
                for i in range(len(energy_density)):
                    f.write(f"{energy_density[i]:.10e} ")
                    f.write(f"{pressure[i]:.10e} ")
                    f.write(f"{enthalpy[i]:.10e} ")
                    f.write(f"{baryon_density[i]:.10e}\n")
            
            self.logger.info(f"Converted {input_file} to RNS format")
        else:
            raise NotImplementedError(f"Format {input_format} not supported")