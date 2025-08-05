# rns_driver/analysis/sequences.py
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import logging


class SequenceAnalyzer:
    """Analyze neutron star sequences."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def find_turning_points(self, 
                          df: pd.DataFrame,
                          x_col: str = 'rho_c',
                          y_col: str = 'M') -> List[Dict[str, float]]:
        """
        Find turning points in sequences.
        
        Returns:
            List of dicts with turning point information
        """
        turning_points = []
        
        for (eos, r_ratio), group in df.groupby(['eos', 'r_ratio']):
            if len(group) < 3:
                continue
            
            group = group.sort_values(x_col)
            x = group[x_col].values
            y = group[y_col].values
            
            # Find local maxima and minima
            for i in range(1, len(y) - 1):
                if y[i] > y[i-1] and y[i] > y[i+1]:
                    turning_points.append({
                        'eos': eos,
                        'r_ratio': r_ratio,
                        'type': 'maximum',
                        x_col: x[i],
                        y_col: y[i],
                        'index': group.index[i]
                    })
                elif y[i] < y[i-1] and y[i] < y[i+1]:
                    turning_points.append({
                        'eos': eos,
                        'r_ratio': r_ratio,
                        'type': 'minimum',
                        x_col: x[i],
                        y_col: y[i],
                        'index': group.index[i]
                    })
        
        return turning_points
    
    def interpolate_sequence(self,
                           df: pd.DataFrame,
                           x_col: str = 'rho_c',
                           y_col: str = 'M',
                           n_points: int = 100) -> pd.DataFrame:
        """
        Interpolate sequences for smooth curves.
        
        Returns:
            DataFrame with interpolated sequences
        """
        interpolated = []
        
        for (eos, r_ratio), group in df.groupby(['eos', 'r_ratio']):
            if len(group) < 3:
                continue
            
            group = group.sort_values(x_col)
            x = group[x_col].values
            y = group[y_col].values
            
            # Create interpolation function
            f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
            
            # Generate interpolated points
            x_new = np.linspace(x.min(), x.max(), n_points)
            y_new = f(x_new)
            
            # Create DataFrame
            for xi, yi in zip(x_new, y_new):
                interpolated.append({
                    'eos': eos,
                    'r_ratio': r_ratio,
                    x_col: xi,
                    y_col: yi,
                    'interpolated': True
                })
        
        return pd.DataFrame(interpolated)
    
    def find_maximum_masses(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Find maximum mass for each r_ratio.
        
        Returns:
            DataFrame with maximum mass configurations
        """
        max_masses = []
        
        for (eos, r_ratio), group in df.groupby(['eos', 'r_ratio']):
            max_idx = group['M'].idxmax()
            max_star = group.loc[max_idx].to_dict()
            max_masses.append(max_star)
        
        return pd.DataFrame(max_masses)
    
    def compute_universal_relations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute universal relations (e.g., I-Love-Q).
        
        Returns:
            Dictionary with fitted relations
        """
        # Filter for valid data
        valid_df = df[(df['I'] > 0) & (df['Phi_2'] > 0) & (df['M'] > 0)]
        
        if len(valid_df) < 10:
            self.logger.warning("Insufficient data for universal relations")
            return {}
        
        # Dimensionless quantities
        I_bar = valid_df['I'] / (valid_df['M']**3)
        Q_bar = -valid_df['Phi_2'] / (valid_df['M']**3)
        
        # Fit I-Q relation: ln(I_bar) = a + b * ln(Q_bar)
        from scipy.stats import linregress
        
        mask = (I_bar > 0) & (Q_bar > 0)
        if mask.sum() < 10:
            return {}
        
        result = linregress(np.log(Q_bar[mask]), np.log(I_bar[mask]))
        
        return {
            'I_Q_relation': {
                'a': np.exp(result.intercept),
                'b': result.slope,
                'r_squared': result.rvalue**2,
                'formula': f'I_bar = {np.exp(result.intercept):.3f} * Q_bar^{result.slope:.3f}'
            }
        }
    
    def find_constant_frequency_sequences(self, 
                                        df: pd.DataFrame,
                                        frequencies: List[float]) -> pd.DataFrame:
        """
        Extract constant frequency sequences by interpolation.
        
        Args:
            df: DataFrame with neutron star models
            frequencies: List of frequencies to extract
        
        Returns:
            DataFrame with constant frequency models
        """
        const_freq_models = []
        
        for eos in df['eos'].unique():
            eos_df = df[df['eos'] == eos]
            
            for freq in frequencies:
                # Find models that bracket this frequency
                for r_ratio in eos_df['r_ratio'].unique():
                    seq = eos_df[eos_df['r_ratio'] == r_ratio].sort_values('Omega')
                    
                    if len(seq) < 2:
                        continue
                    
                    # Check if frequency is in range
                    if seq['Omega'].min() <= freq <= seq['Omega'].max():
                        # Interpolate to find model at exact frequency
                        # This is simplified - real implementation would interpolate all quantities
                        f_M = interp1d(seq['Omega'], seq['M'], kind='linear')
                        f_rho = interp1d(seq['Omega'], seq['rho_c'], kind='linear')
                        
                        const_freq_models.append({
                            'eos': eos,
                            'Omega': freq,
                            'M': float(f_M(freq)),
                            'rho_c': float(f_rho(freq)),
                            'r_ratio': r_ratio
                        })
        
        return pd.DataFrame(const_freq_models)

