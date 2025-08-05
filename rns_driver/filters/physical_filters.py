# rns_driver/filters/physical_filters.py
import numpy as np
import pandas as pd
from typing import List, Optional

from .base_filter import BaseFilter


class PhysicalBoundsFilter(BaseFilter):
    """Filter based on physical constraints."""
    
    def __init__(self,
                 min_mass: float = 0.1,
                 max_mass: float = 3.5,
                 min_radius: float = 5.0,
                 max_radius: float = 30.0):
        super().__init__("PhysicalBounds")
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.min_radius = min_radius
        self.max_radius = max_radius
    
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove physically impossible models."""
        mask = (
            (df['M'] >= self.min_mass) & 
            (df['M'] <= self.max_mass) &
            (df['R'] >= self.min_radius) & 
            (df['R'] <= self.max_radius)
        )
        return df[mask]
    
    def get_outlier_indices(self, df: pd.DataFrame) -> List[int]:
        """Get indices of physically impossible models."""
        mask = ~(
            (df['M'] >= self.min_mass) & 
            (df['M'] <= self.max_mass) &
            (df['R'] >= self.min_radius) & 
            (df['R'] <= self.max_radius)
        )
        return df[mask].index.tolist()


class CausalityFilter(BaseFilter):
    """Ensure models respect causality constraints."""
    
    def __init__(self):
        super().__init__("Causality")
    
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove models violating causality."""
        outlier_indices = self.get_outlier_indices(df)
        return df.drop(outlier_indices)
    
    def get_outlier_indices(self, df: pd.DataFrame) -> List[int]:
        """Find models where sound speed exceeds c."""
        outliers = []
        
        # Group by EOS and r_ratio
        for (eos, r_ratio), group in df.groupby(['eos', 'r_ratio']):
            if len(group) < 2:
                continue
            
            group = group.sort_values('rho_c')
            
            # Approximate sound speed check
            for i in range(1, len(group)):
                drho = group.iloc[i]['rho_c'] - group.iloc[i-1]['rho_c']
                dP = group.iloc[i]['P'] if 'P' in group.columns else 0
                
                if drho > 0 and dP > 0:
                    # Very rough causality check
                    if dP / drho > 1:  # c²=1 in natural units
                        outliers.append(group.index[i])
        
        return outliers


class MonotonicityFilter(BaseFilter):
    """Ensure proper monotonic behavior."""
    
    def __init__(self, strict: bool = False):
        super().__init__(f"Monotonicity_{'strict' if strict else 'relaxed'}")
        self.strict = strict
    
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove non-monotonic sequences."""
        outlier_indices = self.get_outlier_indices(df)
        return df.drop(outlier_indices)
    
    def get_outlier_indices(self, df: pd.DataFrame) -> List[int]:
        """Find points breaking monotonicity."""
        outliers = []
        
        for (eos, r_ratio), group in df.groupby(['eos', 'r_ratio']):
            if len(group) < 3:
                continue
            
            group = group.sort_values('rho_c').reset_index(drop=True)
            
            # For non-rotating stars, M should increase then decrease
            if r_ratio == 1.0:
                # Find maximum
                max_idx = group['M'].idxmax()
                
                # Check monotonicity before and after maximum
                for i in range(1, max_idx):
                    if group.iloc[i]['M'] < group.iloc[i-1]['M']:
                        outliers.append(group.index[i])
                
                for i in range(max_idx + 1, len(group)):
                    if group.iloc[i]['M'] > group.iloc[i-1]['M']:
                        outliers.append(group.index[i])
        
        return outliers