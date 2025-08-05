# rns_driver/filters/outlier_filters.py
import numpy as np
import pandas as pd
from typing import List, Optional, Literal
from scipy import stats

from .base_filter import BaseFilter


class StatisticalOutlierFilter(BaseFilter):
    """Remove statistical outliers based on standard deviation or IQR."""
    
    def __init__(self, 
                 columns: List[str] = ['rho_c', 'M'],
                 threshold: float = 3.0,
                 method: Literal['mean', 'median'] = 'mean'):
        super().__init__(f"Statistical_{method}_{threshold}σ")
        self.columns = columns
        self.threshold = threshold
        self.method = method
    
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from DataFrame."""
        outlier_indices = self.get_outlier_indices(df)
        return df.drop(outlier_indices)
    
    def get_outlier_indices(self, df: pd.DataFrame) -> List[int]:
        """Identify outliers using statistical methods."""
        outliers = []
        
        for col in self.columns:
            if col not in df.columns:
                continue
            
            values = df[col]
            
            if self.method == 'mean':
                center = values.mean()
                spread = values.std()
                outlier_mask = np.abs(values - center) >= self.threshold * spread
            else:  # median
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                center = values.median()
                outlier_mask = np.abs(values - center) >= self.threshold * iqr
            
            outliers.extend(df[outlier_mask].index.tolist())
        
        return list(set(outliers))


class NeighborDistanceFilter(BaseFilter):
    """Filter based on distance from neighboring points."""
    
    def __init__(self, 
                 group_by: str = 'r_ratio',
                 tolerance: float = 0.1,
                 normalize: bool = True):
        super().__init__(f"NeighborDist_{tolerance}")
        self.group_by = group_by
        self.tolerance = tolerance
        self.normalize = normalize
    
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove points far from neighbors."""
        outlier_indices = self.get_outlier_indices(df)
        return df.drop(outlier_indices)
    
    def get_outlier_indices(self, df: pd.DataFrame) -> List[int]:
        """Find points that are far from their neighbors."""
        all_outliers = []
        
        for group_value, group_df in df.groupby(self.group_by):
            if len(group_df) < 3:
                continue
            
            # Sort by rho_c for consistent neighbor finding
            group_df = group_df.sort_values('rho_c').reset_index(drop=True)
            
            # Normalize if requested
            if self.normalize:
                rho_norm = group_df['rho_c'] / group_df['rho_c'].max()
                M_norm = group_df['M'] / group_df['M'].max()
            else:
                rho_norm = group_df['rho_c']
                M_norm = group_df['M']
            
            outliers = []
            for i in range(len(group_df)):
                distances = []
                
                # Distance to previous neighbor
                if i > 0:
                    dist = np.sqrt(
                        (rho_norm.iloc[i] - rho_norm.iloc[i-1])**2 + 
                        (M_norm.iloc[i] - M_norm.iloc[i-1])**2
                    )
                    distances.append(dist)
                
                # Distance to next neighbor
                if i < len(group_df) - 1:
                    dist = np.sqrt(
                        (rho_norm.iloc[i] - rho_norm.iloc[i+1])**2 + 
                        (M_norm.iloc[i] - M_norm.iloc[i+1])**2
                    )
                    distances.append(dist)
                
                # Check if any distance exceeds tolerance
                if distances and min(distances) > self.tolerance:
                    outliers.append(group_df.index[i])
            
            all_outliers.extend(outliers)
        
        return all_outliers


class GradientFilter(BaseFilter):
    """Filter based on gradient changes."""
    
    def __init__(self,
                 group_by: str = 'r_ratio',
                 max_gradient_change: float = 2.0):
        super().__init__(f"Gradient_{max_gradient_change}")
        self.group_by = group_by
        self.max_gradient_change = max_gradient_change
    
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove points with anomalous gradients."""
        outlier_indices = self.get_outlier_indices(df)
        return df.drop(outlier_indices)
    
    def get_outlier_indices(self, df: pd.DataFrame) -> List[int]:
        """Find points where gradient changes dramatically."""
        all_outliers = []
        
        for group_value, group_df in df.groupby(self.group_by):
            if len(group_df) < 3:
                continue
            
            group_df = group_df.sort_values('rho_c').reset_index(drop=True)
            
            # Calculate gradients
            drho = np.diff(group_df['rho_c'])
            dM = np.diff(group_df['M'])
            
            # Avoid division by zero
            gradients = np.where(drho != 0, dM / drho, 0)
            
            # Find gradient changes
            gradient_changes = np.abs(np.diff(gradients))
            
            # Identify outliers (points where gradient changes too much)
            for i in range(1, len(gradient_changes)):
                if gradient_changes[i-1] > self.max_gradient_change * np.median(gradient_changes):
                    all_outliers.append(group_df.index[i])
        
        return all_outliers