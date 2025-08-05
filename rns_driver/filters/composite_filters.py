# rns_driver/filters/composite_filters.py
from typing import List, Dict, Any
import pandas as pd
import logging

from .base_filter import BaseFilter
from .outlier_filters import StatisticalOutlierFilter, NeighborDistanceFilter
from .physical_filters import PhysicalBoundsFilter, MonotonicityFilter


class CompositeFilter(BaseFilter):
    """Combine multiple filters with configurable strategy."""
    
    def __init__(self, 
                 filters: List[BaseFilter],
                 strategy: str = 'sequential'):
        """
        Args:
            filters: List of filters to apply
            strategy: 'sequential' or 'union' or 'intersection'
        """
        super().__init__(f"Composite_{strategy}")
        self.filters = filters
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
    
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all filters according to strategy."""
        if self.strategy == 'sequential':
            result = df
            for filt in self.filters:
                initial_len = len(result)
                result = filt.filter(result)
                self.logger.info(
                    f"Filter {filt.name} removed {initial_len - len(result)} points"
                )
            return result
        
        elif self.strategy == 'union':
            # Remove points flagged by ANY filter
            all_outliers = set()
            for filt in self.filters:
                outliers = set(filt.get_outlier_indices(df))
                all_outliers.update(outliers)
            return df.drop(list(all_outliers))
        
        elif self.strategy == 'intersection':
            # Remove only points flagged by ALL filters
            if not self.filters:
                return df
            
            common_outliers = set(self.filters[0].get_outlier_indices(df))
            for filt in self.filters[1:]:
                outliers = set(filt.get_outlier_indices(df))
                common_outliers.intersection_update(outliers)
            return df.drop(list(common_outliers))
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def get_outlier_indices(self, df: pd.DataFrame) -> List[int]:
        """Get outlier indices according to strategy."""
        if self.strategy == 'sequential':
            # Not well-defined for sequential
            raise NotImplementedError("Sequential strategy doesn't support get_outlier_indices")
        
        elif self.strategy == 'union':
            all_outliers = set()
            for filt in self.filters:
                outliers = set(filt.get_outlier_indices(df))
                all_outliers.update(outliers)
            return list(all_outliers)
        
        elif self.strategy == 'intersection':
            if not self.filters:
                return []
            
            common_outliers = set(self.filters[0].get_outlier_indices(df))
            for filt in self.filters[1:]:
                outliers = set(filt.get_outlier_indices(df))
                common_outliers.intersection_update(outliers)
            return list(common_outliers)


def create_default_filter_pipeline() -> CompositeFilter:
    """Create a default filtering pipeline."""
    filters = [
        # First remove physically impossible models
        PhysicalBoundsFilter(),
        
        # Then statistical outliers
        StatisticalOutlierFilter(threshold=3.0, method='median'),
        
        # Neighbor-based filtering
        NeighborDistanceFilter(tolerance=0.1, normalize=True),
        
        # Ensure monotonicity
        MonotonicityFilter(strict=False)
    ]
    
    return CompositeFilter(filters, strategy='sequential')