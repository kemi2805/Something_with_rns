# rns_driver/filters/base_filter.py
from abc import ABC, abstractmethod
from typing import List, Optional
import pandas as pd
import numpy as np


class BaseFilter(ABC):
    """Abstract base class for all filters."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the filter to a DataFrame."""
        pass
    
    @abstractmethod
    def get_outlier_indices(self, df: pd.DataFrame) -> List[int]:
        """Get indices of outliers without removing them."""
        pass


