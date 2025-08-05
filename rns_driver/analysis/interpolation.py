# rns_driver/analysis/interpolation.py
import numpy as np
from scipy.interpolate import RBFInterpolator, griddata
from typing import Tuple, Optional
import pandas as pd


class SurfaceInterpolator:
    """Interpolate neutron star properties on 2D surfaces."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.interpolators = {}
    
    def create_mass_surface(self, 
                          eos: str,
                          method: str = 'rbf') -> None:
        """
        Create 2D interpolation surface for mass.
        
        Args:
            eos: EOS name
            method: 'rbf' or 'linear'
        """
        eos_df = self.df[self.df['eos'] == eos]
        
        if len(eos_df) < 10:
            raise ValueError(f"Insufficient data for EOS {eos}")
        
        # Extract data points
        points = eos_df[['rho_c', 'r_ratio']].values
        values = eos_df['M'].values
        
        if method == 'rbf':
            # Radial basis function interpolation
            self.interpolators[eos] = RBFInterpolator(
                points, values, 
                smoothing=0.01,
                kernel='thin_plate_spline'
            )
        else:
            # Store for griddata interpolation
            self.interpolators[eos] = {
                'points': points,
                'values': values,
                'method': 'linear'
            }
    
    def evaluate_surface(self,
                        eos: str,
                        rho_c: np.ndarray,
                        r_ratio: np.ndarray) -> np.ndarray:
        """
        Evaluate interpolated surface at given points.
        
        Args:
            eos: EOS name
            rho_c: Central density values
            r_ratio: Axes ratio values
        
        Returns:
            Interpolated mass values
        """
        if eos not in self.interpolators:
            raise ValueError(f"No interpolator for EOS {eos}")
        
        # Create evaluation points
        eval_points = np.column_stack([rho_c.ravel(), r_ratio.ravel()])
        
        interp = self.interpolators[eos]
        
        if isinstance(interp, RBFInterpolator):
            values = interp(eval_points)
        else:
            # Use griddata
            values = griddata(
                interp['points'],
                interp['values'],
                eval_points,
                method=interp['method']
            )
        
        return values.reshape(rho_c.shape)
    
    def find_contours(self,
                     eos: str,
                     levels: list,
                     rho_bounds: Tuple[float, float],
                     r_bounds: Tuple[float, float] = (0.5, 1.0),
                     n_points: int = 100) -> Dict[float, np.ndarray]:
        """
        Find contour lines on the surface.
        
        Args:
            eos: EOS name
            levels: Mass levels for contours
            rho_bounds: Central density bounds
            r_bounds: Axes ratio bounds
            n_points: Grid resolution
        
        Returns:
            Dictionary mapping levels to contour coordinates
        """
        # Create grid
        rho_grid = np.linspace(rho_bounds[0], rho_bounds[1], n_points)
        r_grid = np.linspace(r_bounds[0], r_bounds[1], n_points)
        rho_mesh, r_mesh = np.meshgrid(rho_grid, r_grid)
        
        # Evaluate surface
        mass_surface = self.evaluate_surface(eos, rho_mesh, r_mesh)
        
        # Find contours using matplotlib
        import matplotlib.pyplot as plt
        
        contours = {}
        fig, ax = plt.subplots()
        
        for level in levels:
            cs = ax.contour(rho_mesh, r_mesh, mass_surface, levels=[level])
            
            # Extract contour coordinates
            for collection in cs.collections:
                for path in collection.get_paths():
                    vertices = path.vertices
                    if len(vertices) > 0:
                        if level not in contours:
                            contours[level] = []
                        contours[level].append(vertices)
        
        plt.close(fig)
        
        return contours


# Example usage functions
def example_basic_usage():
    """Example of basic RNS driver usage."""
    from pathlib import Path
    from rns_driver.config.settings import RNSConfig
    from rns_driver.core.eos_catalog import EOSCatalog
    from rns_driver.filters.composite_filters import create_default_filter_pipeline
    
    # Configuration
    config = RNSConfig(
        eos_directory=Path("/path/to/eos/files"),
        output_directory=Path("./output"),
        max_workers=8,
        tolerance=1e-4
    )
    
    # Create catalog and process EOS files
    catalog = EOSCatalog(config)
    
    # Create filter pipeline
    filter_pipeline = create_default_filter_pipeline()
    
    # Process all EOS files
    results = catalog.process_eos_directory(
        config.eos_directory,
        filter_pipeline
    )
    
    # Save results
    from rns_driver.io.writers import DataWriter
    writer = DataWriter()
    writer.write_parquet(results, config.output_directory / "neutron_stars.parquet")
    writer.write_summary(results, config.output_directory / "summary.txt")
    
    # Get summary statistics
    summary = catalog.get_summary_statistics()
    print(summary)


def example_parallel_processing():
    """Example of parallel processing with custom strategy."""
    from pathlib import Path
    from rns_driver.config.settings import RNSConfig
    from rns_driver.parallel.executor import ParallelExecutor
    from rns_driver.parallel.strategies import AdaptiveProcessingStrategy
    
    config = RNSConfig(max_workers=16)
    
    # Create parallel executor
    executor = ParallelExecutor(config)
    
    # Get EOS files
    eos_files = list(Path("/path/to/eos").glob("*.rns"))
    
    # Create processing strategy
    strategy = AdaptiveProcessingStrategy()
    
    # Process in parallel
    results = executor.process_eos_files(
        eos_files,
        lambda eos: strategy.process(eos, config),
        output_dir=Path("./parallel_output")
    )
    
    print(f"Processed {len(results)} models")


def example_analysis():
    """Example of sequence analysis."""
    import pandas as pd
    from rns_driver.analysis.sequences import SequenceAnalyzer
    from rns_driver.analysis.interpolation import SurfaceInterpolator
    
    # Load results
    df = pd.read_parquet("neutron_stars.parquet")
    
    # Analyze sequences
    analyzer = SequenceAnalyzer()
    
    # Find turning points
    turning_points = analyzer.find_turning_points(df)
    print(f"Found {len(turning_points)} turning points")
    
    # Find maximum masses
    max_masses = analyzer.find_maximum_masses(df)
    
    # Compute universal relations
    relations = analyzer.compute_universal_relations(df)
    if 'I_Q_relation' in relations:
        print(f"I-Q relation: {relations['I_Q_relation']['formula']}")
    
    # Create interpolation surface
    interpolator = SurfaceInterpolator(df)
    
    for eos in df['eos'].unique()[:3]:  # First 3 EOS
        interpolator.create_mass_surface(eos)
        
        # Find 2.0 M☉ contour
        contours = interpolator.find_contours(
            eos, 
            levels=[2.0],
            rho_bounds=(1e14, 1e16)
        )


if __name__ == "__main__":
    print(__doc__)