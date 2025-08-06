#!/usr/bin/env python3
"""
RNS Driver - Main Application
============================

Modern, high-performance neutron star modeling system.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd # type: ignore
from datetime import datetime
import yaml  # pyright: ignore[reportMissingModuleSource]

# Import our modules
from rns_driver.config.settings import RNSConfig
from rns_driver.core.eos_catalog import EOSCatalog
from rns_driver.parallel.executor import ParallelExecutor
from rns_driver.parallel.strategies import (
    StandardProcessingStrategy, 
    AdaptiveProcessingStrategy
)
from rns_driver.filters.composite_filters import (
    create_default_filter_pipeline,
    CompositeFilter
)
from rns_driver.filters.outlier_filters import (
    StatisticalOutlierFilter,
    NeighborDistanceFilter
)
from rns_driver.filters.physical_filters import (
    PhysicalBoundsFilter,
    MonotonicityFilter
)
from rns_driver.io.writers import DataWriter
from rns_driver.io.readers import ParquetReader
from rns_driver.analysis.sequences import SequenceAnalyzer
from rns_driver.analysis.interpolation import SurfaceInterpolator


def setup_logging(config: RNSConfig) -> None:
    """Configure logging system."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if config.log_file:
        handlers.append(logging.FileHandler(config.log_file))
    
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format=log_format,
        handlers=handlers
    )


def process_single_eos(eos_path_and_config):
    """Process a single EOS file - pickleable function."""
    eos_path, config, strategy_type = eos_path_and_config
    
    if strategy_type == 'adaptive':
        from rns_driver.parallel.strategies import AdaptiveProcessingStrategy
        strategy = AdaptiveProcessingStrategy()
    else:
        from rns_driver.parallel.strategies import StandardProcessingStrategy
        strategy = StandardProcessingStrategy()
    
    return strategy.process(eos_path, config)


def process_eos_files(config: RNSConfig, 
                     eos_files: Optional[List[Path]] = None,
                     use_parallel: bool = True,
                     strategy: str = 'standard') -> pd.DataFrame:
    """
    Process EOS files to generate neutron star models.
    
    Args:
        config: Configuration object
        eos_files: Optional list of specific EOS files to process
        use_parallel: Whether to use parallel processing
        strategy: Processing strategy ('standard' or 'adaptive')
    
    Returns:
        DataFrame with all computed models
    """
    logger = logging.getLogger(__name__)
    
    # Get EOS files
    if eos_files is None:
        eos_files = list(config.eos_directory.glob("*.rns"))
    
    if not eos_files:
        logger.error(f"No EOS files found in {config.eos_directory}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(eos_files)} EOS files to process")
    
    # Process files
    if use_parallel and len(eos_files) > 1:
        logger.info(f"Using parallel processing with {config.max_workers} workers")
        
        executor = ParallelExecutor(config)
        
        # Prepare arguments for the pickleable function
        eos_args = [(eos_path, config, strategy) for eos_path in eos_files]
        
        results = executor.process_eos_files(
            eos_args,
            process_single_eos,  # Use the top-level function
            output_dir=config.output_directory / "intermediate"
        )
    else:
        logger.info("Using sequential processing")
        
        # Choose processing strategy
        if strategy == 'adaptive':
            processor_strategy = AdaptiveProcessingStrategy()
        else:
            processor_strategy = StandardProcessingStrategy()
        
        catalog = EOSCatalog(config)
        all_results = []
        
        for eos_file in eos_files:
            try:
                collection = catalog._process_single_eos(eos_file)
                if not collection.df.empty:
                    all_results.append(collection.df)
            except Exception as e:
                logger.error(f"Failed to process {eos_file}: {e}")
        
        results = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    
    logger.info(f"Generated {len(results)} neutron star models")
    return results


def apply_filters(df: pd.DataFrame, 
                 filter_config: dict) -> pd.DataFrame:
    """
    Apply filtering pipeline to results.
    
    Args:
        df: Input DataFrame
        filter_config: Filter configuration dict
    
    Returns:
        Filtered DataFrame
    """
    logger = logging.getLogger(__name__)
    
    if filter_config.get('use_default', True):
        pipeline = create_default_filter_pipeline()
    else:
        # Build custom pipeline
        filters = []
        
        if filter_config.get('physical_bounds', True):
            filters.append(PhysicalBoundsFilter(
                min_mass=filter_config.get('min_mass', 0.1),
                max_mass=filter_config.get('max_mass', 3.5),
                min_radius=filter_config.get('min_radius', 5.0),
                max_radius=filter_config.get('max_radius', 30.0)
            ))
        
        if filter_config.get('statistical', True):
            filters.append(StatisticalOutlierFilter(
                threshold=filter_config.get('outlier_threshold', 3.0),
                method=filter_config.get('outlier_method', 'median')
            ))
        
        if filter_config.get('neighbor', True):
            filters.append(NeighborDistanceFilter(
                tolerance=filter_config.get('neighbor_tolerance', 0.1),
                normalize=True
            ))
        
        if filter_config.get('monotonicity', True):
            filters.append(MonotonicityFilter(
                strict=filter_config.get('strict_monotonicity', False)
            ))
        
        pipeline = CompositeFilter(filters, strategy='sequential')
    
    initial_count = len(df)
    filtered_df = pipeline.filter(df)
    final_count = len(filtered_df)
    
    logger.info(f"Filtering: {initial_count} -> {final_count} models "
                f"({initial_count - final_count} removed)")
    
    return filtered_df


def analyze_results(df: pd.DataFrame, 
                   output_dir: Path) -> dict:
    """
    Perform analysis on computed models.
    
    Args:
        df: DataFrame with neutron star models
        output_dir: Directory for analysis outputs
    
    Returns:
        Dictionary with analysis results
    """
    logger = logging.getLogger(__name__)
    analyzer = SequenceAnalyzer()
    
    results = {}
    
    # Find turning points
    turning_points = analyzer.find_turning_points(df)
    results['turning_points'] = pd.DataFrame(turning_points)
    logger.info(f"Found {len(turning_points)} turning points")
    
    # Find maximum masses
    max_masses = analyzer.find_maximum_masses(df)
    results['max_masses'] = max_masses
    
    # Compute universal relations
    universal_relations = analyzer.compute_universal_relations(df)
    results['universal_relations'] = universal_relations
    
    if 'I_Q_relation' in universal_relations:
        logger.info(f"I-Q relation: {universal_relations['I_Q_relation']['formula']}")
    
    # Save analysis results
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    writer = DataWriter()
    writer.write_csv(results['turning_points'], analysis_dir / "turning_points.csv")
    writer.write_csv(results['max_masses'], analysis_dir / "maximum_masses.csv")
    
    with open(analysis_dir / "universal_relations.yaml", 'w') as f:
        yaml.dump(universal_relations, f)
    
    return results


def save_results(df: pd.DataFrame, 
                config: RNSConfig,
                timestamp: str) -> None:
    """Save results in multiple formats."""
    logger = logging.getLogger(__name__)
    writer = DataWriter()
    
    output_dir = config.output_directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Primary output - Parquet (efficient, preserves types)
    main_output = output_dir / f"neutron_stars_{timestamp}.parquet"
    writer.write_parquet(df, main_output)
    
    # Also save as HDF5 for compatibility
    hdf5_output = output_dir / f"neutron_stars_{timestamp}.h5"
    writer.write_hdf5(df, hdf5_output)
    
    # Summary file
    summary_output = output_dir / f"summary_{timestamp}.txt"
    writer.write_summary(df, summary_output)
    
    # Per-EOS files if requested
    if config.output_directory:
        eos_dir = output_dir / "by_eos"
        eos_dir.mkdir(exist_ok=True)
        
        for eos in df['eos'].unique():
            eos_df = df[df['eos'] == eos]
            eos_output = eos_dir / f"{eos}_{timestamp}.parquet"
            writer.write_parquet(eos_df, eos_output)
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="RNS Driver - Modern neutron star modeling system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all EOS files in directory
  python main.py --eos-dir /path/to/eos --output-dir ./results
  
  # Process specific EOS files with custom config
  python main.py --eos-files eos1.rns eos2.rns --config config.yaml
  
  # Use adaptive strategy with more workers
  python main.py --strategy adaptive --workers 16
  
  # Apply custom filtering
  python main.py --filter-config filters.yaml
  
  # Analyze existing results
  python main.py --analyze results.parquet --no-process
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--eos-dir', type=Path, 
                       help='Directory containing EOS files')
    parser.add_argument('--eos-files', nargs='+', type=Path,
                       help='Specific EOS files to process')
    parser.add_argument('--output-dir', type=Path, default=Path('./output'),
                       help='Output directory (default: ./output)')
    
    # Configuration
    parser.add_argument('--config', type=Path,
                       help='Configuration file (YAML)')
    parser.add_argument('--rns-executable', type=Path,
                       help='Path to RNS executable')
    
    # Processing options
    parser.add_argument('--workers', type=int,
                       help='Number of parallel workers')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel processing')
    parser.add_argument('--strategy', choices=['standard', 'adaptive'],
                       default='standard',
                       help='Processing strategy')
    
    # Filtering options
    parser.add_argument('--filter-config', type=Path,
                       help='Filter configuration file')
    parser.add_argument('--no-filter', action='store_true',
                       help='Skip filtering')
    
    # Analysis options
    parser.add_argument('--analyze', type=Path,
                       help='Analyze existing results file')
    parser.add_argument('--no-process', action='store_true',
                       help='Skip processing, only analyze')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Skip analysis')
    
    # Other options
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=Path,
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        config = RNSConfig.from_yaml(args.config)
    else:
        config = RNSConfig()
    
    # Override config with command line arguments
    if args.eos_dir:
        config.eos_directory = args.eos_dir
    if args.output_dir:
        config.output_directory = args.output_dir
    if args.rns_executable:
        config.rns_executable = args.rns_executable
    if args.workers:
        config.max_workers = args.workers
    if args.log_level:
        config.log_level = args.log_level
    if args.log_file:
        config.log_file = args.log_file
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    logger.info("RNS Driver starting...")
    logger.info(f"Configuration: {config}")
    
    # Timestamp for outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Process or load results
    if args.analyze and args.no_process:
        # Just analyze existing results
        logger.info(f"Loading results from {args.analyze}")
        reader = ParquetReader()
        df = pd.read_parquet(args.analyze)
    else:
        # Process EOS files
        eos_files = args.eos_files if args.eos_files else None
        
        df = process_eos_files(
            config,
            eos_files=eos_files,
            use_parallel=not args.no_parallel,
            strategy=args.strategy
        )
        
        if df.empty:
            logger.error("No models computed")
            return 1
        
        # Apply filtering
        if not args.no_filter:
            if args.filter_config:
                with open(args.filter_config, 'r') as f:
                    filter_config = yaml.safe_load(f)
            else:
                filter_config = {'use_default': True}
            
            df = apply_filters(df, filter_config)
        
        # Save results
        save_results(df, config, timestamp)
    
    # Perform analysis
    if not args.no_analysis and not df.empty:
        logger.info("Performing analysis...")
        analysis_results = analyze_results(df, config.output_directory)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total models: {len(df):,}")
        print(f"Number of EOS: {df['eos'].nunique()}")
        print(f"Mass range: {df['M'].min():.3f} - {df['M'].max():.3f} M☉")
        print(f"Radius range: {df['R'].min():.1f} - {df['R'].max():.1f} km")
        print(f"Max frequency: {df['Omega'].max():.0f} × 10⁴ s⁻¹")
        
        if 'max_masses' in analysis_results:
            max_masses = analysis_results['max_masses']
            print(f"\nMaximum masses by EOS:")
            for _, row in max_masses.groupby('eos')['M'].max().items():
                print(f"  {_}: {row:.3f} M☉")
    
    logger.info("RNS Driver completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())