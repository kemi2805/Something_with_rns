#!/usr/bin/env python3
"""
Example script to run RNS Driver
"""

from pathlib import Path
import pandas as pd
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to import the modules - adjust paths as needed
try:
    from rns_driver.config.settings import RNSConfig
    from rns_driver.core.eos_catalog import EOSCatalog
    from rns_driver.filters.composite_filters import create_default_filter_pipeline
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying to import from backup directory...")
    import sys
    sys.path.append('./rns_driver_backup')
    from model.NeutronStar import NeutronStarEOSCatalog

def run_modern_version():
    """Run using the modern modular version"""
    
    # Configuration
    config = RNSConfig(
        rns_executable=Path("/mnt/rafast/miler/codes/Something_with_rns/source/rns.v1.1d/rns"),
        eos_directory=Path("/mnt/rafast/miler/codes/Something_with_rns/EOS/106"),
        output_directory=Path("./output"),
        max_workers=4,  # Adjust based on your system
        timeout=10.0
    )
    
    print(f"RNS executable: {config.rns_executable}")
    print(f"EOS directory: {config.eos_directory}")
    print(f"Output directory: {config.output_directory}")
    
    # Check if paths exist
    if not config.rns_executable.exists():
        print(f"ERROR: RNS executable not found at {config.rns_executable}")
        return
    
    if not config.eos_directory.exists():
        print(f"ERROR: EOS directory not found at {config.eos_directory}")
        return
    
    # Create catalog and process EOS files
    catalog = EOSCatalog(config)
    
    # Create filter pipeline
    filter_pipeline = create_default_filter_pipeline()
    
    # Process a single EOS file first (for testing)
    eos_files = list(config.eos_directory.glob("*.rns"))
    if not eos_files:
        print("No .rns files found!")
        return
    
    print(f"Found {len(eos_files)} EOS files")
    print(f"Processing first file: {eos_files[0].name}")
    
    # Process just the first EOS file for testing
    try:
        collection = catalog._process_single_eos(eos_files[0])
        print(f"Generated {len(collection.df)} models for {eos_files[0].name}")
        
        if not collection.df.empty:
            print("\nFirst few models:")
            print(collection.df[['eos', 'rho_c', 'M', 'R', 'Omega']].head())
            
            # Save results
            config.output_directory.mkdir(parents=True, exist_ok=True)
            output_file = config.output_directory / f"{eos_files[0].stem}_results.csv"
            collection.df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error processing EOS: {e}")
        import traceback
        traceback.print_exc()

def run_backup_version():
    """Run using the backup version"""
    
    print("Running backup version...")
    
    # Test with a single EOS file
    eos_path = "/mnt/rafast/miler/codes/Something_with_rns/EOS/106"
    eos_files = [f for f in Path(eos_path).glob("*.rns")]
    
    if not eos_files:
        print(f"No EOS files found in {eos_path}")
        return
    
    print(f"Found {len(eos_files)} EOS files")
    
    # Test with first EOS
    test_eos = str(eos_files[0])
    print(f"Testing with: {test_eos}")
    
    try:
        # Create catalog
        catalog = NeutronStarEOSCatalog()
        
        # Process single EOS
        collection = catalog._process_single_eos(test_eos)
        
        print(f"Generated {len(collection.df)} models")
        if not collection.df.empty:
            print("\nFirst few models:")
            print(collection.df[['eos', 'rho_c', 'M', 'R', 'Omega']].head())
            
            # Save results
            output_file = f"test_results_{Path(test_eos).stem}.csv"
            collection.df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def simple_single_star_test():
    """Test creating a single neutron star"""
    
    try:
        import sys
        sys.path.append('./rns_driver_backup')
        from model.NeutronStar import NeutronStar
        
        eos_path = "/mnt/rafast/miler/codes/Something_with_rns/EOS/106"
        eos_files = list(Path(eos_path).glob("*.rns"))
        
        if not eos_files:
            print("No EOS files found")
            return
        
        test_eos = str(eos_files[0])
        print(f"Creating single star with EOS: {test_eos}")
        
        # Create a single neutron star
        star = NeutronStar(
            eos=test_eos,
            rho_c=1e15,  # Central density
            r_ratio=1.0  # Non-rotating (static)
        )
        
        if star.is_valid():
            print(f"Success! Created star with:")
            print(f"  Mass: {star.M:.3f} M☉")
            print(f"  Radius: {star.R:.1f} km")
            print(f"  Central density: {star.rho_c:.2e} g/cm³")
        else:
            print("Star is not valid")
            
    except Exception as e:
        print(f"Error creating single star: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("RNS Driver Test")
    print("=" * 40)
    
    # Start with the simplest test
    print("\n1. Testing single star creation...")
    simple_single_star_test()
    
    #print("\n" + "=" * 40)
    #print("2. Testing backup version...")
    #run_backup_version()
    
    # Uncomment this when the modern version is working
    # print("\n" + "=" * 40)
    # print("3. Testing modern version...")
    # run_modern_version()
