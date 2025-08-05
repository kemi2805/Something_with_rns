"""
RNS Driver Package Setup Files
==============================
"""

# setup.py
SETUP_PY = """
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rns-driver",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Modern, high-performance neutron star modeling system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rns-driver",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "pre-commit",
        ],
        "viz": [
            "matplotlib>=3.5",
            "plotly>=5.0",
            "seaborn>=0.12",
        ],
    },
    entry_points={
        "console_scripts": [
            "rns-driver=rns_driver.main:main",
        ],
    },
)
"""

# requirements.txt
REQUIREMENTS_TXT = """
# Core dependencies
numpy>=1.21.0
pandas>=1.4.0
scipy>=1.7.0
h5py>=3.6.0
tables>=3.7.0
pyarrow>=7.0.0
pyyaml>=6.0

# Parallel processing
mpi4py>=3.1.0

# Progress bars and logging
tqdm>=4.65.0

# Optional but recommended
matplotlib>=3.5.0
seaborn>=0.12.0
"""

# pyproject.toml
PYPROJECT_TOML = """
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "rns-driver"
version = "2.0.0"
description = "Modern, high-performance neutron star modeling system"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Physics",
    "Programming Language :: Python :: 3",
]

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\.pyi?$'

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers"
testpaths = [
    "tests",
]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
"""

# Example configuration file
EXAMPLE_CONFIG_YAML = """
# RNS Driver Configuration File
# ============================

# RNS executable path
rns_executable: /home/miler/codes/Something_with_rns/source/rns.v1.1d/rns

# EOS directory
eos_directory: /home/miler/codes/Something_with_rns/EOS/106

# Output directory
output_directory: ./output

# Computation parameters
timeout: 10.0          # Timeout for RNS calculations (seconds)
tolerance: 1.0e-4      # Relative tolerance
accuracy: 1.0e-5       # Accuracy for iterations
relaxation_factor: 1.0 # Relaxation factor for difficult convergence

# Parallelization
max_workers: null      # null means use all available cores
chunk_size: 10         # Number of EOS files per chunk

# Filtering parameters
outlier_threshold: 3.0      # Standard deviations for outlier detection
neighbor_tolerance: 0.1     # Tolerance for neighbor distance filter

# Logging
log_level: INFO
log_file: null  # Set to a path to enable file logging
"""

# Example filter configuration
EXAMPLE_FILTER_YAML = """
# Filter Configuration
# ===================

# Use default pipeline (recommended)
use_default: false

# Physical bounds filter
physical_bounds: true
min_mass: 0.1      # Minimum mass [M☉]
max_mass: 3.5      # Maximum mass [M☉]
min_radius: 5.0    # Minimum radius [km]
max_radius: 30.0   # Maximum radius [km]

# Statistical outlier filter
statistical: true
outlier_threshold: 3.0
outlier_method: median  # 'mean' or 'median'

# Neighbor distance filter
neighbor: true
neighbor_tolerance: 0.1

# Monotonicity filter
monotonicity: true
strict_monotonicity: false
"""

# README.md
README_MD = """
# RNS Driver 2.0

A modern, high-performance Python framework for computing neutron star models using the RNS (Rapidly rotating Neutron Stars) code.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for solving, filtering, analysis, and I/O
- **Parallel Processing**: Built-in support for parallel computation with MPI and multiprocessing
- **Advanced Filtering**: Sophisticated filtering pipeline to remove unphysical and outlier models
- **Flexible Configuration**: YAML-based configuration with sensible defaults
- **Multiple Output Formats**: Support for Parquet, HDF5, and CSV formats
- **Comprehensive Analysis**: Built-in tools for sequence analysis, interpolation, and universal relations
- **Extensible Design**: Easy to add new filters, processing strategies, and analysis methods

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rns-driver.git
cd rns-driver

# Install in development mode
pip install -e .

# Or install from PyPI (when available)
pip install rns-driver
```

## Quick Start

```python
from rns_driver.config.settings import RNSConfig
from rns_driver.core.eos_catalog import EOSCatalog
from rns_driver.filters.composite_filters import create_default_filter_pipeline

# Configure
config = RNSConfig(
    eos_directory=Path("/path/to/eos/files"),
    output_directory=Path("./output"),
    max_workers=8
)

# Process EOS files
catalog = EOSCatalog(config)
results = catalog.process_eos_directory(
    config.eos_directory,
    filter_pipeline=create_default_filter_pipeline()
)

# Save results
from rns_driver.io.writers import DataWriter
writer = DataWriter()
writer.write_parquet(results, Path("neutron_stars.parquet"))
```

## Command Line Usage

```bash
# Process all EOS files in a directory
rns-driver --eos-dir /path/to/eos --output-dir ./results

# Use adaptive processing strategy
rns-driver --strategy adaptive --workers 16

# Apply custom filtering
rns-driver --filter-config my_filters.yaml

# Analyze existing results
rns-driver --analyze results.parquet --no-process
```

## Architecture

The package is organized into several modules:

- **`core/`**: Core data structures (NeutronStar, EOSCollection, EOSCatalog)
- **`solvers/`**: RNS solver interface and optimization routines
- **`filters/`**: Filtering system with various filter implementations
- **`parallel/`**: Parallel execution framework
- **`io/`**: Input/output handlers for various formats
- **`analysis/`**: Analysis tools for sequences and interpolation
- **`config/`**: Configuration management
- **`utils/`**: Utility functions

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:
- Original RNS code: Stergioulas & Friedman (1995)
- This Python framework: [Your citation]
"""

# Print all files
print("=== setup.py ===")
print(SETUP_PY)
print("\n=== requirements.txt ===")
print(REQUIREMENTS_TXT)
print("\n=== pyproject.toml ===")
print(PYPROJECT_TOML)
print("\n=== config.yaml (example) ===")
print(EXAMPLE_CONFIG_YAML)
print("\n=== filters.yaml (example) ===")
print(EXAMPLE_FILTER_YAML)
print("\n=== README.md ===")
print(README_MD)