#!/usr/bin/env python3
"""
Setup script for RNS Driver package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if it exists
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "RNS Driver - Modern neutron star modeling system"

# Define requirements
requirements = [
    "numpy>=1.21.0",
    "pandas>=1.4.0",
    "scipy>=1.7.0",
    "h5py>=3.6.0",
    "pyarrow>=7.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "matplotlib>=3.5.0",
]

setup(
    name="rns-driver",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Modern, high-performance neutron star modeling system",
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        ],
        "viz": [
            "matplotlib>=3.5",
            "seaborn>=0.12",
        ],
        "mpi": [
            "mpi4py>=3.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rns-driver=main:main",
        ],
    },
)