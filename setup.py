#!/usr/bin/env python
"""Setup script for solar-chromosphere-velocity package."""

import os
import re
from setuptools import setup, find_packages

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

def get_version():
    """Read version from __init__.py."""
    init_path = os.path.join("src", "solar_chromosphere_velocity", "__init__.py")
    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Use regex to find version
    version_match = re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', content, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name="solar-chromosphere-velocity",
    version=get_version(),  # Now returns 1.0.1
    author="Solar Physics Team",
    author_email="solar@example.com",
    description="Solar chromosphere velocity data processing package for CHASE mission",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/solar-chromosphere-velocity",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "nbsphinx>=0.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "process-chase=solar_chromosphere_velocity.cli:main",
        ],
    },
    include_package_data=True,
)