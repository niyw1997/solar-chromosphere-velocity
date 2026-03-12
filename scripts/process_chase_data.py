#!/usr/bin/env python3
"""
Script for processing CHASE chromosphere velocity data.
"""

import sys
import os

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solar_chromosphere_velocity.cli import main

if __name__ == "__main__":
    main()