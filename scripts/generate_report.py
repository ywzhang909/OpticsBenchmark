#!/usr/bin/env python3
"""
OptiS Benchmark - Report Generator (thin wrapper)

Import from src.utils.generate_report.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.generate_report import main

if __name__ == "__main__":
    main()
