#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run TrackCoach Pipeline"""

import sys
import os
from pathlib import Path

# Add project root directory to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Change to project root directory
os.chdir(project_root)

# Import and run pipeline
from trackcoach.pipeline import main

if __name__ == '__main__':
    sys.argv = [
        'pipeline',
        '--csv', 'Data/ss_1_v3_processed.csv',
        '--outdir', 'output',
        '--analyze-issues',
        '--generate-advice',
        '--advice-config', 'llamacloud_config.json',
        '--plot', 'true'
    ]
    sys.exit(main())
