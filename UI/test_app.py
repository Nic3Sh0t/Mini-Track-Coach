#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test if app.py can be imported and run normally
"""

import sys
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("=" * 50)
print("Testing TrackCoach UI Application")
print("=" * 50)
print()

# Test 1: Check basic imports
print("1. Testing basic module imports...")
try:
    import streamlit as st
    print("   ✓ streamlit import successful")
except ImportError as e:
    print(f"   ✗ streamlit import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("   ✓ pandas import successful")
except ImportError as e:
    print(f"   ✗ pandas import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    print("   ✓ numpy import successful")
except ImportError as e:
    print(f"   ✗ numpy import failed: {e}")
    sys.exit(1)

# Test 2: Check path settings
print()
print("2. Testing path settings...")
try:
    UI_DIR = Path(__file__).parent
    PROJECT_ROOT = UI_DIR.parent
    NEW_TEST_DIR = PROJECT_ROOT
    
    print(f"   UI Directory: {UI_DIR}")
    print(f"   Project Root Directory: {PROJECT_ROOT}")
    print(f"   Parent Directory: {NEW_TEST_DIR}")
    
    if not PROJECT_ROOT.exists():
        print(f"   ✗ Project root directory does not exist: {PROJECT_ROOT}")
        sys.exit(1)
    else:
        print("   ✓ Project root directory exists")
    
    if not NEW_TEST_DIR.exists():
        print(f"   ⚠ Warning: Parent directory does not exist: {NEW_TEST_DIR}")
    else:
        print("   ✓ Parent directory exists")
        
except Exception as e:
    print(f"   ✗ Path setting failed: {e}")
    sys.exit(1)

# Test 3: Check trackcoach module
print()
print("3. Testing trackcoach module...")
try:
    sys.path.insert(0, str(NEW_TEST_DIR))
    from trackcoach import pipeline
    print("   ✓ trackcoach.pipeline import successful")
except ImportError as e:
    print(f"   ⚠ Warning: trackcoach module import failed: {e}")
    print("   This may affect pipeline functionality, but UI interface can still launch")

# Test 4: Try importing app.py
print()
print("4. Testing app.py import...")
try:
    import app
    print("   ✓ app.py import successful")
except Exception as e:
    print(f"   ✗ app.py import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 50)
print("All tests passed! Streamlit application can be launched")
print("=" * 50)
