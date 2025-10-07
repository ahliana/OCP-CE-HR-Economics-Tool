"""
Quick test to verify logging suppression works
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from data.loader import load_csv_files
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

print("=" * 60)
print("Test: Logging Suppression")
print("=" * 60)

# Load data
load_csv_files()

# Import suppression context manager
from ui.economics_panel import suppress_logging
from core.costs import compare_approaches

print("\n1. WITHOUT suppression (should show logs):")
print("-" * 60)
comparison1 = compare_approaches(1.0, 20, 10, approaches=[2])
print(f"Result: {comparison1.get('status')}")

print("\n2. WITH suppression (should NOT show logs):")
print("-" * 60)
with suppress_logging():
    comparison2 = compare_approaches(1.0, 20, 10, approaches=[3])
print(f"Result: {comparison2.get('status')}")

print("\n3. After suppression context (logs restored):")
print("-" * 60)
comparison3 = compare_approaches(1.0, 20, 10, approaches=[5])
print(f"Result: {comparison3.get('status')}")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
