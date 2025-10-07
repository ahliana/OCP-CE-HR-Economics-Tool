"""Quick test for economics panel data"""
import sys
import logging
sys.path.insert(0, 'python')

# Suppress all logging
logging.basicConfig(level=logging.CRITICAL)
for logger_name in ['core.costs', 'core.lookup', 'core.original_calculations']:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

from data.loader import load_csv_files
from core.costs import compare_approaches

# Load data silently
load_csv_files()

# Get comparison
comparison = compare_approaches(1.0, 20, 10, approaches=[2, 3, 5])

print("\n" + "="*60)
print("ECONOMICS PANEL DATA TEST")
print("="*60)

for approach_key in ['2C', '3C', '5C']:
    data = comparison['approaches'][approach_key]
    print(f"\n{approach_key}:")
    print(f"  Valves:        €{data.get('valves', 0):>10,.0f}")
    print(f"  Installation:  €{data.get('installation_cost', 0):>10,.0f}")
    print(f"  Engineering:   €{data.get('engineering_cost', 0):>10,.0f}")
    print(f"  Contingency:   €{data.get('contingency_cost', 0):>10,.0f}")
    print(f"  Capital Total: €{data.get('capital_total', 0):>10,.0f}")

print("\n" + "="*60)
print("VALIDATION:")
print("="*60)
all_good = True
for approach_key in ['2C', '3C', '5C']:
    data = comparison['approaches'][approach_key]
    if data.get('valves', 0) == 0:
        print(f"❌ {approach_key}: Valves = 0")
        all_good = False
    if data.get('installation_cost', 0) == 0:
        print(f"❌ {approach_key}: Installation = 0")
        all_good = False
    if data.get('engineering_cost', 0) == 0:
        print(f"❌ {approach_key}: Engineering = 0")
        all_good = False
    if data.get('contingency_cost', 0) == 0:
        print(f"❌ {approach_key}: Contingency = 0")
        all_good = False

if all_good:
    print("✓ ALL VALUES NON-ZERO - TEST PASSED!")
else:
    print("\n⚠ SOME VALUES ARE ZERO - TEST FAILED")
