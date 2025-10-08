"""
Test script for chart reorganization verification
Tests the new 3-pie-chart implementation in Economics panel
"""

import sys
import os
import io

# Set UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root and python directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'python'))

# Load CSV data
from python.data.loader import load_csv_files
print("Loading CSV data...")
load_csv_files()
print("✓ CSV data loaded\n")

# Import required modules
from python.core.costs import compare_approaches
from python.ui.economics_panel import create_approach_cost_breakdown_charts, suppress_logging
import matplotlib.pyplot as plt

# Test cases
test_cases = [
    {"wha": 1.0, "T1": 20, "temp_rise": 10, "name": "Test 1: 1.0 MW, 20°C, +10°C"},
    {"wha": 2.5, "T1": 25, "temp_rise": 12, "name": "Test 2: 2.5 MW, 25°C, +12°C"},
    {"wha": 5.0, "T1": 15, "temp_rise": 8, "name": "Test 3: 5.0 MW, 15°C, +8°C"},
]

print("=" * 80)
print("CHART REORGANIZATION TEST")
print("=" * 80)

for i, test in enumerate(test_cases, 1):
    print(f"\n{test['name']}")
    print("-" * 80)

    wha = test['wha']
    T1 = test['T1']
    temp_rise = test['temp_rise']

    # Get comparison data
    print(f"Fetching data for {wha} MW, {T1}°C, +{temp_rise}°C rise...")

    with suppress_logging():
        comparison = compare_approaches(wha, T1, temp_rise, approaches=[2, 3, 5])

    if comparison.get('status') != 'success':
        print(f"❌ ERROR: Failed to get comparison data")
        continue

    approaches_data = comparison['approaches']

    # Verify all approaches are present
    print(f"✓ Data retrieved for approaches: {list(approaches_data.keys())}")

    # Analyze each approach
    for approach in [2, 3, 5]:
        key = f"{approach}C"
        if key not in approaches_data:
            print(f"❌ ERROR: Missing data for {approach}°C approach")
            continue

        data = approaches_data[key]

        # Extract cost components
        heat_exchanger = data.get('heat_exchanger', 0)
        pumps = data.get('pumps', 0)
        pipe_fittings = data.get('pipe_fittings', 0)
        instrumentation = data.get('instrumentation', 0)
        valves = data.get('valves', 0)

        # Calculate I&C Subtotal
        ic_subtotal = sum([
            data.get('installation_cost', 0),
            data.get('engineering_cost', 0),
            data.get('contingency_cost', 0)
        ])

        # Calculate total and percentages
        total = heat_exchanger + pumps + pipe_fittings + instrumentation + valves + ic_subtotal

        print(f"\n  {approach}°C Approach:")
        print(f"    Heat Exchangers:  €{heat_exchanger:>10,.0f}  ({heat_exchanger/total*100:>5.1f}%)")
        print(f"    Pumps:            €{pumps:>10,.0f}  ({pumps/total*100:>5.1f}%)")
        print(f"    Piping & Fittings:€{pipe_fittings:>10,.0f}  ({pipe_fittings/total*100:>5.1f}%)")
        print(f"    Instrumentation:  €{instrumentation:>10,.0f}  ({instrumentation/total*100:>5.1f}%)")
        print(f"    Valves:           €{valves:>10,.0f}  ({valves/total*100:>5.1f}%)")
        print(f"    I&C Subtotal:     €{ic_subtotal:>10,.0f}  ({ic_subtotal/total*100:>5.1f}%)")
        print(f"    ─────────────────────────────────────────")
        print(f"    TOTAL:            €{total:>10,.0f}  (100.0%)")

        # Verify total matches capital_total
        capital_total = data.get('capital_total', 0)
        if abs(total - capital_total) <= 500:  # Allow €500 rounding tolerance
            print(f"    ✓ Total matches capital_total (€{capital_total:,.0f})")
        else:
            print(f"    ⚠ Total mismatch: €{total:,.0f} vs capital_total €{capital_total:,.0f}")

    # Key observations
    print(f"\n  Key Observations:")
    heat_2c = approaches_data['2C'].get('heat_exchanger', 0)
    heat_5c = approaches_data['5C'].get('heat_exchanger', 0)
    pump_2c = approaches_data['2C'].get('pumps', 0)
    pump_5c = approaches_data['5C'].get('pumps', 0)

    if heat_2c > heat_5c:
        print(f"    ✓ 2°C has higher HX cost (€{heat_2c:,.0f} vs €{heat_5c:,.0f})")
    else:
        print(f"    ⚠ Expected 2°C to have higher HX cost")

    if pump_5c >= pump_2c:
        print(f"    ✓ 5°C has higher/equal pump cost (€{pump_5c:,.0f} vs €{pump_2c:,.0f})")
    else:
        print(f"    ⚠ Expected 5°C to have higher pump cost")

print("\n" + "=" * 80)
print("VISUAL TEST")
print("=" * 80)
print("\nCreating visual test with Test Case 1 (1.0 MW, 20°C, +10°C)...")

# Create a mock output area class for testing
class MockOutputArea:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def clear_output(self, wait=False):
        pass

output_area = MockOutputArea()

try:
    # Test the chart function
    print("Calling create_approach_cost_breakdown_charts()...")
    create_approach_cost_breakdown_charts(1.0, 20, 10, output_area)
    print("✓ Charts created successfully!")
    print("\nA matplotlib window should appear showing 3 pie charts side-by-side.")
    print("Verify:")
    print("  1. Three pie charts are visible (2°C, 3°C, 5°C)")
    print("  2. Each has 6 colored segments")
    print("  3. Percentages are shown in white bold text")
    print("  4. Percentages sum to 100% in each chart")
    print("  5. 2°C chart has higher % for Heat Exchangers")
    print("  6. 5°C chart has higher % for Pumps")

except Exception as e:
    print(f"❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
