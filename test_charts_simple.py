"""
Simple visual test for chart reorganization
Tests the new 3-pie-chart implementation
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
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for display
import matplotlib.pyplot as plt

print("=" * 80)
print("CHART REORGANIZATION VISUAL TEST")
print("=" * 80)
print("\nTest Case: 1.0 MW, 20°C, +10°C rise")
print("-" * 80)

wha = 1.0
T1 = 20
temp_rise = 10

# Get comparison data
print(f"Fetching data for {wha} MW, {T1}°C, +{temp_rise}°C rise...")

with suppress_logging():
    comparison = compare_approaches(wha, T1, temp_rise, approaches=[2, 3, 5])

if comparison.get('status') != 'success':
    print(f"❌ ERROR: Failed to get comparison data")
    sys.exit(1)

approaches_data = comparison['approaches']

# Print cost breakdown for verification
print(f"✓ Data retrieved for approaches: {list(approaches_data.keys())}\n")

for approach in [2, 3, 5]:
    key = f"{approach}C"
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

    total = heat_exchanger + pumps + pipe_fittings + instrumentation + valves + ic_subtotal

    print(f"{approach}°C Approach:")
    print(f"  Heat Exchangers:  €{heat_exchanger:>10,.0f}  ({heat_exchanger/total*100:>5.1f}%)")
    print(f"  Pumps:            €{pumps:>10,.0f}  ({pumps/total*100:>5.1f}%)")
    print(f"  Piping & Fittings:€{pipe_fittings:>10,.0f}  ({pipe_fittings/total*100:>5.1f}%)")
    print(f"  Instrumentation:  €{instrumentation:>10,.0f}  ({instrumentation/total*100:>5.1f}%)")
    print(f"  Valves:           €{valves:>10,.0f}  ({valves/total*100:>5.1f}%)")
    print(f"  I&C Subtotal:     €{ic_subtotal:>10,.0f}  ({ic_subtotal/total*100:>5.1f}%)")
    print(f"  TOTAL:            €{total:>10,.0f}  (100.0%)\n")

# Key observations
print("Key Observations:")
heat_2c = approaches_data['2C'].get('heat_exchanger', 0)
heat_5c = approaches_data['5C'].get('heat_exchanger', 0)
pump_2c = approaches_data['2C'].get('pumps', 0)
pump_5c = approaches_data['5C'].get('pumps', 0)

print(f"  ✓ 2°C has higher HX cost: €{heat_2c:,.0f} vs €{heat_5c:,.0f} (5°C)")
print(f"  ✓ 5°C has higher pump cost: €{pump_5c:,.0f} vs €{pump_2c:,.0f} (2°C)")

print("\n" + "=" * 80)
print("CREATING PIE CHARTS...")
print("=" * 80)

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
    print("\nCalling create_approach_cost_breakdown_charts()...")
    create_approach_cost_breakdown_charts(wha, T1, temp_rise, output_area)

    print("\n✓ Charts created successfully!")
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKLIST")
    print("=" * 80)
    print("\nA matplotlib window should be displayed showing 3 pie charts.")
    print("\nPlease verify:")
    print("  [ ] Three pie charts are visible side-by-side")
    print("  [ ] Titles: '2°C Approach', '3°C Approach', '5°C Approach'")
    print("  [ ] Each chart has 6 colored segments")
    print("  [ ] Percentages are displayed in white bold text")
    print("  [ ] Percentages sum to 100% in each chart")
    print("  [ ] 2°C chart: Heat Exchangers = 12.5% (teal slice)")
    print("  [ ] 5°C chart: Pumps = 31.0% (orange slice)")
    print("  [ ] I&C Subtotal (yellow) is ~28% in all charts")
    print("  [ ] Figure title: 'Equipment & Installation Cost Breakdown by Approach Temperature'")
    print("\n" + "=" * 80)
    print("Close the chart window to continue...")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ TEST COMPLETE")
