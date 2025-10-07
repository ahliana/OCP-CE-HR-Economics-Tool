"""
Clean test script for costs.py module (no debug output)
"""

import sys
import logging
sys.path.insert(0, 'python')

# Disable INFO logging
logging.basicConfig(level=logging.WARNING)

from data.loader import load_csv_files
from core.costs import compare_approaches, calculate_order_of_magnitude_estimate
import warnings
warnings.filterwarnings('ignore')

# Load CSV data
load_csv_files()

print("=" * 70)
print("COST MODULE VALIDATION - Heat Reuse Economics Tool")
print("=" * 70)

# Test: Compare approaches
result = compare_approaches(1.0, 20, 10)

if result.get('status') == 'success':
    print("\n[OK] Module functioning correctly!\n")

    print(f"{'Approach':<12} {'HX Cost':>12} {'Pumps':>12} {'Pipe&Fit':>12} {'Instr':>12} {'TOTAL':>14} {'Energy':>12}")
    print("-" * 90)

    for approach_key, data in sorted(result['approaches'].items()):
        approach = data['approach']
        print(f"{approach}°C          €{data['heat_exchanger']:>10,.0f}  €{data['pumps']:>10,.0f}  €{data['pipe_fittings']:>10,.0f}  €{data['instrumentation']:>10,.0f}  €{data['capital_total']:>12,.0f}  {data['operating_energy_kwh_year']:>10,.0f} kWh")

    print("\n" + "=" * 70)
    print("TARGET VALUES COMPARISON")
    print("=" * 70)

    targets = {
        2: {
            'heat_exchanger': 89000,
            'pumps': 35000,
            'pipe_fittings': 41500,
            'instrumentation': 30000,
            'capital_total': 195500,
            'operating_energy_kwh_year': 9026
        },
        3: {
            'heat_exchanger': 68000,
            'pumps': 35000,
            'pipe_fittings': 41500,
            'instrumentation': 30000,
            'capital_total': 174500,
            'operating_energy_kwh_year': 17690
        },
        5: {
            'heat_exchanger': 50000,
            'pumps': 45000,
            'pipe_fittings': 32300,
            'instrumentation': 30000,
            'capital_total': 157300,
            'operating_energy_kwh_year': 56411
        }
    }

    for approach in [2, 3, 5]:
        print(f"\nApproach {approach}°C:")
        estimate = calculate_order_of_magnitude_estimate(1.0, 20, 10, approach)

        if estimate.get('status') != 'success':
            print("  [XX] Calculation failed")
            continue

        target_values = targets[approach]

        for component in ['heat_exchanger', 'pumps', 'pipe_fittings', 'instrumentation', 'capital_total', 'operating_energy_kwh_year']:
            calculated = estimate.get(component, 0)
            target = target_values[component]
            error_pct = abs(calculated - target) / target * 100
            status = "[OK]" if error_pct < 10 else "[!!]" if error_pct < 20 else "[XX]"

            if component == 'operating_energy_kwh_year':
                print(f"  {status} {component:30s}: {calculated:>10,.0f} kWh (target: {target:>10,.0f} kWh, error: {error_pct:>5.1f}%)")
            else:
                print(f"  {status} {component:30s}: €{calculated:>10,.0f} (target: €{target:>10,.0f}, error: {error_pct:>5.1f}%)")

else:
    print("\n[FAIL] Module test failed!")
    print(result)

print("\n" + "=" * 70)
