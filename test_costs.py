"""
Test script for costs.py module
"""

import sys
sys.path.insert(0, 'python')

from data.loader import load_csv_files
from core.costs import compare_approaches, format_cost_summary, calculate_order_of_magnitude_estimate

# Load CSV data
print("Loading CSV data...")
load_csv_files()
print("CSV data loaded.\n")

# Test 1: Compare approaches
print("=" * 70)
print("TEST 1: Compare Approaches (2°C, 3°C, 5°C)")
print("=" * 70)

result = compare_approaches(1.0, 20, 10)

if result.get('status') == 'success':
    print("✅ Test successful!\n")

    for approach_key, data in result['approaches'].items():
        print(f"{approach_key.replace('C', '°C')}:")
        print(f"  Heat Exchanger:   €{data['heat_exchanger']:>10,.0f}")
        print(f"  Pumps:            €{data['pumps']:>10,.0f}")
        print(f"  Pipe & Fittings:  €{data['pipe_fittings']:>10,.0f}")
        print(f"  Instrumentation:  €{data['instrumentation']:>10,.0f}")
        print(f"  Capital Total:    €{data['capital_total']:>10,.0f}")
        print(f"  Operating Energy:  {data['operating_energy_kwh_year']:>9,.0f} kWh/year")
        print()
else:
    print("❌ Test failed!")
    print(result)

# Test 2: Detailed estimate for 2°C approach
print("=" * 70)
print("TEST 2: Detailed Estimate - Approach 2°C")
print("=" * 70)

estimate = calculate_order_of_magnitude_estimate(1.0, 20, 10, 2)

if estimate.get('status') == 'success':
    print(format_cost_summary(estimate))
else:
    print("❌ Test failed!")
    print(estimate)

# Test 3: Target value comparison
print("=" * 70)
print("TEST 3: Validation Against Target Values")
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

for approach, target_values in targets.items():
    print(f"\nApproach {approach}°C:")
    estimate = calculate_order_of_magnitude_estimate(1.0, 20, 10, approach)

    if estimate.get('status') != 'success':
        print("  ❌ Calculation failed")
        continue

    for component, target in target_values.items():
        calculated = estimate.get(component, 0)
        error_pct = abs(calculated - target) / target * 100
        status = "✅" if error_pct < 10 else "⚠️" if error_pct < 20 else "❌"

        print(f"  {status} {component:30s}: €{calculated:>10,.0f} (target: €{target:>10,.0f}, error: {error_pct:>5.1f}%)")

print("\n" + "=" * 70)
print("Testing complete!")
