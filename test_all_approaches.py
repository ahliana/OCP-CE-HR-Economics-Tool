"""
Test cost restructuring for all three approaches
"""
import sys
sys.path.append('python')

from data.loader import load_csv_files
from core.costs import calculate_costs, compare_approaches

print("="*80)
print("COST RESTRUCTURING VALIDATION TEST")
print("="*80)

# Load CSV data
print("\nLoading CSV data...")
load_csv_files()

print("\n" + "="*80)
print("TEST 1: Individual Approach Testing")
print("="*80)

approaches = [2, 3, 5]
all_passed = True

for approach in approaches:
    print(f"\n--- Testing Approach {approach}°C ---")

    costs = calculate_costs(1.0, 20, 10, approach)

    if costs.get('status') == 'success':
        # Verify structure
        has_base = 'base_costs' in costs
        has_cont = 'contingencies' in costs
        has_ops = 'operating_costs' in costs
        has_total = 'capital_total' in costs

        structure_ok = has_base and has_cont and has_ops and has_total

        if structure_ok:
            print(f"  [PASS] Structure")

            # Verify calculations
            base_sum = sum(v for k, v in costs['base_costs'].items() if k != 'equipment_subtotal')
            expected_subtotal = costs['base_costs']['equipment_subtotal']

            calc_ok = abs(base_sum - expected_subtotal) < 0.01

            if calc_ok:
                print(f"  [PASS] Calculations")
                print(f"    - Base Equipment: EUR {costs['base_costs']['equipment_subtotal']:,.0f}")
                print(f"    - Contingencies: EUR {costs['contingencies']['total_contingencies']:,.0f}")
                print(f"    - Capital Total: EUR {costs['capital_total']:,.0f}")
            else:
                print(f"  [FAIL] Calculations (sum mismatch)")
                all_passed = False
        else:
            print(f"  [FAIL] Structure")
            print(f"    - base_costs: {has_base}")
            print(f"    - contingencies: {has_cont}")
            print(f"    - operating_costs: {has_ops}")
            print(f"    - capital_total: {has_total}")
            all_passed = False
    else:
        print(f"  [FAIL] Calculation FAILED: {costs.get('error')}")
        all_passed = False

print("\n" + "="*80)
print("TEST 2: Comparison Function")
print("="*80)

comparison = compare_approaches(1.0, 20, 10)

if comparison.get('status') == 'success':
    print("\n[PASS] compare_approaches() executed successfully")
    print("\nApproach Comparison:")
    for approach_key, data in sorted(comparison['approaches'].items()):
        print(f"\n  {approach_key}:")
        print(f"    Capital Total: EUR {data['capital_total']:>10,.0f}")
        print(f"    Operating (kWh/yr): {data['operating_energy_kwh_year']:>9,.0f}")

    if comparison.get('recommendation'):
        rec = comparison['recommendation']
        print("\n  Recommendations:")
        print(f"    - Lowest capital cost: {rec['lowest_capital_cost']}")
        print(f"    - Lowest operating cost: {rec['lowest_operating_cost']}")
else:
    print("\n[FAIL] compare_approaches() FAILED")
    all_passed = False

print("\n" + "="*80)
print("TEST 3: Backward Compatibility")
print("="*80)

costs = calculate_costs(1.0, 20, 10, 2)
legacy_fields = [
    'heat_exchanger', 'pumps', 'pipe_fittings', 'instrumentation',
    'valves', 'equipment_subtotal', 'installation_cost',
    'engineering_cost', 'contingency_cost', 'operating_energy_kwh_year',
    'operating_cost_eur_year', 'pump_power_kw'
]

missing_fields = [f for f in legacy_fields if f not in costs]

if not missing_fields:
    print("\n[PASS] All legacy fields present")

    # Verify legacy fields match new structure
    matches = True
    if costs['heat_exchanger'] != costs['base_costs']['heat_exchanger']:
        print("  [FAIL] heat_exchanger mismatch")
        matches = False
    if costs['pumps'] != costs['base_costs']['pumps']:
        print("  [FAIL] pumps mismatch")
        matches = False
    if costs['installation_cost'] != costs['contingencies']['installation']:
        print("  [FAIL] installation_cost mismatch")
        matches = False

    if matches:
        print("[PASS] Legacy fields match new structure")
    else:
        all_passed = False
else:
    print(f"\n[FAIL] Missing legacy fields: {missing_fields}")
    all_passed = False

print("\n" + "="*80)
if all_passed:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
print("="*80)
