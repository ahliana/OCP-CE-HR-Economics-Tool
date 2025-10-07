"""
Test script to verify economics panel data is correct.
Tests that valve costs and contingency costs are properly populated.
"""

import sys
sys.path.insert(0, 'python')

from data.loader import load_csv_files
from core.costs import compare_approaches

# Load CSV data
print("Loading CSV data...")
load_csv_files()

# Test comparison for 1MW system
print("\n" + "=" * 80)
print("Testing compare_approaches function")
print("=" * 80)

wha = 1.0
T1 = 20
temp_rise = 10

comparison = compare_approaches(wha, T1, temp_rise, approaches=[2, 3, 5])

if comparison.get('status') == 'success':
    print(f"\n✓ Comparison successful for {wha}MW system, {T1}°C inlet, {temp_rise}°C rise")

    approaches_data = comparison['approaches']

    for approach_key in ['2C', '3C', '5C']:
        data = approaches_data.get(approach_key, {})

        print(f"\n{'=' * 80}")
        print(f"Approach: {approach_key.replace('C', '°C')}")
        print(f"{'=' * 80}")

        # Base equipment costs
        print("\nBASE EQUIPMENT COSTS:")
        print(f"  Heat Exchanger:     €{data.get('heat_exchanger', 0):>12,.0f}")
        print(f"  Pumps:              €{data.get('pumps', 0):>12,.0f}")
        print(f"  Pipe & Fittings:    €{data.get('pipe_fittings', 0):>12,.0f}")
        print(f"  Instrumentation:    €{data.get('instrumentation', 0):>12,.0f}")
        print(f"  Valves:             €{data.get('valves', 0):>12,.0f}")

        # Calculate equipment subtotal
        equipment_subtotal = sum([
            data.get('heat_exchanger', 0),
            data.get('pumps', 0),
            data.get('pipe_fittings', 0),
            data.get('instrumentation', 0),
            data.get('valves', 0)
        ])
        print(f"  {'─' * 40}")
        print(f"  Equipment Subtotal: €{equipment_subtotal:>12,.0f}")

        # Contingencies
        print("\nCONTINGENCIES:")
        print(f"  Installation (15%): €{data.get('installation_cost', 0):>12,.0f}")
        print(f"  Engineering (10%):  €{data.get('engineering_cost', 0):>12,.0f}")
        print(f"  Contingency (10%):  €{data.get('contingency_cost', 0):>12,.0f}")

        # Calculate I&C subtotal
        ic_subtotal = sum([
            data.get('installation_cost', 0),
            data.get('engineering_cost', 0),
            data.get('contingency_cost', 0)
        ])
        print(f"  {'─' * 40}")
        print(f"  I&C Subtotal:       €{ic_subtotal:>12,.0f}")

        # Capital Total
        print("\nCAPITAL TOTAL:")
        capital_total = data.get('capital_total', 0)
        expected_total = equipment_subtotal + ic_subtotal
        print(f"  Capital Total:      €{capital_total:>12,.0f}")
        print(f"  Expected Total:     €{expected_total:>12,.0f}")

        # Validation
        diff = abs(capital_total - expected_total)
        if diff <= 500:
            print(f"  ✓ Validation PASSED (difference: €{diff:.0f})")
        else:
            print(f"  ⚠ Validation WARNING (difference: €{diff:.0f})")

        # Operating costs
        print("\nOPERATING COSTS:")
        print(f"  Annual Energy:      {data.get('operating_energy_kwh_year', 0):>12,.0f} kWh")
        print(f"  Annual Cost:        €{data.get('operating_cost_eur_year', 0):>12,.0f}")
        print(f"  Pump Power:         {data.get('pump_power_kw', 0):>12,.2f} kW")

    # Summary validation
    print(f"\n{'=' * 80}")
    print("VALIDATION SUMMARY")
    print(f"{'=' * 80}")

    issues_found = []

    for approach_key in ['2C', '3C', '5C']:
        data = approaches_data.get(approach_key, {})

        # Check for zero valves
        if data.get('valves', 0) == 0:
            issues_found.append(f"❌ {approach_key}: Valves cost is 0")
        else:
            print(f"✓ {approach_key}: Valves cost = €{data.get('valves', 0):,.0f}")

        # Check for zero contingencies
        if data.get('installation_cost', 0) == 0:
            issues_found.append(f"❌ {approach_key}: Installation cost is 0")
        else:
            print(f"✓ {approach_key}: Installation cost = €{data.get('installation_cost', 0):,.0f}")

        if data.get('engineering_cost', 0) == 0:
            issues_found.append(f"❌ {approach_key}: Engineering cost is 0")
        else:
            print(f"✓ {approach_key}: Engineering cost = €{data.get('engineering_cost', 0):,.0f}")

        if data.get('contingency_cost', 0) == 0:
            issues_found.append(f"❌ {approach_key}: Contingency cost is 0")
        else:
            print(f"✓ {approach_key}: Contingency cost = €{data.get('contingency_cost', 0):,.0f}")

    if issues_found:
        print(f"\n{'=' * 80}")
        print("ISSUES FOUND:")
        print(f"{'=' * 80}")
        for issue in issues_found:
            print(issue)
    else:
        print(f"\n{'=' * 80}")
        print("✓ ALL VALIDATION CHECKS PASSED!")
        print(f"{'=' * 80}")

else:
    print(f"❌ Comparison failed: {comparison.get('error', 'Unknown error')}")
