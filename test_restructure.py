"""
Test script for restructured cost calculation
"""
import sys
sys.path.append('python')

from data.loader import load_csv_files
from core.costs import calculate_costs

# Load CSV data
print("Loading CSV data...")
load_csv_files()

# Test with approach 2
print("\nCalculating costs for 1MW system, 20°C inlet, 10°C rise, 2°C approach...")
costs = calculate_costs(1.0, 20, 10, 2)

if costs.get('status') == 'success':
    print('\n' + '='*60)
    print('SUCCESS: Cost calculation completed')
    print('='*60)

    print('\nStructure validation:')
    print('  - base_costs:', 'PASS' if 'base_costs' in costs else 'FAIL')
    print('  - contingencies:', 'PASS' if 'contingencies' in costs else 'FAIL')
    print('  - operating_costs:', 'PASS' if 'operating_costs' in costs else 'FAIL')
    print('  - capital_total:', 'PASS' if 'capital_total' in costs else 'FAIL')

    print('\n' + '='*60)
    print('BASE EQUIPMENT COSTS (Raw costs without factors):')
    print('='*60)
    for key, value in costs['base_costs'].items():
        print(f'  {key:25s}: EUR {value:>12,.0f}')

    print('\n' + '='*60)
    print('CONTINGENCIES:')
    print('='*60)
    for key, value in costs['contingencies'].items():
        print(f'  {key:25s}: EUR {value:>12,.0f}')

    print('\n' + '='*60)
    print(f'CAPITAL TOTAL (rounded):  EUR {costs["capital_total"]:>12,.0f}')
    print('='*60)

    print('\n' + '='*60)
    print('OPERATING COSTS:')
    print('='*60)
    for key, value in costs['operating_costs'].items():
        if 'kwh' in key.lower():
            print(f'  {key:25s}: {value:>15,.0f}')
        else:
            print(f'  {key:25s}: {value:>15,.2f}')

    # Test backward compatibility
    print('\n' + '='*60)
    print('BACKWARD COMPATIBILITY CHECK:')
    print('='*60)
    legacy_fields = [
        'heat_exchanger', 'pumps', 'pipe_fittings', 'instrumentation',
        'valves', 'equipment_subtotal', 'installation_cost',
        'engineering_cost', 'contingency_cost'
    ]
    all_present = all(field in costs for field in legacy_fields)
    print(f'  Legacy fields present: {"PASS" if all_present else "FAIL"}')
    if not all_present:
        missing = [f for f in legacy_fields if f not in costs]
        print(f'  Missing fields: {missing}')
else:
    print('\nFAILED: Cost calculation failed')
    print(f'Error: {costs.get("error", "Unknown error")}')
