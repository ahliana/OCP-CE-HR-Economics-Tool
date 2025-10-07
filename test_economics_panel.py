"""
Test Economics Panel
Validate the economics analysis panel functionality
"""

import sys
import os
import io

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

from data.loader import load_csv_files
from core.costs import compare_approaches, calculate_order_of_magnitude_estimate

def test_economics_comparison():
    """Test the economics comparison functionality."""
    print("=" * 60)
    print("Testing Economics Analysis Panel")
    print("=" * 60)

    # Load CSV data
    print("\n1. Loading CSV data...")
    load_csv_files()
    print("   ✓ CSV data loaded")

    # Test parameters
    wha = 1.0
    T1 = 20
    temp_rise = 10

    print(f"\n2. Test Parameters:")
    print(f"   - System Power: {wha} MW")
    print(f"   - Inlet Temperature: {T1}°C")
    print(f"   - Temperature Rise: {temp_rise}°C")

    # Test comparison function
    print("\n3. Running approach comparison (2°C, 3°C, 5°C)...")
    comparison = compare_approaches(wha, T1, temp_rise, approaches=[2, 3, 5])

    if comparison.get('status') == 'success':
        print("   ✓ Comparison successful")

        # Display results table
        print("\n4. Economics Comparison Results:")
        print("   " + "-" * 90)
        print(f"   {'Component':<25} {'2°C Approach':>20} {'3°C Approach':>20} {'5°C Approach':>20}")
        print("   " + "-" * 90)

        approaches_data = comparison['approaches']

        # Capital costs
        components = [
            ('Heat Exchanger', 'heat_exchanger'),
            ('Pumps', 'pumps'),
            ('Pipe & Fittings', 'pipe_fittings'),
            ('Instruments', 'instrumentation'),
            ('Capital Total', 'capital_total'),
        ]

        for label, key in components:
            val_2c = approaches_data.get('2C', {}).get(key, 0)
            val_3c = approaches_data.get('3C', {}).get(key, 0)
            val_5c = approaches_data.get('5C', {}).get(key, 0)
            print(f"   {label:<25} €{val_2c:>18,.0f} €{val_3c:>18,.0f} €{val_5c:>18,.0f}")

        print("   " + "-" * 90)

        # Operating costs
        print(f"\n5. Annual Operating Costs:")
        for approach in ['2C', '3C', '5C']:
            energy_kwh = approaches_data.get(approach, {}).get('operating_energy_kwh_year', 0)
            cost_eur = approaches_data.get(approach, {}).get('operating_cost_eur_year', 0)
            print(f"   {approach} Approach: {energy_kwh:>10,.0f} kWh/year → €{cost_eur:>10,.0f}/year")

        # Recommendations
        if comparison.get('recommendation'):
            rec = comparison['recommendation']
            print(f"\n6. Recommendations:")
            print(f"   - Lowest Capital Cost: {rec.get('lowest_capital_cost', 'N/A')} (€{rec.get('lowest_capital_cost_eur', 0):,.0f})")
            print(f"   - Lowest Operating Cost: {rec.get('lowest_operating_cost', 'N/A')} ({rec.get('lowest_operating_energy_kwh', 0):,.0f} kWh/year)")

    else:
        print("   ✗ Comparison failed")
        print(f"   Error: {comparison.get('error', 'Unknown error')}")

    # Test individual calculation
    print("\n7. Testing individual approach calculation (3°C)...")
    estimate = calculate_order_of_magnitude_estimate(wha, T1, temp_rise, 3)

    if estimate.get('status') == 'success':
        print("   ✓ Calculation successful")
        print(f"   - Heat Exchanger: €{estimate['heat_exchanger']:,.0f}")
        print(f"   - Pumps: €{estimate['pumps']:,.0f}")
        print(f"   - Pipe & Fittings: €{estimate['pipe_fittings']:,.0f}")
        print(f"   - Instrumentation: €{estimate['instrumentation']:,.0f}")
        print(f"   - Capital Total: €{estimate['capital_total']:,.0f}")
        print(f"   - Annual Operating Energy: {estimate['operating_energy_kwh_year']:,.0f} kWh/year")
        print(f"   - Annual Operating Cost: €{estimate['operating_cost_eur_year']:,.0f}/year")
    else:
        print("   ✗ Calculation failed")
        print(f"   Error: {estimate.get('error', 'Unknown error')}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_economics_comparison()
