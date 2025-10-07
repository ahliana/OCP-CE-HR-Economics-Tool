"""
Example Usage of Cost Calculation Module

This script demonstrates how to use the new cost calculation functions.
"""

import sys
sys.path.insert(0, 'python')

# Suppress logging for cleaner output
import logging
logging.disable(logging.CRITICAL)
import warnings
warnings.filterwarnings('ignore')

from data.loader import load_csv_files
from core.costs import (
    calculate_order_of_magnitude_estimate,
    compare_approaches,
    format_cost_summary
)

# Initialize: Load CSV data files
print("Loading data...")
load_csv_files()
print("Data loaded successfully!\n")

# ==============================================================================
# EXAMPLE 1: Single Approach Estimate
# ==============================================================================
print("=" * 80)
print("EXAMPLE 1: Order of Magnitude Estimate for Single Approach")
print("=" * 80)
print("\nSystem Parameters:")
print("  - Power: 1 MW")
print("  - Inlet Temperature: 20°C")
print("  - Temperature Rise: 10°C")
print("  - Approach: 2°C")

estimate = calculate_order_of_magnitude_estimate(
    wha=1.0,           # 1 MW system
    T1=20,             # 20°C inlet
    temp_rise=10,      # 10°C temperature rise
    approach=2         # 2°C approach temperature
)

print(format_cost_summary(estimate))

# ==============================================================================
# EXAMPLE 2: Compare All Approaches
# ==============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 2: Compare Different Temperature Approaches")
print("=" * 80)

comparison = compare_approaches(
    wha=1.0,       # 1 MW system
    T1=20,         # 20°C inlet
    temp_rise=10   # 10°C temperature rise
)

print("\nComparison Table:")
print(f"{'Approach':<12} {'Capital Cost':>15} {'Operating Energy':>20}")
print("-" * 50)

for approach_key in sorted(comparison['approaches'].keys()):
    data = comparison['approaches'][approach_key]
    print(f"{data['approach']:>2}°C         "
          f"EUR {data['capital_total']:>12,.0f}  "
          f"{data['operating_energy_kwh_year']:>15,.0f} kWh/yr")

# Show recommendation
if comparison['recommendation']:
    rec = comparison['recommendation']
    print(f"\nRecommendations:")
    print(f"  Lowest Capital Cost: {rec['lowest_capital_cost']} "
          f"(EUR {rec['lowest_capital_cost_eur']:,.0f})")
    print(f"  Lowest Operating Energy: {rec['lowest_operating_cost']} "
          f"({rec['lowest_operating_energy_kwh']:,.0f} kWh/yr)")

# ==============================================================================
# EXAMPLE 3: Detailed Cost Breakdown
# ==============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 3: Detailed Cost Breakdown for Each Approach")
print("=" * 80)

for approach in [2, 3, 5]:
    print(f"\nApproach {approach}°C:")
    estimate = calculate_order_of_magnitude_estimate(1.0, 20, 10, approach)

    print(f"  Component Costs:")
    print(f"    Heat Exchanger:  EUR {estimate['heat_exchanger']:>10,.0f}")
    print(f"    Pumps:           EUR {estimate['pumps']:>10,.0f}")
    print(f"    Pipe & Fittings: EUR {estimate['pipe_fittings']:>10,.0f}")
    print(f"    Instrumentation: EUR {estimate['instrumentation']:>10,.0f}")
    print(f"  " + "-" * 40)
    print(f"    Equipment Total: EUR {estimate['equipment_subtotal']:>10,.0f}")
    print(f"    Installation:    EUR {estimate['installation_cost']:>10,.0f}")
    print(f"    Engineering:     EUR {estimate['engineering_cost']:>10,.0f}")
    print(f"    Contingency:     EUR {estimate['contingency_cost']:>10,.0f}")
    print(f"  " + "=" * 40)
    print(f"    CAPITAL TOTAL:   EUR {estimate['capital_total']:>10,.0f}")
    print(f"\n  Operating Costs:")
    print(f"    Pump Power:      {estimate['pump_power_kw']:>10,.2f} kW")
    print(f"    Annual Energy:   {estimate['operating_energy_kwh_year']:>10,.0f} kWh/yr")
    print(f"    Annual Cost:     EUR {estimate['operating_cost_eur_year']:>10,.0f}/yr")

# ==============================================================================
# EXAMPLE 4: Custom Parameters
# ==============================================================================
print("\n" + "=" * 80)
print("EXAMPLE 4: Custom Engineering Factors")
print("=" * 80)

# Calculate with different engineering factors
custom_estimate = calculate_order_of_magnitude_estimate(
    wha=1.0,
    T1=20,
    temp_rise=10,
    approach=3,
    installation_factor=1.20,    # +20% instead of +15%
    engineering_factor=1.15,     # +15% instead of +10%
    contingency_factor=1.15      # +15% instead of +10%
)

print("\nCustom Factors Applied:")
print("  Installation:  +20%")
print("  Engineering:   +15%")
print("  Contingency:   +15%")
print(f"\nResulting Capital Cost: EUR {custom_estimate['capital_total']:>10,.0f}")

# Compare with standard factors
standard_estimate = calculate_order_of_magnitude_estimate(1.0, 20, 10, 3)
print(f"Standard Capital Cost:  EUR {standard_estimate['capital_total']:>10,.0f}")
difference = custom_estimate['capital_total'] - standard_estimate['capital_total']
print(f"Difference:             EUR {difference:>10,.0f} ({difference/standard_estimate['capital_total']*100:+.1f}%)")

print("\n" + "=" * 80)
print("Examples Complete!")
print("=" * 80)
print("\nUseful Functions:")
print("  - calculate_order_of_magnitude_estimate(wha, T1, temp_rise, approach)")
print("  - compare_approaches(wha, T1, temp_rise)")
print("  - format_cost_summary(estimate)")
print("  - calculate_operating_costs(system_data, approach)")
print("\nSee python/core/costs.py for full documentation.")
