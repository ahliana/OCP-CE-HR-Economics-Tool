# Cost Calculation API Usage

## Overview

The `calculate_costs()` function in `python/core/costs.py` provides transparent cost breakdown separating base equipment costs from contingencies.

## Structure

The function returns a dictionary with the following structure:

```python
{
    'base_costs': {
        'heat_exchanger': float,      # Raw HX cost without factors
        'pumps': float,                # Base pump cost
        'piping_fittings': float,      # Piping and fittings cost
        'instrumentation': float,      # Instrumentation cost
        'valves': float,               # Control and isolation valves
        'equipment_subtotal': float    # Sum of all base costs
    },
    'contingencies': {
        'installation': float,         # 15% of equipment_subtotal
        'engineering': float,          # 10% of (equipment + installation)
        'contingency': float,          # 10% of (equipment + installation + engineering)
        'total_contingencies': float   # Sum of all contingencies
    },
    'capital_total': float,            # Total rounded to nearest €500
    'operating_costs': {
        'annual_energy_kwh': float,
        'annual_cost_eur': float,
        'pump_power_kw': float,
        'energy_price_eur_per_kwh': float,
        'operating_hours': float
    },
    'status': 'success'
}
```

## Usage Example

```python
from data.loader import load_csv_files
from core.costs import calculate_costs

# Load required CSV data
load_csv_files()

# Calculate costs for a 1 MW system
costs = calculate_costs(
    wha=1.0,           # System power (MW)
    T1=20,             # Inlet temperature (°C)
    temp_rise=10,      # Temperature rise (°C)
    approach=2         # Approach temperature (2, 3, or 5°C)
)

if costs['status'] == 'success':
    # Access base costs
    base = costs['base_costs']
    print(f"Heat Exchanger: €{base['heat_exchanger']:,.0f}")
    print(f"Equipment Total: €{base['equipment_subtotal']:,.0f}")

    # Access contingencies
    cont = costs['contingencies']
    print(f"Installation: €{cont['installation']:,.0f}")
    print(f"Total Contingencies: €{cont['total_contingencies']:,.0f}")

    # Access total
    print(f"Capital Total: €{costs['capital_total']:,.0f}")

    # Access operating costs
    ops = costs['operating_costs']
    print(f"Annual Energy: {ops['annual_energy_kwh']:,.0f} kWh")
    print(f"Annual Cost: €{ops['annual_cost_eur']:,.0f}")
```

## UI Display Example

For transparent display in the UI:

```python
# Display base equipment costs (collapsible section)
print("Base Equipment Costs:")
for item, cost in costs['base_costs'].items():
    if item != 'equipment_subtotal':
        print(f"  {item}: €{cost:,.0f}")
print(f"Subtotal: €{costs['base_costs']['equipment_subtotal']:,.0f}")

# Display contingencies (collapsible section)
print("\nContingencies:")
for item, cost in costs['contingencies'].items():
    if item != 'total_contingencies':
        print(f"  {item}: €{cost:,.0f}")
print(f"Subtotal: €{costs['contingencies']['total_contingencies']:,.0f}")

# Display total (prominent)
print(f"\nTotal Capital Cost: €{costs['capital_total']:,.0f}")
```

## Backward Compatibility

The function maintains legacy fields for backward compatibility:
- `heat_exchanger`, `pumps`, `pipe_fittings`, `instrumentation`, `valves`
- `equipment_subtotal`, `installation_cost`, `engineering_cost`, `contingency_cost`
- `operating_energy_kwh_year`, `operating_cost_eur_year`, `pump_power_kw`

These fields contain the same values as the new nested structure but may be deprecated in future versions.

## Contingency Factors

The contingencies are calculated cumulatively:

1. **Installation** (15%): Applied to equipment subtotal
2. **Engineering** (10%): Applied to (equipment + installation)
3. **Contingency** (10%): Applied to (equipment + installation + engineering)

Custom factors can be specified:

```python
costs = calculate_costs(
    wha=1.0,
    T1=20,
    temp_rise=10,
    approach=2,
    installation_factor=1.20,   # 20% instead of 15%
    engineering_factor=1.15,    # 15% instead of 10%
    contingency_factor=1.12     # 12% instead of 10%
)
```

## Error Handling

Always check the status field:

```python
costs = calculate_costs(1.0, 20, 10, 2)

if costs['status'] == 'success':
    # Process results
    pass
elif costs['status'] == 'failed':
    print(f"Error: {costs.get('error', 'Unknown error')}")
```

## Related Functions

- `calculate_order_of_magnitude_estimate()`: Internal function that powers `calculate_costs()`
- `compare_approaches()`: Compare multiple approach temperatures
- `format_cost_summary()`: Generate human-readable text summary
