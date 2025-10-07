# Cost Calculation Restructuring - Implementation Summary

## Overview

The cost calculation system has been restructured to provide transparent separation between base equipment costs and contingencies, enabling clear display in the UI.

## Changes Made

### 1. Core API Function

**New Function**: `calculate_costs()` in [python/core/costs.py:412](../python/core/costs.py#L412)

This is the main entry point for cost calculations. It wraps `calculate_order_of_magnitude_estimate()` with a cleaner API.

### 2. Structured Return Format

The function now returns a clearly structured dictionary:

```python
{
    'base_costs': {
        'heat_exchanger': float,      # Raw HX cost
        'pumps': float,                # Base pump cost
        'piping_fittings': float,      # Pipes + fittings
        'instrumentation': float,      # Controls & instruments
        'valves': float,               # Control + isolation valves
        'equipment_subtotal': float    # Sum of all base costs
    },
    'contingencies': {
        'installation': float,         # 15% of equipment
        'engineering': float,          # 10% of (equipment + installation)
        'contingency': float,          # 10% of all previous
        'total_contingencies': float   # Sum of all contingencies
    },
    'capital_total': float,            # Total rounded to €500
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

### 3. Backward Compatibility

All legacy field names are preserved in the return dictionary to maintain compatibility:
- `heat_exchanger`, `pumps`, `pipe_fittings`, `instrumentation`, `valves`
- `equipment_subtotal`, `installation_cost`, `engineering_cost`, `contingency_cost`
- `operating_energy_kwh_year`, `operating_cost_eur_year`, `pump_power_kw`

### 4. Updated Functions

**Modified Functions**:
- `calculate_order_of_magnitude_estimate()` - Returns new structured format
- `format_cost_summary()` - Updated to handle both new and legacy format
- `compare_approaches()` - Works with new structure

**New Functions**:
- `calculate_costs()` - Clean API wrapper

## Usage Example

```python
from data.loader import load_csv_files
from core.costs import calculate_costs

# Load CSV data
load_csv_files()

# Calculate costs
costs = calculate_costs(
    wha=1.0,           # System power (MW)
    T1=20,             # Inlet temperature (°C)
    temp_rise=10,      # Temperature rise (°C)
    approach=2         # Approach temperature (2, 3, or 5°C)
)

# Access structured costs
if costs['status'] == 'success':
    print(f"Base Equipment: €{costs['base_costs']['equipment_subtotal']:,.0f}")
    print(f"Contingencies: €{costs['contingencies']['total_contingencies']:,.0f}")
    print(f"Total Capital: €{costs['capital_total']:,.0f}")
```

## Test Results

All tests passing:

### Test 1: Individual Approach Testing
- **Approach 2°C**: Base Equipment €96,710, Contingencies €37,862, Total €134,500 ✓
- **Approach 3°C**: Base Equipment €93,270, Contingencies €36,515, Total €130,000 ✓
- **Approach 5°C**: Base Equipment €99,851, Contingencies €39,092, Total €139,000 ✓

### Test 2: Comparison Function
- `compare_approaches()` works correctly with new structure ✓

### Test 3: Backward Compatibility
- All legacy fields present ✓
- Legacy fields match new structured values ✓

## UI Implementation Notes

### Recommended Display Structure

```
┌─────────────────────────────────────────┐
│ CAPITAL COST BREAKDOWN                   │
├─────────────────────────────────────────┤
│                                          │
│ ▼ Base Equipment Costs                   │
│   • Heat Exchanger:        €17,616      │
│   • Pumps:                 €35,000      │
│   • Piping & Fittings:     €14,094      │
│   • Instrumentation:       €30,000      │
│   • Valves:                    €0       │
│   ─────────────────────────────────────  │
│   Subtotal:                €96,710      │
│                                          │
│ ▼ Contingencies                          │
│   • Installation (15%):    €14,506      │
│   • Engineering (10%):     €11,122      │
│   • Contingency (10%):     €12,234      │
│   ─────────────────────────────────────  │
│   Subtotal:                €37,862      │
│                                          │
│ ═════════════════════════════════════════│
│ TOTAL CAPITAL COST:       €134,500      │
│ ═════════════════════════════════════════│
└─────────────────────────────────────────┘
```

### Collapsible Sections

Both "Base Equipment Costs" and "Contingencies" can be collapsible sections, with the Total Capital Cost always visible.

### Tooltips

- **Installation**: Labor and installation costs (15% of equipment)
- **Engineering**: Design, drawings, and project management (10% cumulative)
- **Contingency**: Project risk buffer (10% cumulative)

## Documentation

- **API Documentation**: [Cost_API_Usage.md](Cost_API_Usage.md)
- **UI Mapping**: [UI_Calculation_Map.md](UI_Calculation_Map.md) (updated)

## Benefits

1. **Transparency**: Users can see exactly where costs come from
2. **Flexibility**: UI can show/hide details as needed
3. **Maintainability**: Clear separation of concerns
4. **Backward Compatible**: Existing code continues to work
5. **Testable**: Structure enables better validation

## Future Enhancements

Potential future improvements:
1. Make contingency factors configurable per approach
2. Add cost breakdown by material vs. labor
3. Include detailed pump sizing rationale
4. Add cost sensitivity analysis
5. Export cost breakdown to PDF/Excel

## Migration Notes

**For UI Developers**:
- Use new `base_costs` and `contingencies` dictionaries for display
- Legacy fields remain available but are deprecated
- Test UI with all three approaches (2°C, 3°C, 5°C)

**For Backend Developers**:
- `calculate_costs()` is the preferred API function
- `calculate_order_of_magnitude_estimate()` still works but use `calculate_costs()` for new code
- Both functions return the same structured format

## Files Modified

1. [python/core/costs.py](../python/core/costs.py)
   - Added `calculate_costs()` function (Line 412)
   - Modified `calculate_order_of_magnitude_estimate()` (Line 480)
   - Updated `format_cost_summary()` (Line 667)

2. [docs/UI_Calculation_Map.md](UI_Calculation_Map.md)
   - Updated cost components section with new structure

3. [docs/Cost_API_Usage.md](Cost_API_Usage.md) (NEW)
   - Complete API documentation with examples

## Testing

Test scripts available:
- `test_restructure.py` - Basic structure validation
- `test_all_approaches.py` - Comprehensive testing of all approaches

Run tests:
```bash
cd C:\Files\Code\OCP-CE-HR-Economics-Tool
python test_all_approaches.py
```

---

**Date**: 2025-10-07
**Status**: ✓ Completed and tested
**Breaking Changes**: None (backward compatible)
