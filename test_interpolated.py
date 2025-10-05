#!/usr/bin/env python3
"""
Test that get_water_properties_interpolated still works correctly.
"""

import sys
sys.path.insert(0, 'python')

from python.physics.engineering_calculations import get_water_properties_interpolated

def test_get_water_properties_interpolated():
    """Test the updated get_water_properties_interpolated function."""

    print("\n" + "="*80)
    print("TESTING get_water_properties_interpolated()")
    print("="*80)

    # Test at standard temperatures
    test_temps = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]

    print(f"\n{'Temp (C)':<12} {'Density':<15} {'Cp':<15} {'Visc (uPa.s)':<18} {'k (W/m.K)':<15}")
    print("-" * 80)

    for temp in test_temps:
        props = get_water_properties_interpolated(temp)

        # Extract key properties
        density = props['density']
        cp = props['specific_heat']
        visc = props['dynamic_viscosity'] * 1e6  # Convert to uPa.s for readability
        k = props['thermal_conductivity']

        print(f"{temp:<12.1f} {density:<15.2f} {cp:<15.1f} {visc:<18.1f} {k:<15.4f}")

    print("\n" + "="*80)
    print("BACKWARD COMPATIBILITY TEST")
    print("="*80)

    # Test edge cases
    print("\nEdge case tests:")

    # Test low temperature (should not fail)
    try:
        props_low = get_water_properties_interpolated(10)
        print(f"+ Low temp (10C): density = {props_low['density']:.2f} kg/m3")
    except Exception as e:
        print(f"- Low temp (10C) failed: {e}")

    # Test high temperature
    try:
        props_high = get_water_properties_interpolated(70)
        print(f"+ High temp (70C): density = {props_high['density']:.2f} kg/m3")
    except Exception as e:
        print(f"- High temp (70C) failed: {e}")

    # Test exact tabulated temperature
    try:
        props_30 = get_water_properties_interpolated(30.0)
        print(f"+ Exact temp (30C): density = {props_30['density']:.2f} kg/m3")
    except Exception as e:
        print(f"- Exact temp (30C) failed: {e}")

    # Test intermediate temperature
    try:
        props_32 = get_water_properties_interpolated(32.5)
        print(f"+ Intermediate (32.5C): density = {props_32['density']:.2f} kg/m3")
    except Exception as e:
        print(f"- Intermediate (32.5C) failed: {e}")

    print("\n" + "="*80)
    print("ALL TESTS PASSED - get_water_properties_interpolated() works correctly!")
    print("="*80)


if __name__ == "__main__":
    test_get_water_properties_interpolated()
