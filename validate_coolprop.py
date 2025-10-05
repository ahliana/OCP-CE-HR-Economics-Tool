#!/usr/bin/env python3
"""
Validation script for CoolProp integration.
Compares CoolProp values with tabulated dictionary values.
"""

import sys
sys.path.insert(0, 'python')

from python.physics.constants import (
    get_water_properties,
    get_air_properties,
    WATER_PROPERTIES,
    AIR_PROPERTIES
)

def compare_properties(temp_c, coolprop_props, dict_props, fluid_name):
    """Compare CoolProp properties with dictionary values."""
    print(f"\n{'='*80}")
    print(f"{fluid_name.upper()} PROPERTIES AT {temp_c}°C")
    print(f"{'='*80}")
    print(f"{'Property':<30} {'Dict Value':<18} {'CoolProp':<18} {'Diff %':<12}")
    print(f"{'-'*80}")

    # Properties to compare (common to both)
    common_properties = [
        'density',
        'specific_heat',
        'thermal_conductivity',
        'dynamic_viscosity',
        'kinematic_viscosity',
        'prandtl_number'
    ]

    for prop in common_properties:
        if prop in dict_props and prop in coolprop_props:
            dict_val = dict_props[prop]
            cp_val = coolprop_props[prop]
            diff_pct = ((cp_val - dict_val) / dict_val) * 100

            print(f"{prop:<30} {dict_val:<18.6g} {cp_val:<18.6g} {diff_pct:>10.3f}%")


def main():
    print("\n" + "="*80)
    print("COOLPROP VALIDATION - Comparing with Tabulated Values")
    print("="*80)

    # Test required water temperatures
    water_temps = [20, 30, 45, 60]
    water_keys = ['20C', '30C', '45C', '60C']

    print("\n" + "="*80)
    print("WATER PROPERTIES VALIDATION")
    print("="*80)

    for temp, key in zip(water_temps, water_keys):
        coolprop_props = get_water_properties(temp)
        dict_props = WATER_PROPERTIES[key]
        compare_properties(temp, coolprop_props, dict_props, "Water")

    # Test required air temperatures
    air_temps = [20, 35]
    air_keys = ['20C', '35C']

    print("\n\n" + "="*80)
    print("AIR PROPERTIES VALIDATION")
    print("="*80)

    for temp, key in zip(air_temps, air_keys):
        coolprop_props = get_air_properties(temp)
        dict_props = AIR_PROPERTIES[key]
        compare_properties(temp, coolprop_props, dict_props, "Air")

    # Test intermediate temperature (to show interpolation works)
    print("\n\n" + "="*80)
    print("INTERMEDIATE TEMPERATURE TEST (25°C - Between tabulated values)")
    print("="*80)

    water_25c = get_water_properties(25)
    print(f"\nWater at 25°C (CoolProp):")
    print(f"  Density: {water_25c['density']:.2f} kg/m³")
    print(f"  Specific heat: {water_25c['specific_heat']:.1f} J/(kg·K)")
    print(f"  Viscosity: {water_25c['dynamic_viscosity']:.3e} Pa·s")
    print(f"  Thermal conductivity: {water_25c['thermal_conductivity']:.4f} W/(m·K)")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\nNotes:")
    print("  - Small differences (<5%) are expected and acceptable")
    print("  - CoolProp uses IAPWS formulations (NIST-quality)")
    print("  - Tabulated values are from engineering handbooks")
    print("  - Both sources are valid for engineering calculations")


if __name__ == "__main__":
    main()
