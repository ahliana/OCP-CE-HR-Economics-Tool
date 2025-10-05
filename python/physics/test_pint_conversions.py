"""
Validation script for Mini-Stage 1b: Pint-based unit conversions
Compares old (direct calculation) vs new (pint-based) implementations
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physics.constants import CONVERSION_FACTORS
from physics.units import (
    celsius_to_kelvin,
    kelvin_to_celsius,
    liters_per_minute_to_m3_per_second,
    m3_per_second_to_liters_per_minute,
)

def old_celsius_to_kelvin(celsius: float) -> float:
    """Old implementation using direct calculation."""
    return celsius + CONVERSION_FACTORS['celsius_to_kelvin']

def old_kelvin_to_celsius(kelvin: float) -> float:
    """Old implementation using direct calculation."""
    return kelvin - CONVERSION_FACTORS['celsius_to_kelvin']

def old_liters_per_minute_to_m3_per_second(lpm: float) -> float:
    """Old implementation using direct calculation."""
    return lpm * CONVERSION_FACTORS['liters_to_m3'] / CONVERSION_FACTORS['minutes_to_seconds']

def old_m3_per_second_to_liters_per_minute(m3s: float) -> float:
    """Old implementation using direct calculation."""
    return m3s / CONVERSION_FACTORS['liters_to_m3'] * CONVERSION_FACTORS['minutes_to_seconds']


def compare_results(test_name, old_func, new_func, test_values, tolerance=1e-10):
    """Compare old vs new implementation results."""
    print(f"\n{'='*70}")
    print(f"Testing: {test_name}")
    print(f"{'='*70}")
    print(f"{'Input':<15} {'Old Result':<20} {'New Result':<20} {'Diff':<15} {'Status':<10}")
    print(f"{'-'*70}")

    all_passed = True
    for value in test_values:
        old_result = old_func(value)
        new_result = new_func(value)
        diff = abs(old_result - new_result)
        status = "PASS" if diff <= tolerance else "FAIL"

        if diff > tolerance:
            all_passed = False

        print(f"{value:<15.6f} {old_result:<20.10f} {new_result:<20.10f} {diff:<15.2e} {status:<10}")

    return all_passed


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("MINI-STAGE 1b VALIDATION: PINT-BASED UNIT CONVERSIONS")
    print("="*70)
    print("\nComparing old (direct) vs new (pint-based) implementations")
    print("Tolerance: 1e-10 (floating point precision)")

    all_tests_passed = True

    # Test 1: Celsius to Kelvin
    test_values_temp = [0.0, 20.0, 30.0, 45.0, 60.0, 100.0, -273.15]
    passed = compare_results(
        "celsius_to_kelvin",
        old_celsius_to_kelvin,
        celsius_to_kelvin,
        test_values_temp
    )
    all_tests_passed = all_tests_passed and passed

    # Test 2: Kelvin to Celsius
    test_values_kelvin = [273.15, 293.15, 303.15, 318.15, 333.15, 373.15, 0.0]
    passed = compare_results(
        "kelvin_to_celsius",
        old_kelvin_to_celsius,
        kelvin_to_celsius,
        test_values_kelvin
    )
    all_tests_passed = all_tests_passed and passed

    # Test 3: Liters per minute to m³/s
    test_values_flow = [0.0, 100.0, 1000.0, 1493.0, 5000.0, 10000.0]
    passed = compare_results(
        "liters_per_minute_to_m3_per_second",
        old_liters_per_minute_to_m3_per_second,
        liters_per_minute_to_m3_per_second,
        test_values_flow
    )
    all_tests_passed = all_tests_passed and passed

    # Test 4: m³/s to liters per minute
    test_values_m3s = [0.0, 0.001, 0.01, 0.01667, 0.05, 0.1]
    passed = compare_results(
        "m3_per_second_to_liters_per_minute",
        old_m3_per_second_to_liters_per_minute,
        m3_per_second_to_liters_per_minute,
        test_values_m3s
    )
    all_tests_passed = all_tests_passed and passed

    # Round-trip tests
    print(f"\n{'='*70}")
    print(f"Round-trip tests (conversion and back)")
    print(f"{'='*70}")

    # Temperature round-trip
    print("\nTemperature (C -> K -> C):")
    for temp_c in [20.0, 30.0, 45.0]:
        temp_k = celsius_to_kelvin(temp_c)
        temp_c_back = kelvin_to_celsius(temp_k)
        diff = abs(temp_c - temp_c_back)
        status = "PASS" if diff < 1e-10 else "FAIL"
        print(f"  {temp_c}C -> {temp_k}K -> {temp_c_back}C (diff: {diff:.2e}) {status}")
        if diff >= 1e-10:
            all_tests_passed = False

    # Flow round-trip
    print("\nFlow (L/min -> m³/s -> L/min):")
    for flow_lpm in [1000.0, 1493.0, 5000.0]:
        flow_m3s = liters_per_minute_to_m3_per_second(flow_lpm)
        flow_lpm_back = m3_per_second_to_liters_per_minute(flow_m3s)
        diff = abs(flow_lpm - flow_lpm_back)
        status = "PASS" if diff < 1e-8 else "FAIL"  # Slightly larger tolerance for flow
        print(f"  {flow_lpm} L/min -> {flow_m3s:.6f} m3/s -> {flow_lpm_back:.6f} L/min (diff: {diff:.2e}) {status}")
        if diff >= 1e-8:
            all_tests_passed = False

    # Final summary
    print(f"\n{'='*70}")
    if all_tests_passed:
        print("ALL TESTS PASSED - Pint implementations match original results!")
    else:
        print("SOME TESTS FAILED - Review differences above")
    print(f"{'='*70}\n")

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    exit(main())
