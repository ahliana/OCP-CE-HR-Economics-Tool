"""
STAGE 3 VALIDATION: Heat Transfer Library Replacement

Compares old manual implementations vs new library implementations:
- Prandtl: manual vs fluids.Prandtl()
- Nusselt laminar: manual vs ht.Nu_conv_internal()
- Nusselt turbulent: manual Gnielinski vs ht.Nu_conv_internal(Method='Gnielinski')
- Graetz: manual vs fluids.Graetz_heat()
"""

import math
import fluids
import ht
from constants import WATER_PROPERTIES

# =============================================================================
# OLD IMPLEMENTATIONS (for comparison)
# =============================================================================

def prandtl_number_OLD(specific_heat, dynamic_viscosity, thermal_conductivity):
    """OLD: Manual calculation"""
    if thermal_conductivity <= 0:
        raise ValueError("Thermal conductivity must be positive")
    return (specific_heat * dynamic_viscosity) / thermal_conductivity


def graetz_number_OLD(reynolds, prandtl, length_diameter_ratio):
    """OLD: Manual calculation"""
    return reynolds * prandtl * length_diameter_ratio


def nusselt_number_laminar_pipe_OLD(reynolds, prandtl, length_diameter_ratio=None):
    """OLD: Manual laminar correlation"""
    if reynolds >= 2300:
        raise ValueError("Use turbulent correlation for Re >= 2300")

    if length_diameter_ratio is None or length_diameter_ratio > 60:
        return 4.36  # Fully developed
    else:
        gz = graetz_number_OLD(reynolds, prandtl, 1.0 / length_diameter_ratio)
        if gz > 100:
            return 1.86 * (gz)**(1/3) * (prandtl / 0.7)**(0.14)
        else:
            return 4.36


def nusselt_number_turbulent_pipe_OLD(reynolds, prandtl, length_diameter_ratio=None):
    """OLD: Manual Gnielinski correlation"""
    if reynolds < 2300:
        raise ValueError("Use laminar correlation for Re < 2300")

    if reynolds < 10000:
        # Transition region
        f = (0.79 * math.log(reynolds) - 1.64)**(-2)
        numerator = (f/8) * (reynolds - 1000) * prandtl
        denominator = 1 + 12.7 * math.sqrt(f/8) * (prandtl**(2/3) - 1)
        nu_gnielinski = numerator / denominator

        weight = (reynolds - 2300) / (10000 - 2300)
        nu_laminar = 4.36
        return weight * nu_gnielinski + (1 - weight) * nu_laminar
    else:
        # Fully turbulent - Gnielinski
        f = (0.79 * math.log(reynolds) - 1.64)**(-2)
        numerator = (f/8) * (reynolds - 1000) * prandtl
        denominator = 1 + 12.7 * math.sqrt(f/8) * (prandtl**(2/3) - 1)
        return numerator / denominator


# =============================================================================
# NEW IMPLEMENTATIONS (library-based)
# =============================================================================

def prandtl_number_NEW(specific_heat, dynamic_viscosity, thermal_conductivity):
    """NEW: Using fluids.Prandtl()"""
    if thermal_conductivity <= 0:
        raise ValueError("Thermal conductivity must be positive")
    return fluids.Prandtl(Cp=specific_heat, mu=dynamic_viscosity, k=thermal_conductivity)


def graetz_number_NEW(reynolds, prandtl, length_diameter_ratio):
    """NEW: Still using manual (simple formula)"""
    return reynolds * prandtl * length_diameter_ratio


def nusselt_number_laminar_pipe_NEW(reynolds, prandtl, length_diameter_ratio=None):
    """NEW: Using ht.Nu_conv_internal()"""
    if reynolds >= 2300:
        raise ValueError("Use turbulent correlation for Re >= 2300")
    return ht.Nu_conv_internal(Re=reynolds, Pr=prandtl, Method='Laminar - constant Q')


def nusselt_number_turbulent_pipe_NEW(reynolds, prandtl, length_diameter_ratio=None):
    """NEW: Using ht.Nu_conv_internal() with Gnielinski"""
    if reynolds < 2300:
        raise ValueError("Use laminar correlation for Re < 2300")
    return ht.Nu_conv_internal(Re=reynolds, Pr=prandtl, Method='Gnielinski')


# =============================================================================
# VALIDATION TESTS
# =============================================================================

def validate_prandtl():
    """Compare Prandtl number calculations"""
    print("\n" + "="*80)
    print("PRANDTL NUMBER VALIDATION")
    print("="*80)

    test_cases = [
        ("Water 20°C", WATER_PROPERTIES['20C']),
        ("Water 30°C", WATER_PROPERTIES['30C']),
        ("Water 45°C", WATER_PROPERTIES['45C']),
    ]

    results = []
    for name, props in test_cases:
        cp = props['specific_heat']
        mu = props['dynamic_viscosity']
        k = props['thermal_conductivity']

        old = prandtl_number_OLD(cp, mu, k)
        new = prandtl_number_NEW(cp, mu, k)
        diff = abs(old - new) / old * 100 if old != 0 else 0

        print(f"\n{name}:")
        print(f"  Cp={cp:.1f} J/kg-K, mu={mu:.6f} Pa-s, k={k:.3f} W/m-K")
        print(f"  OLD (manual):        Pr = {old:.4f}")
        print(f"  NEW (fluids):        Pr = {new:.4f}")
        print(f"  Difference:          {diff:.6f}%")

        results.append({
            'test': name,
            'old': old,
            'new': new,
            'diff_pct': diff,
            'pass': diff < 0.01  # <0.01% difference
        })

    return results


def validate_nusselt_laminar():
    """Compare laminar Nusselt calculations"""
    print("\n" + "="*80)
    print("NUSSELT NUMBER - LAMINAR FLOW")
    print("="*80)

    test_cases = [
        ("Laminar Re=1000, Pr=6", 1000, 6.0),
        ("Laminar Re=2000, Pr=6", 2000, 6.0),
        ("Laminar Re=500, Pr=5", 500, 5.0),
    ]

    results = []
    for name, re, pr in test_cases:
        old = nusselt_number_laminar_pipe_OLD(re, pr)
        new = nusselt_number_laminar_pipe_NEW(re, pr)
        diff = abs(old - new) / old * 100 if old != 0 else 0

        print(f"\n{name}:")
        print(f"  OLD (manual):        Nu = {old:.4f}")
        print(f"  NEW (ht library):    Nu = {new:.4f}")
        print(f"  Difference:          {diff:.2f}%")

        results.append({
            'test': name,
            'old': old,
            'new': new,
            'diff_pct': diff,
            'pass': diff < 1.0  # <1% acceptable
        })

    return results


def validate_nusselt_turbulent():
    """Compare turbulent Nusselt calculations"""
    print("\n" + "="*80)
    print("NUSSELT NUMBER - TURBULENT FLOW (GNIELINSKI)")
    print("="*80)

    test_cases = [
        ("Turbulent Re=10,000, Pr=6", 10000, 6.0),
        ("Turbulent Re=50,000, Pr=6", 50000, 6.0),
        ("Turbulent Re=100,000, Pr=6", 100000, 6.0),
        ("Transition Re=5,000, Pr=6", 5000, 6.0),
    ]

    results = []
    for name, re, pr in test_cases:
        old = nusselt_number_turbulent_pipe_OLD(re, pr)
        new = nusselt_number_turbulent_pipe_NEW(re, pr)
        diff = abs(old - new) / old * 100 if old != 0 else 0

        print(f"\n{name}:")
        print(f"  OLD (manual Gniel):  Nu = {old:.4f}")
        print(f"  NEW (ht Gnielinski): Nu = {new:.4f}")
        print(f"  Difference:          {diff:.2f}%")

        results.append({
            'test': name,
            'old': old,
            'new': new,
            'diff_pct': diff,
            'pass': diff < 5.0  # <5% acceptable (different implementations)
        })

    return results


def validate_graetz():
    """Compare Graetz number calculations"""
    print("\n" + "="*80)
    print("GRAETZ NUMBER")
    print("="*80)

    test_cases = [
        ("Re=10000, Pr=6, D/L=0.1", 10000, 6.0, 0.1),
        ("Re=5000, Pr=5, D/L=0.05", 5000, 5.0, 0.05),
    ]

    results = []
    for name, re, pr, d_over_l in test_cases:
        old = graetz_number_OLD(re, pr, d_over_l)
        new = graetz_number_NEW(re, pr, d_over_l)
        diff = abs(old - new) / old * 100 if old != 0 else 0

        print(f"\n{name}:")
        print(f"  OLD (manual):        Gz = {old:.4f}")
        print(f"  NEW (still manual):  Gz = {new:.4f}")
        print(f"  Difference:          {diff:.6f}%")

        results.append({
            'test': name,
            'old': old,
            'new': new,
            'diff_pct': diff,
            'pass': diff < 0.01  # Should be identical
        })

    return results


def validate_cdu_conditions():
    """Validate with typical CDU operating conditions"""
    print("\n" + "="*80)
    print("CDU TYPICAL CONDITIONS VALIDATION")
    print("="*80)

    # Water at 25°C (interpolated properties)
    props_25c = {
        'density': 997.0,
        'specific_heat': 4182.0,
        'thermal_conductivity': 0.607,
        'dynamic_viscosity': 0.000891,
        'kinematic_viscosity': 8.94e-7,
        'prandtl_number': 6.14
    }

    print("\nTest Case: Water at 25°C in CDU")
    print(f"  Properties: Pr={props_25c['prandtl_number']:.2f}")

    # Test various Reynolds numbers
    re_values = [1000, 2000, 5000, 10000, 50000, 100000]

    print("\n  Reynolds | Nu (OLD) | Nu (NEW) | Diff %  | Flow Regime")
    print("  " + "-"*65)

    for re in re_values:
        pr = props_25c['prandtl_number']

        if re < 2300:
            old_nu = nusselt_number_laminar_pipe_OLD(re, pr)
            new_nu = nusselt_number_laminar_pipe_NEW(re, pr)
            regime = "Laminar"
        else:
            old_nu = nusselt_number_turbulent_pipe_OLD(re, pr)
            new_nu = nusselt_number_turbulent_pipe_NEW(re, pr)
            regime = "Turbulent" if re > 10000 else "Transition"

        diff = abs(old_nu - new_nu) / old_nu * 100 if old_nu != 0 else 0

        print(f"  {re:8d} | {old_nu:8.2f} | {new_nu:8.2f} | {diff:6.2f}% | {regime}")

    return True


def run_all_validations():
    """Run complete validation suite"""
    print("\n" + "#"*80)
    print("# STAGE 3: HEAT TRANSFER LIBRARY REPLACEMENT VALIDATION")
    print("# Comparing OLD (manual) vs NEW (fluids/ht library) implementations")
    print("#"*80)

    all_results = []

    # Run all validation tests
    all_results.extend(validate_prandtl())
    all_results.extend(validate_nusselt_laminar())
    all_results.extend(validate_nusselt_turbulent())
    all_results.extend(validate_graetz())
    validate_cdu_conditions()

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    total = len(all_results)
    passed = sum(1 for r in all_results if r['pass'])
    failed = total - passed

    print(f"\nTotal tests:  {total}")
    print(f"Passed:       {passed} [PASS]")
    print(f"Failed:       {failed} [FAIL]")

    if failed > 0:
        print("\nFailed tests:")
        for r in all_results:
            if not r['pass']:
                print(f"  - {r['test']}: {r['diff_pct']:.2f}% difference")

    print("\n" + "="*80)
    print("REPLACEMENT SUMMARY:")
    print("  1. Prandtl:     OLD manual -> NEW fluids.Prandtl()")
    print("  2. Graetz:      Kept simple formula (identical)")
    print("  3. Nu laminar:  OLD manual -> NEW ht.Nu_conv_internal('Laminar - constant Q')")
    print("  4. Nu turb:     OLD Gnielinski -> NEW ht.Nu_conv_internal('Gnielinski')")
    print("="*80)

    return passed == total


if __name__ == "__main__":
    success = run_all_validations()
    exit(0 if success else 1)
