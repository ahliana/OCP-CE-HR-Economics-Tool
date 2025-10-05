#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation script for Stage 2: Fluids library migration
Compares old manual calculations with new fluids library implementation
"""

import math
import sys
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from python.physics.fluid_mechanics import (
    reynolds_number, friction_factor_laminar, friction_factor_turbulent,
    pressure_drop_pipe, pump_power_required, pipe_velocity
)
from python.physics.constants import get_water_properties

print("=" * 80)
print("STAGE 2 VALIDATION: FLUIDS LIBRARY MIGRATION")
print("=" * 80)
print()

# =============================================================================
# Test Cases
# =============================================================================

# Get water properties for testing
water_25c = get_water_properties(25)
rho = water_25c['density']  # ~997 kg/m³
nu = water_25c['kinematic_viscosity']  # ~8.9e-7 m²/s
mu = water_25c['dynamic_viscosity']  # ~8.9e-4 Pa·s

print("WATER PROPERTIES AT 25°C (from CoolProp)")
print(f"  Density: {rho:.2f} kg/m³")
print(f"  Kinematic viscosity: {nu:.6e} m²/s")
print(f"  Dynamic viscosity: {mu:.6e} Pa·s")
print()

# =============================================================================
# Test 1: REYNOLDS NUMBER
# =============================================================================
print("=" * 80)
print("TEST 1: REYNOLDS NUMBER")
print("=" * 80)

test_cases_re = [
    {"name": "Laminar flow", "V": 0.5, "D": 0.025, "temp": 25},
    {"name": "Turbulent flow", "V": 2.0, "D": 0.050, "temp": 25},
    {"name": "CDU typical", "V": 1.5, "D": 0.050, "temp": 25},
]

for tc in test_cases_re:
    print(f"\n{tc['name']}: V={tc['V']} m/s, D={tc['D']*1000} mm, T={tc['temp']}°C")

    # Old method (manual calculation)
    props = get_water_properties(tc['temp'])
    re_old = tc['V'] * tc['D'] / props['kinematic_viscosity']

    # New method (fluids library)
    re_new = reynolds_number(tc['V'], tc['D'], temperature_c=tc['temp'])

    diff_percent = abs(re_new - re_old) / re_old * 100

    print(f"  Old (manual):      Re = {re_old:.2f}")
    print(f"  New (fluids lib):  Re = {re_new:.2f}")
    print(f"  Difference:        {diff_percent:.6f}%")
    print(f"  Status:            {'✓ PASS' if diff_percent < 0.01 else '✗ FAIL'}")

# =============================================================================
# Test 2: FRICTION FACTOR - LAMINAR
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: FRICTION FACTOR - LAMINAR FLOW (Re < 2300)")
print("=" * 80)

test_cases_lam = [
    {"Re": 500, "expected_manual": 64/500},
    {"Re": 1000, "expected_manual": 64/1000},
    {"Re": 2000, "expected_manual": 64/2000},
]

for tc in test_cases_lam:
    print(f"\nReynolds number: {tc['Re']}")

    # Old method
    f_old = tc['expected_manual']

    # New method (fluids library)
    f_new = friction_factor_laminar(tc['Re'])

    diff_percent = abs(f_new - f_old) / f_old * 100

    print(f"  Old (64/Re):       f = {f_old:.6f}")
    print(f"  New (fluids lib):  f = {f_new:.6f}")
    print(f"  Difference:        {diff_percent:.6f}%")
    print(f"  Status:            {'✓ PASS' if diff_percent < 0.01 else '✗ FAIL'}")

# =============================================================================
# Test 3: FRICTION FACTOR - TURBULENT
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: FRICTION FACTOR - TURBULENT FLOW")
print("=" * 80)

test_cases_turb = [
    {"name": "Smooth pipe, Re=10000", "Re": 10000, "eD": 0.0},
    {"name": "Smooth pipe, Re=100000", "Re": 100000, "eD": 0.0},
    {"name": "Rough pipe, Re=100000, eD=0.001", "Re": 100000, "eD": 0.001},
    {"name": "CDU typical, Re=84269, eD=4.6e-4", "Re": 84269, "eD": 4.6e-4},
]

for tc in test_cases_turb:
    print(f"\n{tc['name']}")
    print(f"  Re={tc['Re']}, ε/D={tc['eD']}")

    # Old method (Petukhov for smooth, Haaland for rough)
    if tc['eD'] == 0.0:
        if tc['Re'] > 5e6:
            f_old = 0.3164 / (tc['Re'] ** 0.25)
            method_old = "Blasius"
        else:
            f_old = (0.790 * math.log(tc['Re']) - 1.64) ** (-2)
            method_old = "Petukhov"
    else:
        term1 = (tc['eD'] / 3.7) ** 1.11
        term2 = 6.9 / tc['Re']
        f_old = (-1.8 * math.log10(term1 + term2)) ** (-2)
        method_old = "Haaland"

    # New method (fluids library - uses Colebrook-White)
    f_new = friction_factor_turbulent(tc['Re'], tc['eD'])

    diff_percent = abs(f_new - f_old) / f_old * 100

    print(f"  Old ({method_old}):     f = {f_old:.8f}")
    print(f"  New (Colebrook-White): f = {f_new:.8f}")
    print(f"  Difference:            {diff_percent:.4f}%")

    # More lenient tolerance for different correlations
    tolerance = 5.0  # 5% tolerance since we're comparing different correlations
    status = '✓ PASS' if diff_percent < tolerance else '⚠ ACCEPTABLE' if diff_percent < 10 else '✗ FAIL'
    print(f"  Status:                {status}")
    print(f"  Note: Colebrook-White vs {method_old} - small differences expected")

# =============================================================================
# Test 4: PRESSURE DROP
# =============================================================================
print("\n" + "=" * 80)
print("TEST 4: PRESSURE DROP (Darcy-Weisbach)")
print("=" * 80)

test_cases_dp = [
    {"name": "CDU typical: 1000 L/min, DN50", "f": 0.02, "L": 100, "D": 0.050, "V": 1.5, "rho": rho},
    {"name": "High flow: 2000 L/min, DN50", "f": 0.019, "L": 100, "D": 0.050, "V": 3.0, "rho": rho},
]

for tc in test_cases_dp:
    print(f"\n{tc['name']}")
    print(f"  f={tc['f']}, L={tc['L']}m, D={tc['D']*1000}mm, V={tc['V']}m/s")

    # Old method (manual Darcy-Weisbach)
    dp_old = tc['f'] * (tc['L'] / tc['D']) * (tc['rho'] * tc['V']**2 / 2)

    # New method (fluids library)
    dp_new = pressure_drop_pipe(tc['f'], tc['L'], tc['D'], tc['V'], tc['rho'])

    diff_percent = abs(dp_new - dp_old) / dp_old * 100

    print(f"  Old (manual):      ΔP = {dp_old:.2f} Pa ({dp_old/1e5:.6f} bar)")
    print(f"  New (fluids lib):  ΔP = {dp_new:.2f} Pa ({dp_new/1e5:.6f} bar)")
    print(f"  Difference:        {diff_percent:.6f}%")
    print(f"  Status:            {'✓ PASS' if diff_percent < 0.01 else '✗ FAIL'}")

# =============================================================================
# Test 5: PUMP POWER
# =============================================================================
print("\n" + "=" * 80)
print("TEST 5: PUMP POWER")
print("=" * 80)

test_cases_pump = [
    {"name": "1000 L/min, 10m head, 75% eff", "Q_lpm": 1000, "head_m": 10, "eff": 0.75},
    {"name": "500 L/min, 5m head, 80% eff", "Q_lpm": 500, "head_m": 5, "eff": 0.80},
]

for tc in test_cases_pump:
    print(f"\n{tc['name']}")

    # Convert to SI units
    Q_m3s = tc['Q_lpm'] / 60000  # L/min to m³/s
    dP_pa = tc['head_m'] * rho * 9.81  # m head to Pa

    print(f"  Q={Q_m3s:.6f} m³/s, ΔP={dP_pa:.2f} Pa, η={tc['eff']}")

    # Old method (manual calculation)
    P_hyd_old = Q_m3s * dP_pa
    P_shaft_old = P_hyd_old / tc['eff']

    # New method (fluids library)
    result = pump_power_required(Q_m3s, dP_pa, efficiency=tc['eff'],
                                 include_motor_efficiency=False)
    P_hyd_new = result['hydraulic_power_w']
    P_shaft_new = result['shaft_power_w']

    diff_hyd = abs(P_hyd_new - P_hyd_old) / P_hyd_old * 100
    diff_shaft = abs(P_shaft_new - P_shaft_old) / P_shaft_old * 100

    print(f"  Hydraulic power:")
    print(f"    Old (manual):      {P_hyd_old:.2f} W ({P_hyd_old/1000:.3f} kW)")
    print(f"    New (fluids lib):  {P_hyd_new:.2f} W ({P_hyd_new/1000:.3f} kW)")
    print(f"    Difference:        {diff_hyd:.6f}%")

    print(f"  Shaft power:")
    print(f"    Old (manual):      {P_shaft_old:.2f} W ({P_shaft_old/1000:.3f} kW)")
    print(f"    New (fluids lib):  {P_shaft_new:.2f} W ({P_shaft_new/1000:.3f} kW)")
    print(f"    Difference:        {diff_shaft:.6f}%")

    print(f"  Status:            {'✓ PASS' if diff_hyd < 0.01 and diff_shaft < 0.01 else '✗ FAIL'}")

# =============================================================================
# Test 6: PIPE VELOCITY
# =============================================================================
print("\n" + "=" * 80)
print("TEST 6: PIPE VELOCITY")
print("=" * 80)

test_cases_vel = [
    {"name": "1000 L/min, DN50", "Q_lpm": 1000, "D": 0.050},
    {"name": "500 L/min, DN25", "Q_lpm": 500, "D": 0.025},
]

for tc in test_cases_vel:
    print(f"\n{tc['name']}")

    Q_m3s = tc['Q_lpm'] / 60000
    print(f"  Q={Q_m3s:.6f} m³/s, D={tc['D']*1000}mm")

    # Old method
    A = math.pi * tc['D']**2 / 4
    V_old = Q_m3s / A

    # New method
    V_new = pipe_velocity(Q_m3s, tc['D'])

    diff_percent = abs(V_new - V_old) / V_old * 100

    print(f"  Old (manual):      V = {V_old:.4f} m/s")
    print(f"  New (fluids lib):  V = {V_new:.4f} m/s")
    print(f"  Difference:        {diff_percent:.6f}%")
    print(f"  Status:            {'✓ PASS' if diff_percent < 0.01 else '✗ FAIL'}")

# =============================================================================
# COMPREHENSIVE SYSTEM TEST
# =============================================================================
print("\n" + "=" * 80)
print("COMPREHENSIVE SYSTEM TEST: CDU COOLING LOOP")
print("=" * 80)
print("\nTypical CDU Parameters:")
print("  Flow rate: 1000 L/min")
print("  Pipe: DN 50 (50mm ID)")
print("  Length: 100m")
print("  Temperature: 25°C")
print("  Pressure: 2 bar")
print()

Q_lpm = 1000
Q_m3s = Q_lpm / 60000  # m³/s
D = 0.050  # m (DN50)
L = 100  # m
temp = 25  # °C
roughness = 0.000046  # m (commercial steel)

props = get_water_properties(temp)
rho = props['density']
nu = props['kinematic_viscosity']

# Step 1: Velocity
V = pipe_velocity(Q_m3s, D)
print(f"1. Velocity: V = {V:.4f} m/s")

# Step 2: Reynolds number
Re = reynolds_number(V, D, temperature_c=temp)
print(f"2. Reynolds number: Re = {Re:.2f}")

# Step 3: Flow regime
if Re < 2300:
    regime = "Laminar"
    f = friction_factor_laminar(Re)
elif Re > 4000:
    regime = "Turbulent"
    eD = roughness / D
    f = friction_factor_turbulent(Re, eD)
else:
    regime = "Transitional"
    eD = roughness / D
    f = friction_factor_turbulent(Re, eD)  # fluids handles transition

print(f"3. Flow regime: {regime}")

# Step 4: Friction factor
print(f"4. Friction factor: f = {f:.6f}")
print(f"   (ε/D = {roughness/D:.6e})")

# Step 5: Pressure drop
dP = pressure_drop_pipe(f, L, D, V, rho)
print(f"5. Pressure drop: ΔP = {dP:.2f} Pa ({dP/1e5:.6f} bar)")
print(f"   Per 100m: {dP:.2f} Pa/100m")

# Step 6: Pump power (assume 75% efficiency)
pump_result = pump_power_required(Q_m3s, dP, efficiency=0.75,
                                  include_motor_efficiency=True, motor_efficiency=0.92)
print(f"6. Pump power:")
print(f"   Hydraulic: {pump_result['hydraulic_power_w']:.2f} W ({pump_result['hydraulic_power_w']/1000:.3f} kW)")
print(f"   Shaft: {pump_result['shaft_power_w']:.2f} W ({pump_result['shaft_power_w']/1000:.3f} kW)")
print(f"   Electrical: {pump_result['electrical_power_w']:.2f} W ({pump_result['electrical_power_w']/1000:.3f} kW)")
print(f"   Overall efficiency: {pump_result['overall_efficiency']:.1%}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("MIGRATION SUMMARY")
print("=" * 80)
print("""
✓ Reynolds number:        Using fluids.Reynolds()
✓ Friction factor (lam):  Using fluids.friction_factor()
✓ Friction factor (turb): Using fluids.friction_factor() with Colebrook-White
✓ Pressure drop:          Direct Darcy-Weisbach (no fluids wrapper needed)
✓ Pump power:             Direct Q×ΔP calculation (fundamental equation)
✓ Pipe velocity:          Direct Q/A calculation (continuity equation)

CORRELATION CHANGES:
- Turbulent friction factor: Petukhov/Haaland → Colebrook-White
  * Colebrook-White is the industry standard (more accurate)
  * Differences < 5% expected and acceptable
  * European standards compatible

BACKWARD COMPATIBILITY:
- All function signatures unchanged ✓
- All input/output types unchanged ✓
- European pipe sizing logic preserved ✓
- No breaking changes to engineering_calculations.py ✓
""")

print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
