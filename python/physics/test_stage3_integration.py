"""
Test Stage 3 integration - verify heat_transfer module works with new libraries
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the updated heat_transfer module
import heat_transfer

def test_imports():
    """Test that all imports work"""
    print("Testing imports...")
    print(f"  fluids: {heat_transfer.fluids}")
    print(f"  ht: {heat_transfer.ht}")
    print("  ✓ All imports successful")

def test_prandtl():
    """Test Prandtl number calculation"""
    print("\nTesting Prandtl number...")

    # Water at 20°C
    cp = 4182.0  # J/kg·K
    mu = 0.001002  # Pa·s
    k = 0.598  # W/m·K

    pr = heat_transfer.prandtl_number(cp, mu, k)
    print(f"  Pr = {pr:.4f} (expected ~7.0)")
    assert 6.5 < pr < 7.5, f"Prandtl number out of range: {pr}"
    print("  ✓ Prandtl test passed")

def test_nusselt_laminar():
    """Test laminar Nusselt number"""
    print("\nTesting laminar Nusselt...")

    re = 1000
    pr = 6.0

    nu = heat_transfer.nusselt_number_laminar_pipe(re, pr)
    print(f"  Nu (laminar, Re={re}) = {nu:.4f} (expected ~4.36)")
    assert 4.0 < nu < 5.0, f"Laminar Nu out of range: {nu}"
    print("  ✓ Laminar Nusselt test passed")

def test_nusselt_turbulent():
    """Test turbulent Nusselt number"""
    print("\nTesting turbulent Nusselt (Gnielinski)...")

    re = 10000
    pr = 6.0

    nu = heat_transfer.nusselt_number_turbulent_pipe(re, pr)
    print(f"  Nu (turbulent, Re={re}) = {nu:.4f} (expected ~74)")
    assert 60 < nu < 90, f"Turbulent Nu out of range: {nu}"
    print("  ✓ Turbulent Nusselt test passed")

def test_universal():
    """Test universal Nusselt number function"""
    print("\nTesting universal Nusselt function...")

    # Laminar case
    re_lam = 2000
    pr = 6.0
    nu_lam = heat_transfer.nusselt_number_pipe_universal(re_lam, pr)
    print(f"  Nu (universal, Re={re_lam}) = {nu_lam:.4f} (laminar)")

    # Turbulent case
    re_turb = 50000
    nu_turb = heat_transfer.nusselt_number_pipe_universal(re_turb, pr)
    print(f"  Nu (universal, Re={re_turb}) = {nu_turb:.4f} (turbulent)")

    assert nu_turb > nu_lam, "Turbulent Nu should be > laminar Nu"
    print("  ✓ Universal function test passed")

def test_heat_transfer_coefficient():
    """Test heat transfer coefficient calculation"""
    print("\nTesting heat transfer coefficient...")

    nu = 75.0
    k = 0.6  # W/m·K
    D = 0.1  # m

    h = heat_transfer.heat_transfer_coefficient(nu, k, D)
    print(f"  h = {h:.1f} W/m²·K (expected 450)")
    assert 400 < h < 500, f"Heat transfer coeff out of range: {h}"
    print("  ✓ Heat transfer coefficient test passed")

def test_graetz():
    """Test Graetz number"""
    print("\nTesting Graetz number...")

    re = 10000
    pr = 6.0
    d_over_l = 0.1

    gz = heat_transfer.graetz_number(re, pr, d_over_l)
    print(f"  Gz = {gz:.1f} (expected 6000)")
    assert abs(gz - 6000) < 1, f"Graetz number incorrect: {gz}"
    print("  ✓ Graetz number test passed")

def run_all_tests():
    """Run all integration tests"""
    print("="*80)
    print("STAGE 3 INTEGRATION TESTS - Heat Transfer Module")
    print("="*80)

    try:
        test_imports()
        test_prandtl()
        test_nusselt_laminar()
        test_nusselt_turbulent()
        test_universal()
        test_heat_transfer_coefficient()
        test_graetz()

        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nStage 3 heat transfer library replacement successful!")
        print("- Prandtl: fluids.Prandtl() ✓")
        print("- Nusselt laminar: ht.Nu_conv_internal('Laminar - constant Q') ✓")
        print("- Nusselt turbulent: ht.Nu_conv_internal('Gnielinski') ✓")
        print("- All functions working correctly ✓")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
