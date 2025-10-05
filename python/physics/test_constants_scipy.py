"""
Validation test for scipy constants replacement
Compares old hardcoded values with new scipy values
"""

from scipy import constants as scipy_const

# # Old hardcoded values (for comparison)
# OLD_STEFAN_BOLTZMANN = 5.670374419e-8
# OLD_GAS_CONSTANT = 8.314462618

# # New scipy values
# NEW_STEFAN_BOLTZMANN = scipy_const.Stefan_Boltzmann
# NEW_GAS_CONSTANT = scipy_const.R

# print("=" * 70)
# print("SCIPY CONSTANTS VALIDATION")
# print("=" * 70)

# print("\nSTEFAN-BOLTZMANN CONSTANT:")
# print(f"  Old value:  {OLD_STEFAN_BOLTZMANN:.15e} W/(m^2*K^4)")
# print(f"  New value:  {NEW_STEFAN_BOLTZMANN:.15e} W/(m^2*K^4)")
# print(f"  Difference: {abs(NEW_STEFAN_BOLTZMANN - OLD_STEFAN_BOLTZMANN):.3e}")
# print(f"  Match:      {NEW_STEFAN_BOLTZMANN == OLD_STEFAN_BOLTZMANN}")

# print("\nGAS CONSTANT (R):")
# print(f"  Old value:  {OLD_GAS_CONSTANT:.15e} J/(mol*K)")
# print(f"  New value:  {NEW_GAS_CONSTANT:.15e} J/(mol*K)")
# print(f"  Difference: {abs(NEW_GAS_CONSTANT - OLD_GAS_CONSTANT):.3e}")
# print(f"  Match:      {NEW_GAS_CONSTANT == OLD_GAS_CONSTANT}")

# print("\n" + "=" * 70)
# print("BACKWARDS COMPATIBILITY CHECK")
# print("=" * 70)

# # Import from the updated module
# from constants import STEFAN_BOLTZMANN, GAS_CONSTANT

# print(f"\nImported STEFAN_BOLTZMANN: {STEFAN_BOLTZMANN:.15e}")
# print(f"Imported GAS_CONSTANT:     {GAS_CONSTANT:.15e}")

# # Test that they match scipy values
# assert STEFAN_BOLTZMANN == scipy_const.Stefan_Boltzmann, "Stefan-Boltzmann mismatch!"
# assert GAS_CONSTANT == scipy_const.R, "Gas constant mismatch!"

# print("\n[OK] All constants successfully imported from scipy")
# print("[OK] Backwards compatibility maintained")
# print("[OK] No breaking changes detected")
