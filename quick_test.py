"""Quick validation test"""
import sys
sys.path.insert(0, 'python')

# Suppress all logging
import logging
logging.disable(logging.CRITICAL)

from data.loader import load_csv_files
from core.costs import calculate_order_of_magnitude_estimate
import warnings
warnings.filterwarnings('ignore')

load_csv_files()

print("COST MODULE VALIDATION")
print("="*80)

for approach in [2, 3, 5]:
    est = calculate_order_of_magnitude_estimate(1.0, 20, 10, approach)
    print(f"\nApproach {approach}C:")
    print(f"  HX:      EUR {est['heat_exchanger']:>10,.0f}")
    print(f"  Pumps:   EUR {est['pumps']:>10,.0f}")
    print(f"  Pipe+Fit:EUR {est['pipe_fittings']:>10,.0f}")
    print(f"  Instr:   EUR {est['instrumentation']:>10,.0f}")
    print(f"  TOTAL:   EUR {est['capital_total']:>10,.0f}")
    print(f"  Energy:      {est['operating_energy_kwh_year']:>10,.0f} kWh/yr")

print("\n" + "="*80)
