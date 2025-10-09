# python/core/__init__.py
"""
Author: Ahliana Byrd <ahliana.byrd@gmail.com>
Created: 2025-10-08
"""

"""
Core business logic and calculations for Heat Reuse Tool

This module contains the main calculation functions and business logic
for datacenter heat reuse system analysis.
"""

# Import main calculation functions and make them available at package level
from .original_calculations import (
    get_MW,
    get_MW_divd,
    quick_wha_calculation
)

# Import cost calculation functions
from .costs import (
    calculate_order_of_magnitude_estimate,
    calculate_operating_costs,
    compare_approaches,
    format_cost_summary
)

# Make functions available when importing from core
__all__ = [
    'get_MW',
    'get_MW_divd',
    'quick_wha_calculation',
    'calculate_order_of_magnitude_estimate',
    'calculate_operating_costs',
    'compare_approaches',
    'format_cost_summary'
]

__version__ = "1.0.0"