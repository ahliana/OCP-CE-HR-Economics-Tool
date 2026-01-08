"""
Author: Ahliana Byrd <ahliana.byrd@gmail.com>
Created: 2025-10-08
"""

"""
Main UI Interface for Heat Reuse Tool
Simple init file that imports main display function
Updated 2026-01-07: Added styles module for light/dark mode compatibility
Updated 2026-01-08: Added advanced economic analysis module
"""

from .interface import display_interface, auto_initialize_interface
from .styles import inject_global_styles, COLORS
from .advanced_economics import SHOW_ADVANCED_ECONOMICS

# Make main functions available when importing from ui
__all__ = [
    'display_interface',
    'auto_initialize_interface',
    'inject_global_styles',
    'COLORS',
    'SHOW_ADVANCED_ECONOMICS'
]

__version__ = "1.2.0"  # Added advanced economic analysis