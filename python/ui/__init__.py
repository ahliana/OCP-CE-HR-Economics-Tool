"""
Author: Ahliana Byrd <ahliana.byrd@gmail.com>
Created: 2025-10-08
"""

"""
Main UI Interface for Heat Reuse Tool
Simple init file that imports main display function
Updated 2026-01-07: Added styles module for light/dark mode compatibility
"""

from .interface import display_interface, auto_initialize_interface
from .styles import inject_global_styles, COLORS

# Make main functions available when importing from ui
__all__ = [
    'display_interface',
    'auto_initialize_interface',
    'inject_global_styles',
    'COLORS'
]

__version__ = "1.1.0"  # Updated for visual styling improvements