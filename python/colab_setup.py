# python/colab_setup.py - Google Colab Environment Setup
"""
Author: Ahliana Byrd <ahliana.byrd@gmail.com>
Created: 2025-12-10
"""

"""
Handles all Colab-specific setup including:
- Repository cloning
- Dependency installation
- Widget manager configuration
- Interface display
"""

import sys
import os
import subprocess
import time

def setup_environment():
    """Configure Colab environment and install dependencies."""
    # Ensure we're in the right directory
    os.chdir('/content/OCP-CE-HR-Economics-Tool')

    # Enable widget manager FIRST
    from google.colab import output
    output.enable_custom_widget_manager()

    # Install dependencies quietly
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', '-r', 'requirements.txt'],
        capture_output=True
    )

    # Import ipywidgets to force initialization
    import ipywidgets

    # Give widget manager time to initialize
    time.sleep(1)

    return True

def launch_interface():
    """Launch the Heat Reuse Tool interface."""
    from IPython.display import display, clear_output

    # Add python path
    if '/content/OCP-CE-HR-Economics-Tool/python' not in sys.path:
        sys.path.insert(0, '/content/OCP-CE-HR-Economics-Tool/python')

    # Import UI components directly instead of autostart
    from ui.interface import display_logo, display_hxsimpledrawing, create_heat_reuse_interface

    # Display logo and drawing
    display_logo()
    display_hxsimpledrawing()

    # Create interface
    interface_components = create_heat_reuse_interface()

    # Explicitly display the widget
    if interface_components and 'interface' in interface_components:
        display(interface_components['interface'])

    return interface_components

def run():
    """Main entry point - setup and launch."""
    print("Setting up Heat Reuse Economics Tool...")
    setup_environment()
    print("Launching interface...")
    print("-" * 40)
    return launch_interface()
