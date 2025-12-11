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

    # CRITICAL: Downgrade ipywidgets to 7.x for Colab compatibility
    # Colab's widget manager has issues with ipywidgets 8.x
    # See: https://github.com/googlecolab/colabtools/issues/3020
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', 'ipywidgets>=7,<8'],
        capture_output=True
    )

    # Install other dependencies quietly
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', '-r', 'requirements.txt'],
        capture_output=True
    )

    # Enable widget manager after downgrade
    from google.colab import output
    output.enable_custom_widget_manager()

    # Import ipywidgets to force initialization
    import ipywidgets

    # Give widget manager time to initialize
    time.sleep(1)

    return True

def launch_interface():
    """Launch the Heat Reuse Tool interface."""
    from IPython.display import display
    import ipywidgets as widgets

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

    # Use Output widget to force proper rendering
    if interface_components and 'interface' in interface_components:
        out = widgets.Output()
        with out:
            display(interface_components['interface'])
        display(out)

    return interface_components

def run():
    """Main entry point - setup and launch."""
    print("Setting up Heat Reuse Economics Tool...")
    setup_environment()

    # Clear output and display fresh
    from IPython.display import clear_output
    clear_output(wait=True)

    return launch_interface()
