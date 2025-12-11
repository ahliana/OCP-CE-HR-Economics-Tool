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
import time

def setup_environment():
    """Configure Colab environment and install dependencies."""
    # Ensure we're in the right directory
    os.chdir('/content/OCP-CE-HR-Economics-Tool')

    print("[DEBUG] Installing dependencies from requirements.txt...")

    # Use os.system for visible output in Colab
    os.system(f'{sys.executable} -m pip install ipywidgets>=7,<8')
    os.system(f'{sys.executable} -m pip install -r requirements.txt')

    # Check what got installed
    import ipywidgets
    print(f"[DEBUG] ipywidgets version: {ipywidgets.__version__}")

    try:
        import CoolProp
        print(f"[DEBUG] CoolProp version: {CoolProp.__version__}")
    except ImportError as e:
        print(f"[DEBUG] CoolProp import failed: {e}")

    # Enable widget manager after installs
    print("[DEBUG] Enabling custom widget manager...")
    from google.colab import output
    output.enable_custom_widget_manager()

    # Give widget manager time to initialize
    print("[DEBUG] Waiting for widget manager...")
    time.sleep(2)

    print("[DEBUG] Setup complete")
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

    # Temporarily disabled clear_output for debugging
    # from IPython.display import clear_output
    # clear_output(wait=True)

    print("[DEBUG] Launching interface...")
    return launch_interface()
