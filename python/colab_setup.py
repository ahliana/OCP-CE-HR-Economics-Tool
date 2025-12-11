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

Usage in notebook cell:
    !pip install --quiet gitpython
    import sys
    sys.path.insert(0, '/content')
    !git clone --quiet https://github.com/opencomputeproject/OCP-CE-HR-Economics-Tool.git /content/OCP-CE-HR-Economics-Tool 2>/dev/null || true
    sys.path.insert(0, '/content/OCP-CE-HR-Economics-Tool/python')
    from colab_setup import run
    run()
"""

import sys
import os
import subprocess

def setup_environment():
    """Configure Colab environment and install dependencies."""
    # Ensure we're in the right directory
    os.chdir('/content/OCP-CE-HR-Economics-Tool')

    # Enable widget manager
    from google.colab import output
    output.enable_custom_widget_manager()

    # Install dependencies quietly
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', '-r', 'requirements.txt'],
        capture_output=True
    )

    # Import ipywidgets to ensure it's loaded
    import ipywidgets

    return True

def launch_interface():
    """Launch the Heat Reuse Tool interface."""
    # Add python path
    sys.path.insert(0, '/content/OCP-CE-HR-Economics-Tool/python')

    # Import and run
    import autostart
    return autostart

def run():
    """Main entry point - setup and launch."""
    print("Setting up Heat Reuse Economics Tool...")
    setup_environment()
    print("Launching interface...")
    print("-" * 40)
    return launch_interface()
