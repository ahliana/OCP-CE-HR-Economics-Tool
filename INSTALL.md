# Installation Guide

Complete installation instructions for the OCP Heat Reuse Economics Tool.

This guide covers all platforms and installation methods. Choose the section
that matches your situation.


## Table of Contents

1. [Choose Your Installation Method](#choose-your-installation-method)
2. [Prerequisites](#prerequisites)
3. [Installation Steps](#installation-steps)
4. [Verification](#verification)
5. [Running the Tool](#running-the-tool)
6. [Troubleshooting](#troubleshooting)


## Choose Your Installation Method

```
+------------------------------------------------------------------+
|                    Installation Decision Guide                    |
+------------------------------------------------------------------+
|                                                                  |
|  Do you need to save results permanently?                        |
|       |                                                          |
|       +-- No --> Use Google Colab (no installation)              |
|       |          See: docs/SETUP_COLAB.md                        |
|       |                                                          |
|       +-- Yes --> Local installation required                    |
|                   Continue reading below                         |
|                                                                  |
+------------------------------------------------------------------+
|                                                                  |
|  What is your operating system?                                  |
|       |                                                          |
|       +-- Windows --> docs/SETUP_WINDOWS.md                      |
|       +-- macOS ----> docs/SETUP_MAC.md                          |
|       +-- Linux ----> docs/SETUP_LINUX.md                        |
|                                                                  |
+------------------------------------------------------------------+
```

**Quick Recommendation:**

| Your Situation | Recommended Method |
|----------------|-------------------|
| Just want to try it | Google Colab |
| Need to modify pricing data | Local installation |
| Will use regularly | Local installation |
| No admin rights on computer | Google Colab |
| Behind corporate firewall | Check with IT, then local |


## Prerequisites

### For Google Colab

- Google account
- Web browser
- Internet connection

No software installation required.


### For Local Installation

**Required:**

| Component | Minimum Version | How to Check |
|-----------|-----------------|--------------|
| Python | 3.10 | `python --version` |
| pip | 20.0 | `pip --version` |
| Git | 2.0 | `git --version` |

**Recommended:**

| Component | Purpose |
|-----------|---------|
| VS Code | Code editor with Jupyter support |
| 8 GB RAM | Comfortable performance |


### Installing Prerequisites

**Python:**

Download from https://www.python.org/downloads/

During installation:
- Windows: Check "Add Python to PATH"
- macOS: Use the installer package
- Linux: Often pre-installed; use package manager if needed

**Git:**

Download from https://git-scm.com/downloads

Default installation options are acceptable for all platforms.


## Installation Steps

### Step 1: Download the Project

Open a terminal (PowerShell on Windows, Terminal on Mac/Linux):

```bash
git clone https://github.com/opencomputeproject/OCP-CE-HR-Economics-Tool.git
cd OCP-CE-HR-Economics-Tool
```

Alternative: Download ZIP from GitHub and extract.


### Step 2: Create Virtual Environment

A virtual environment keeps this project's dependencies separate from other
Python projects on your system.

```bash
python -m venv .venv
```

If `python` is not recognized, try `python3`.


### Step 3: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

If you get an execution policy error:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activation again.

**Windows (Command Prompt):**
```cmd
.\.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Verification:** Your prompt should now show `(.venv)` at the beginning.


### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Jupyter notebook environment
- pandas, numpy (data analysis)
- matplotlib (visualization)
- ipywidgets (interactive interface)

For advanced physics calculations (optional):
```bash
pip install -r requirements-full.txt
```


### Step 5: Register Jupyter Kernel

This step ensures Jupyter can find your virtual environment:

```bash
python -m ipykernel install --user --name=heat-reuse-tool --display-name="Heat Reuse Tool"
```


### Step 6: Start Jupyter

```bash
jupyter notebook
```

Your web browser will open automatically. If not, look for a URL in the
terminal output (starts with `http://localhost:8888`).


## Verification

Run the verification script to check your setup:

```bash
python tools/setup/verify_setup.py
```

Expected output:
```
Heat Reuse Tool - Setup Verification
=====================================

[PASS] Python version: 3.11.0
[PASS] Virtual environment active
[PASS] Required packages installed
[PASS] Jupyter kernel registered
[PASS] Project structure valid
[PASS] CSV data files present
[PASS] Autostart module loads

Result: 7/7 checks passed
```

If any checks fail, see [Troubleshooting](#troubleshooting).


## Running the Tool

### First Time

1. Start Jupyter: `jupyter notebook`
2. Open `Interactive Analysis Tool.ipynb`
3. Select kernel: "Heat Reuse Tool" or your `.venv` Python
4. Run the cell: Shift+Enter or "Run All" from menu

### Subsequent Uses

1. Open terminal in project directory
2. Activate environment: `.\.venv\Scripts\Activate.ps1` (Windows) or
   `source .venv/bin/activate` (Mac/Linux)
3. Start Jupyter: `jupyter notebook`
4. Open notebook and run


## Troubleshooting

### Common Issues

**"python is not recognized"**

Python is not in your system PATH.
- Windows: Reinstall Python, checking "Add to PATH"
- Mac/Linux: Try `python3` instead of `python`

**"pip is not recognized"**

- Try: `python -m pip` instead of `pip`

**Execution policy error (Windows PowerShell)**

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**"No module named X"**

Virtual environment may not be activated, or packages not installed.
1. Check for `(.venv)` in your prompt
2. Run: `pip install -r requirements.txt`

**Kernel not found in Jupyter**

Run the kernel registration again:
```bash
python -m ipykernel install --user --name=heat-reuse-tool --display-name="Heat Reuse Tool"
```

**Widgets not displaying**

For classic Jupyter Notebook:
```bash
jupyter nbextension enable --py widgetsnbextension
```

For JupyterLab:
```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```


### Getting Help

1. Check [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for detailed solutions
2. Run `python tools/setup/verify_setup.py` to diagnose issues
3. Open an issue: https://github.com/opencomputeproject/OCP-CE-HR-Economics-Tool/issues


## Next Steps

After successful installation:

1. Open `Interactive Analysis Tool.ipynb`
2. Run the cell to load the interface
3. Select input parameters using the dropdowns
4. Click Calculate to see results

See [docs/UI_CALCULATION_MAP.md](docs/UI_CALCULATION_MAP.md) for technical
details on the calculations.


---

For platform-specific detailed guides:
- [Windows Setup](docs/SETUP_WINDOWS.md)
- [Mac Setup](docs/SETUP_MAC.md)
- [Linux Setup](docs/SETUP_LINUX.md)
- [Google Colab](docs/SETUP_COLAB.md)
