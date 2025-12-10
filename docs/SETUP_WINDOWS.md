# Windows Setup Guide

Step-by-step instructions for installing the Heat Reuse Economics Tool on
Windows 10 or Windows 11.

Estimated time: 15-20 minutes


## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Install Python](#install-python)
3. [Install Git](#install-git)
4. [Install VS Code](#install-vs-code-recommended)
5. [Download the Project](#download-the-project)
6. [Set Up Python Environment](#set-up-python-environment)
7. [Run the Tool](#run-the-tool)
8. [Troubleshooting](#troubleshooting)


## Prerequisites

Before starting, verify you have:

- Windows 10 or Windows 11
- Administrator access (for software installation)
- Internet connection
- Approximately 2 GB free disk space


## Install Python

### Step 1: Download Python

1. Go to https://www.python.org/downloads/
2. Click "Download Python 3.x.x" (latest version)
3. Save the installer

### Step 2: Run Installer

**Important:** Check these options during installation:

```
+--------------------------------------------------+
|  [x] Add Python to PATH    <-- CRITICAL          |
|  [x] Install for all users (recommended)         |
+--------------------------------------------------+
```

Click "Install Now" and wait for completion.

### Step 3: Verify Installation

Open PowerShell (press Windows key, type "powershell", press Enter):

```powershell
python --version
```

Expected output: `Python 3.11.x` (or similar)

If you see an error, Python was not added to PATH. Reinstall with the
checkbox selected.


## Install Git

### Step 1: Download Git

1. Go to https://git-scm.com/download/win
2. Download will start automatically
3. Run the installer

### Step 2: Installation Options

Accept default options for most screens. Key settings:

- Default editor: Your preference (Notepad is fine)
- PATH environment: "Git from the command line and also from 3rd-party software"
- Line endings: "Checkout Windows-style, commit Unix-style"

### Step 3: Verify Installation

Open a new PowerShell window:

```powershell
git --version
```

Expected output: `git version 2.x.x`


## Install VS Code (Recommended)

VS Code provides the best experience for running Jupyter notebooks on Windows.

### Step 1: Download VS Code

1. Go to https://code.visualstudio.com/
2. Click "Download for Windows"
3. Run the installer with default options

### Step 2: Install Extensions

Open VS Code, then:

1. Click the Extensions icon (left sidebar, looks like four squares)
2. Search and install:
   - "Python" (by Microsoft)
   - "Jupyter" (by Microsoft)

```
+----------------------------------------+
|  Extensions to Install                 |
+----------------------------------------+
|  [Install] Python       by Microsoft   |
|  [Install] Jupyter      by Microsoft   |
+----------------------------------------+
```


## Download the Project

### Option A: Using Git (Recommended)

Open PowerShell and run:

```powershell
# Navigate to where you want the project
cd C:\

# Create a directory for code projects (if it does not exist)
mkdir Code -ErrorAction SilentlyContinue
cd Code

# Download the project
git clone https://github.com/opencomputeproject/OCP-CE-HR-Economics-Tool.git

# Enter the project directory
cd OCP-CE-HR-Economics-Tool
```

### Option B: Download ZIP

1. Go to https://github.com/opencomputeproject/OCP-CE-HR-Economics-Tool
2. Click green "Code" button
3. Select "Download ZIP"
4. Extract to `C:\Code\OCP-CE-HR-Economics-Tool`


## Set Up Python Environment

### Step 1: Fix PowerShell Execution Policy

PowerShell restricts script execution by default. Run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Type `Y` and press Enter to confirm.

### Step 2: Create Virtual Environment

In PowerShell, from the project directory:

```powershell
python -m venv .venv
```

### Step 3: Activate Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

Your prompt should change to show `(.venv)`:

```
(.venv) PS C:\Code\OCP-CE-HR-Economics-Tool>
```

### Step 4: Install Dependencies

```powershell
pip install -r requirements.txt
```

Wait for installation to complete. This may take 2-3 minutes.

### Step 5: Register Jupyter Kernel

```powershell
python -m ipykernel install --user --name=heat-reuse-tool --display-name="Heat Reuse Tool"
```

### Step 6: Verify Setup

```powershell
python tools/setup/verify_setup.py
```

All checks should pass.


## Run the Tool

### Using VS Code (Recommended)

1. Open VS Code
2. File > Open Folder > Select `C:\Code\OCP-CE-HR-Economics-Tool`
3. Open `Interactive Analysis Tool.ipynb`
4. When prompted for kernel, select:
   - "Python Environments" > your `.venv` environment, OR
   - "Heat Reuse Tool" kernel
5. Click "Run All" or press Ctrl+F9

### Using Jupyter Notebook

From PowerShell (with virtual environment activated):

```powershell
jupyter notebook
```

Browser opens automatically. Click `Interactive Analysis Tool.ipynb` to open.


## Troubleshooting

### "python is not recognized"

Python was not added to PATH during installation.

**Fix:** Reinstall Python, ensuring "Add Python to PATH" is checked.

**Alternative:** Use full path:
```powershell
C:\Users\YourName\AppData\Local\Programs\Python\Python311\python.exe --version
```

### "Activate.ps1 cannot be loaded"

Execution policy is blocking the script.

**Fix:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### "No module named X"

Packages not installed or wrong environment.

**Fix:**
1. Verify `(.venv)` appears in your prompt
2. Run: `pip install -r requirements.txt`

### Widgets Not Displaying

Enable the widget extension:

```powershell
jupyter nbextension enable --py widgetsnbextension
```

### "Permission denied" Errors

Close any programs using the project files (including VS Code or other
notebooks) and try again.


## Quick Reference

Commands you will use regularly:

```powershell
# Navigate to project
cd C:\Code\OCP-CE-HR-Economics-Tool

# Activate environment
.\.venv\Scripts\Activate.ps1

# Start Jupyter
jupyter notebook

# Verify setup
python tools/setup/verify_setup.py

# Update packages (when needed)
pip install -r requirements.txt --upgrade
```


## Next Steps

- Review [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) if you encounter issues
- See [../README.md](../README.md) for usage instructions
- Check [UI_CALCULATION_MAP.md](UI_CALCULATION_MAP.md) for calculation details


---

Need help? Open an issue at:
https://github.com/opencomputeproject/OCP-CE-HR-Economics-Tool/issues
