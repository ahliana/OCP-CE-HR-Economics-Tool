# Linux Setup Guide

Step-by-step instructions for installing the OCP Heat Reuse Economics Tool on Linux.

**Estimated time:** 10-15 minutes

## Prerequisites

- Ubuntu 20.04+, Fedora 35+, Debian 11+, or similar distribution
- sudo access
- Internet connection

## Step 1: Install Python and Git

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv git
```

### Fedora

```bash
sudo dnf install python3 python3-pip git
```

### Arch Linux

```bash
sudo pacman -S python python-pip git
```

### Verify Installation

```bash
python3 --version
git --version
```

You should see Python 3.9+ and Git 2.x.

## Step 2: Clone the Repository

Navigate to where you want the project:

```bash
cd ~/Documents
git clone https://github.com/opencomputeproject/OCP-CE-HR-Economics-Tool.git
cd OCP-CE-HR-Economics-Tool
```

## Step 3: Create Virtual Environment

A virtual environment keeps this project's packages separate from system packages:

```bash
python3 -m venv venv
```

### Activate the Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt. This indicates the virtual environment is active.

**Note:** You must activate the virtual environment every time you open a new terminal to work on this project.

## Step 4: Install Dependencies

With the virtual environment active:

```bash
pip install -r requirements.txt
```

This installs all required packages. It may take 5-10 minutes.

### If You See Build Errors

Some packages may need development headers. Install them first:

**Ubuntu / Debian:**
```bash
sudo apt install python3-dev build-essential
pip install -r requirements.txt
```

**Fedora:**
```bash
sudo dnf install python3-devel gcc gcc-c++
pip install -r requirements.txt
```

## Step 5: Run the Notebook

### Option A: Command Line Jupyter

```bash
jupyter notebook "Interactive Analysis Tool.ipynb"
```

This opens the notebook in your default web browser.

### Option B: VS Code

1. Install VS Code:
   - Download from [code.visualstudio.com](https://code.visualstudio.com/)
   - Or install via package manager:
     - Ubuntu: `sudo snap install code --classic`
     - Fedora: `sudo rpm --import https://packages.microsoft.com/keys/microsoft.asc && sudo dnf install code`

2. Open VS Code and install extensions:
   - "Python" by Microsoft
   - "Jupyter" by Microsoft

3. Open the project folder and the notebook file

### Option C: JupyterLab

If you prefer JupyterLab:

```bash
pip install jupyterlab
jupyter lab
```

## Step 6: Verify Everything Works

In the notebook:

1. Click "Cell" menu > "Run All"
2. Wait for all cells to execute
3. You should see interactive widgets and gauges appear
4. Try adjusting a slider - the outputs should update

If you encounter errors, see [Troubleshooting](TROUBLESHOOTING.md).

## Running the Tool Later

Each time you want to use the tool:

1. Open Terminal
2. Navigate to the project:
   ```bash
   cd ~/Documents/OCP-CE-HR-Economics-Tool
   ```
3. Activate virtual environment:
   ```bash
   source venv/bin/activate
   ```
4. Launch notebook:
   ```bash
   jupyter notebook "Interactive Analysis Tool.ipynb"
   ```

## Troubleshooting

### "externally-managed-environment" Error

On newer distributions (Ubuntu 23.04+, Fedora 38+), pip may refuse to install packages system-wide. This is why we use a virtual environment. Make sure the venv is activated before running pip.

### Widgets Not Displaying

Enable Jupyter widgets:
```bash
jupyter nbextension enable --py widgetsnbextension
```

### CoolProp Build Fails

Install build dependencies:
```bash
# Ubuntu/Debian
sudo apt install cmake

# Fedora
sudo dnf install cmake

# Then retry
pip install --no-binary CoolProp CoolProp
```

### Permission Denied on /dev/shm

Some Jupyter operations use shared memory. Ensure proper permissions:
```bash
sudo chmod 1777 /dev/shm
```

For more solutions, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
