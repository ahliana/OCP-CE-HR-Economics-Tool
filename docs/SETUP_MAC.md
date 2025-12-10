# macOS Setup Guide

Step-by-step instructions for installing the OCP Heat Reuse Economics Tool on macOS.

**Estimated time:** 15-20 minutes

## Prerequisites

- macOS 12 (Monterey) or later
- Administrator access
- Internet connection

## Step 1: Install Xcode Command Line Tools

This installs Git and other development tools. Open Terminal (Applications > Utilities > Terminal) and run:

```bash
xcode-select --install
```

A popup will appear. Click "Install" and wait for completion.

### Verify Git Installation

```bash
git --version
```

You should see `git version 2.x.x`.

## Step 2: Install Python

macOS comes with Python, but you should install a current version.

### Option A: Homebrew (Recommended)

If you do not have Homebrew installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Follow any instructions to add Homebrew to your PATH.

Then install Python:

```bash
brew install python@3.12
```

### Option B: Python.org

1. Go to [python.org/downloads/macos](https://www.python.org/downloads/macos/)
2. Download the latest Python 3.12.x installer
3. Open the downloaded `.pkg` file
4. Follow the installation wizard

### Verify Python Installation

```bash
python3 --version
```

You should see `Python 3.12.x` (or your installed version).

**Note:** On macOS, use `python3` and `pip3` instead of `python` and `pip`.

## Step 3: Install VS Code (Recommended)

Visual Studio Code makes working with Jupyter notebooks easier.

1. Go to [code.visualstudio.com](https://code.visualstudio.com/)
2. Click "Download for macOS"
3. Open the downloaded `.zip` file
4. Drag Visual Studio Code to your Applications folder

### Install Python Extension

1. Open VS Code
2. Click the Extensions icon (square icon on left sidebar) or press `Cmd+Shift+X`
3. Search for "Python"
4. Install "Python" by Microsoft
5. Search for "Jupyter"
6. Install "Jupyter" by Microsoft

## Step 4: Clone the Repository

Open Terminal and navigate to where you want the project:

```bash
cd ~/Documents
git clone https://github.com/opencomputeproject/OCP-CE-HR-Economics-Tool.git
cd OCP-CE-HR-Economics-Tool
```

## Step 5: Create Virtual Environment

A virtual environment keeps this project's packages separate from other Python projects:

```bash
python3 -m venv venv
```

### Activate the Virtual Environment

```bash
source venv/bin/activate
```

You should see `(venv)` at the beginning of your terminal prompt. This indicates the virtual environment is active.

**Note:** You must activate the virtual environment every time you open a new Terminal to work on this project.

## Step 6: Install Dependencies

With the virtual environment active:

```bash
pip install -r requirements.txt
```

This installs all required packages. It may take 5-10 minutes.

### If CoolProp Fails to Install

On Apple Silicon (M1/M2/M3) Macs, CoolProp may need to be built from source. If you see errors:

```bash
pip install --no-binary CoolProp CoolProp
```

If that still fails, you may need to install build tools:

```bash
brew install cmake
pip install --no-binary CoolProp CoolProp
```

## Step 7: Run the Notebook

### Option A: Using Jupyter in Browser

```bash
jupyter notebook "Interactive Analysis Tool.ipynb"
```

This opens the notebook in your default web browser.

### Option B: Using VS Code

1. Open VS Code
2. File > Open Folder > select `OCP-CE-HR-Economics-Tool`
3. Open `Interactive Analysis Tool.ipynb`
4. VS Code will ask to select a kernel - choose the `venv` Python interpreter
5. Click "Run All" at the top

## Step 8: Verify Everything Works

In the notebook:

1. Click "Cell" menu > "Run All" (or "Run All" button in VS Code)
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

Or open the folder in VS Code and open the notebook file.

## Troubleshooting

### "command not found: python3"
If using Homebrew Python, you may need to add it to your PATH. Add to `~/.zshrc`:
```bash
export PATH="/opt/homebrew/bin:$PATH"
```
Then restart Terminal or run `source ~/.zshrc`.

### Widgets Not Displaying
Enable Jupyter widgets:
```bash
jupyter nbextension enable --py widgetsnbextension
```

### Permission Denied Errors
If you get permission errors when installing packages:
```bash
pip install --user -r requirements.txt
```

For more solutions, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
