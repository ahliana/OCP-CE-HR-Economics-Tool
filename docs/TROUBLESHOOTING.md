# Troubleshooting Guide

Common issues and solutions for the OCP Heat Reuse Economics Tool.

## Quick Fixes

Before diving into specific errors, try these steps:

1. **Restart the kernel:** In Jupyter, click "Kernel" > "Restart & Run All"
2. **Check virtual environment:** Make sure you see `(venv)` in your terminal
3. **Update packages:** Run `pip install -r requirements.txt --upgrade`
4. **Check Python version:** Run `python --version` (must be 3.9+)

## Common Errors

### "pip is not recognized" / "pip: command not found"

**Cause:** pip is not in your system PATH, or Python was not installed correctly.

**Solutions:**

Windows:
```cmd
python -m pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m pip install -r requirements.txt
```

If that does not work, reinstall Python and ensure "Add to PATH" is checked during installation.

### "python is not recognized" / "python: command not found"

**Cause:** Python is not installed or not in your PATH.

**Solutions:**

Windows:
- Try `py` instead of `python`
- Reinstall Python from python.org, checking "Add Python to PATH"
- Or install from Microsoft Store

macOS:
- Use `python3` instead of `python`
- Install via Homebrew: `brew install python`

Linux:
- Use `python3` instead of `python`
- Install: `sudo apt install python3` (Ubuntu/Debian)

### "No kernel found" / "Kernel not found"

**Cause:** Jupyter cannot find the Python environment with required packages.

**Solutions:**

1. Install ipykernel in your virtual environment:
   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name=venv
   ```

2. In Jupyter, select the correct kernel:
   - Click "Kernel" > "Change kernel" > select "venv"

3. In VS Code:
   - Click the kernel selector in the top right
   - Select the Python interpreter from your venv folder

### Widgets Not Displaying / Blank Output

**Cause:** Jupyter widget extensions are not enabled.

**Solutions:**

1. Enable widget extensions:
   ```bash
   jupyter nbextension enable --py widgetsnbextension
   jupyter nbextension enable --py --sys-prefix widgetsnbextension
   ```

2. For JupyterLab:
   ```bash
   jupyter labextension install @jupyter-widgets/jupyterlab-manager
   ```

3. For VS Code:
   - Make sure the Jupyter extension is installed and updated
   - Try reloading the window: `Ctrl+Shift+P` > "Developer: Reload Window"

4. Clear browser cache and refresh the notebook

### "ModuleNotFoundError: No module named 'xyz'"

**Cause:** Required package is not installed, or you are not in the virtual environment.

**Solutions:**

1. Activate your virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

2. Install missing package:
   ```bash
   pip install xyz
   ```

3. Reinstall all requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. Make sure you are in the correct directory containing requirements.txt

### "Permission denied" Errors

**Windows:**
```cmd
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux:**
```bash
pip install --user -r requirements.txt
```

Or fix directory permissions:
```bash
sudo chown -R $USER:$USER ~/Documents/OCP-CE-HR-Economics-Tool
```

### CoolProp Installation Fails

CoolProp requires compilation on some systems.

**macOS (Apple Silicon M1/M2/M3):**
```bash
brew install cmake
pip install --no-binary CoolProp CoolProp
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt install cmake python3-dev build-essential

# Fedora
sudo dnf install cmake python3-devel gcc gcc-c++

# Then retry
pip install --no-binary CoolProp CoolProp
```

**Windows:**
- Install Visual Studio Build Tools from Microsoft
- Or use pre-built wheels: `pip install CoolProp`

### "externally-managed-environment" Error (Linux)

**Cause:** Modern Linux distributions protect system Python from pip modifications.

**Solution:** Always use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Jupyter Notebook Opens But Cells Do Not Run

**Cause:** Kernel connection issue or browser problem.

**Solutions:**

1. Check if Jupyter server is running in your terminal
2. Try a different browser
3. Clear browser cache
4. Restart Jupyter:
   - Close browser tab
   - Press `Ctrl+C` in terminal to stop Jupyter
   - Run `jupyter notebook` again

### Plots/Gauges Not Appearing

**Cause:** Matplotlib backend issue or display problem.

**Solutions:**

1. Add this at the start of the notebook:
   ```python
   %matplotlib inline
   ```

2. Install matplotlib backend:
   ```bash
   pip install matplotlib
   ```

3. For interactive plots:
   ```bash
   pip install ipympl
   ```
   And use `%matplotlib widget` in the notebook.

## Platform-Specific Issues

### Windows

**Long Path Names:**
Enable long paths in Windows:
1. Run `regedit` as Administrator
2. Navigate to `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
3. Set `LongPathsEnabled` to `1`
4. Restart computer

**Antivirus Blocking:**
Some antivirus software blocks Python or pip. Temporarily disable it during installation, or add Python to the exceptions list.

### macOS

**"Cannot be opened because the developer cannot be verified":**
```bash
xattr -d com.apple.quarantine /path/to/file
```

**Rosetta Issues (Apple Silicon):**
If you have issues with M1/M2/M3 Macs, ensure you are using native ARM Python:
```bash
arch -arm64 brew install python
```

### Linux

**Missing Display Server:**
If running on a headless server:
```bash
export DISPLAY=:0
# Or use Xvfb for virtual display
```

**SELinux Blocking:**
On Fedora/RHEL with SELinux:
```bash
sudo setenforce 0  # Temporary
# Or configure proper SELinux policies
```

## Getting More Help

If none of these solutions work:

1. **Check error messages carefully** - they often contain hints about the cause

2. **Search the error message** - copy the key part of the error and search online

3. **Open an issue** - Report problems at [GitHub Issues](https://github.com/opencomputeproject/OCP-CE-HR-Economics-Tool/issues) with:
   - Your operating system and version
   - Python version (`python --version`)
   - Complete error message
   - Steps you tried

4. **Try Google Colab** - If local installation continues to fail, use [Colab](SETUP_COLAB.md) as a workaround
