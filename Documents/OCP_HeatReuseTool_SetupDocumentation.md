
# Heat Reuse Tool - Setup Guide

## Google Colab

### Complete Setup
1. **Go to Colab**: [https://colab.research.google.com](https://colab.research.google.com)
   - Sign in with your Google account
   - Click **File → New notebook**

2. **Copy this code** into the first cell and run it (Shift+Enter):
```python

# Complete Heat Reuse Tool setup and run
!git clone https://github.com/ahliana/OCP-CE-HR-Economics-Tool.git
%cd OCP-CE-HR-Economics-Tool
!pip install -q pandas numpy matplotlib ipywidgets
from google.colab import output
output.enable_custom_widget_manager()

# Now run the actual tool
import sys, os
sys.path.insert(0, f"{os.getcwd()}/python")
import autostart


```

3. **Use the tool** - you'll see 4 dropdowns and a Calculate button. Select values and click Calculate!

### Notes
- Tool runs in Google's cloud
- Results are temporary - download before closing


---


## VSCode

### Complete Setup
1. **Install Python**: Go to [python.org/downloads](https://python.org/downloads/)
   - Download Python 3.8 or newer
   - **Windows**: Check "Add Python to PATH" during install
   - **Test**: Open PowerShell, type `python --version`

2. **Install VSCode**: Go to [code.visualstudio.com](https://code.visualstudio.com/)
   - Download and install with default settings
   - Launch VSCode

3. **Install VSCode extensions**:
   - Open VSCode
   - Go to Extensions (left sidebar, square icon)
   - Search and install: "Python" (by Microsoft)
   - Search and install: "Jupyter" (by Microsoft)

4. **Fix PowerShell execution policy** (Windows only):
   - Open VSCode terminal: View → Terminal (this should open a PowerShell Terminal at the bottom of VSCode)
   - Run this command: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
   - This allows virtual environment activation

5. **Install Git**
- Go to: https://git-scm.com/download/win                       
- Install with default settings
- Test: Open VS Code terminal and type git --version
- 
- Add Git to PATH for current session
```powershell

$env:PATH += ";C:\Program Files\Git\cmd;C:\Program Files\Git\bin"
[Environment]::SetEnvironmentVariable("PATH", $env:PATH + ";C:\Program Files\Git\cmd;C:\Program Files\Git\bin", "User")

```


6. **Download the project**:
   - If VSCode terminal is not still open, open it: View → Terminal (this should open a PowerShell Terminal at the bottom of VSCode)
   - Create a directory  `md C:\Code`
   - Navigate to the directory where you want the project  `cd C:\Code`
   - Run: `git clone https://github.com/ahliana/OCP-CE-HR-Economics-Tool.git`


7. **Open project in VSCode**:
   - In VSCode: File → Open Folder
   - Select the `OCP-CE-HR-Economics-Tool` folder
   - You should see all files in the left Explorer panel

8. **Create virtual environment**:
   - Open VSCode terminal: View → Terminal (this should open a PowerShell Terminal at the bottom of VSCode)
   - Run these commands:
```powershell

##### WINDOWS #####
## PowerShell - Create virtual environment
python -m venv .venv
## Activate it
.\.venv\Scripts\Activate.ps1
---

##### MAC/LINUX #####
## PowerShell - Create virtual environment
python3 -m venv .venv
## Activate it
source .venv/bin/activate

# You should see (.venv) at the left in your terminal prompt

```

9. **Install packages and register Jupyter kernel**:
   - Open VSCode terminal: View → Terminal (this should open a PowerShell Terminal at the bottom of VSCode)
```powershell

# Windows PowerShell - Install packages
pip install -r requirements.txt

# Register Jupyter kernel (CRITICAL STEP!)
python -m ipykernel install --user --name=heat-reuse-tool --display-name="Heat Reuse Tool"

```

10. **Configure VSCode Python interpreter**:
   - After creating the `.venv`, VSCode may show a popup: "We noticed a new environment has been created. Do you want to select it for the workspace folder?"
   - **Click "Yes"** - this automatically configures VSCode to use your virtual environment
   - If you need to do it manually:
     1. Press `Ctrl+Shift+P`
     2. Type: "Python: Select Interpreter"
     3. Choose the interpreter from your `.venv` folder: `.\venv\Scripts\python.exe`

11. **Open and run the notebook**:
    - Click on `Interactive Analysis Tool.ipynb` in Explorer
    - **CRITICAL**: When VSCode shows "Select Kernel" dialog at the top:
      - Click "Python Environments..." (NOT "Existing Jupyter Server")
      - Look for your virtual environment with `.venv` in the path
      - Example: `Python 3.13.2 ('.venv': venv) C:\...\OCP_HeatReuseTool\.venv\Scripts\python.exe`
      - Click on the `.venv` option
    - **Verify kernel selection**: Look at top-right corner - should show "Python 3.13.2 (.venv)"
    - Run the cells to see your interface

12. **Test the setup**:
    - In VSCode terminal with virtual environment activated: `(.venv)` should be visible
    - Run: `python tools/setup/verify_setup.py`
    - **Success**: Should show "9/9 checks passed"

### Notes
- Uses PowerShell terminal in VSCode on Windows
- Jupyter kernel registration is essential for proper notebook functionality
- Select correct Python interpreter (.venv) when prompted

### Troubleshooting
**If virtual environment activation fails:**
```powershell

# Fix PowerShell execution policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then try activation again
.\.venv\Scripts\Activate.ps1

```

**If kernel not showing in notebook:**
- Ensure you ran the ipykernel install command
- Restart VSCode
- Check that you selected the .venv interpreter

**If packages not found:**
- Verify virtual environment is activated ((.venv) in prompt)
- Verify you selected the correct kernel in the notebook
- Re-run: `pip install -r requirements.txt`

---
