# OCP Heat Reuse Economics Tool

An interactive analysis tool for evaluating datacenter heat reuse systems.
Developed by the Open Compute Project (OCP) Community.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opencomputeproject/OCP-CE-HR-Economics-Tool/blob/master/Interactive%20Analysis%20Tool.ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


## Overview

This tool provides order-of-magnitude cost estimates for heat exchanger systems
used in datacenter heat reuse applications. It helps engineers, facility
planners, and sustainability teams evaluate the economics of capturing and
reusing waste heat from datacenters.

```
+------------------+     +-----------------+     +------------------+
|    Datacenter    | --> | Heat Exchanger  | --> | District Heating |
|   (Heat Source)  |     |     System      |     |   or Other Use   |
+------------------+     +-----------------+     +------------------+
        |                        |                       |
        v                        v                       v
   Server Cooling          Cost Analysis           Heat Recovery
   Water Circuit           Capital Costs           Economic Value
                          Operating Costs
```

### What This Tool Calculates

- Heat exchanger sizing and selection
- Capital cost estimates (equipment, piping, valves, instrumentation)
- Installation, engineering, and contingency factors
- Operating costs (pump energy consumption)
- Comparison across different approach temperatures (2C, 3C, 5C)


## Quick Start

Choose the option that best fits your situation:

```
                    How do you want to run the tool?
                                  |
            +---------------------+---------------------+
            |                     |                     |
            v                     v                     v
    [No Installation]      [Local Install]       [Developer]
     Google Colab          Your Computer         Contribute
            |                     |                     |
            v                     v                     v
      Click badge           Follow guide          Full setup
      Run the cell          15-20 minutes         Clone + venv
```

### Option 1: No Installation (Google Colab)

The fastest way to start. Requires only a Google account.

1. Click the "Open in Colab" badge above
2. Click "Run All" from the Runtime menu (or press Ctrl+F9)
3. Wait approximately 30 seconds for setup
4. Use the tool

No software installation required. Your session is temporary; download any
results before closing.


### Option 2: Local Installation

Run the tool on your own computer. Works offline after initial setup.

**Prerequisites:** Python 3.8 or newer

**Quick Install:**

```bash
# Download the project
git clone https://github.com/opencomputeproject/OCP-CE-HR-Economics-Tool.git
cd OCP-CE-HR-Economics-Tool

# Create isolated environment
python -m venv .venv

# Activate environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name=heat-reuse-tool --display-name="Heat Reuse Tool"

# Start Jupyter
jupyter notebook
```

Open `Interactive Analysis Tool.ipynb` and run the cell.

For detailed platform-specific instructions, see:
- [Windows Setup Guide](docs/SETUP_WINDOWS.md)
- [Mac Setup Guide](docs/SETUP_MAC.md)
- [Linux Setup Guide](docs/SETUP_LINUX.md)


### Option 3: Developer Setup

For contributors who want to modify the tool or submit improvements.

See [CONTRIBUTING.md](CONTRIBUTING.md) for complete instructions.


## Documentation

| Document | Description |
|----------|-------------|
| [INSTALL.md](INSTALL.md) | Comprehensive installation guide |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common issues and solutions |
| [docs/SETUP_COLAB.md](docs/SETUP_COLAB.md) | Google Colab detailed guide |
| [docs/UI_CALCULATION_MAP.md](docs/UI_CALCULATION_MAP.md) | Technical calculation details |
| [docs/Cost_API_Usage.md](docs/Cost_API_Usage.md) | Cost calculation API reference |


## Project Structure

```
OCP-CE-HR-Economics-Tool/
|
|-- Interactive Analysis Tool.ipynb    Main notebook (start here)
|
|-- python/                            Source code
|   |-- core/                          Cost calculations
|   |-- physics/                       Engineering calculations
|   |-- ui/                            User interface
|   +-- data/                          Data loading
|
|-- Data/                              Reference data (CSV files)
|   |-- ALLHX.csv                      Heat exchanger database
|   |-- PIPCOST.csv                    Piping costs
|   +-- ...                            Valves, fittings, etc.
|
|-- docs/                              Documentation
+-- tools/                             Setup utilities
```


## System Requirements

**Minimum:**
- Python 3.8+
- 4 GB RAM
- Modern web browser (for Jupyter interface)
- Internet connection (for initial setup only)

**Supported Platforms:**
- Windows 10/11
- macOS 10.14+
- Linux (Ubuntu 20.04+, Fedora, Debian)

**Supported Browsers (for Jupyter):**
- Chrome/Chromium
- Firefox
- Safari
- Edge


## Data Customization

The tool uses CSV files in the `Data/` directory for pricing and equipment
specifications. You can modify these files for your local conditions:

| File | Contents |
|------|----------|
| `ALLHX.csv` | Heat exchanger specifications and costs |
| `PIPCOST.csv` | Piping costs by material and size |
| `PIPSZ.csv` | Pipe sizing data |
| `CVALV.csv` | Control valve costs |
| `IVALV.csv` | Isolation valve costs |
| `JOINTS.csv` | Fitting and joint costs |

Keep backups before modifying. Changes affect only your local installation.


## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for:

- How to report issues
- How to suggest improvements
- How to submit changes
- Code style guidelines

All contributions are reviewed before inclusion in the official repository.


## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE)
for details.


## Acknowledgments

Developed by the Open Compute Project (OCP) Community.

**Contributors:**
- Ahliana Byrd (Primary Developer)
- OCP Heat Reuse Subproject Members

**References:**
- VDI Heat Atlas (Heat exchanger calculations)
- European district heating standards


## Support

- **Issues:** [GitHub Issues](https://github.com/opencomputeproject/OCP-CE-HR-Economics-Tool/issues)
- **Documentation:** See the `docs/` directory
- **OCP Community:** [Open Compute Project](https://www.opencompute.org/)

---

Open Compute Project | Heat Reuse Subproject
