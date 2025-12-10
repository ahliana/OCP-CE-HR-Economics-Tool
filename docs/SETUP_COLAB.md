# Google Colab Setup Guide

Run the OCP Heat Reuse Economics Tool directly in your browser with no installation required.

**Time required:** Under 5 minutes

## What is Google Colab?

Google Colab (Colaboratory) is a free, cloud-based Jupyter notebook environment provided by Google. It runs entirely in your web browser and requires no software installation.

## Requirements

- A Google account (Gmail)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection

That's it. No Python, no Git, no downloads.

## Quick Start

1. Click this button:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opencomputeproject/OCP-CE-HR-Economics-Tool/blob/master/Interactive%20Analysis%20Tool.ipynb)

2. Sign in to your Google account if prompted

3. Click "Runtime" menu > "Run all" (or press `Ctrl+F9`)

4. Wait for all cells to execute (2-3 minutes the first time)

5. Start using the interactive tool

## Step-by-Step Instructions

### Step 1: Open the Notebook

Click the "Open in Colab" badge above, or:

1. Go to [colab.research.google.com](https://colab.research.google.com/)
2. Click "GitHub" tab
3. Enter: `opencomputeproject/OCP-CE-HR-Economics-Tool`
4. Select "Interactive Analysis Tool.ipynb"

### Step 2: Run the Notebook

1. Click "Runtime" in the menu bar
2. Select "Run all"
3. If prompted about the notebook not being authored by Google, click "Run anyway"
4. Wait for execution to complete (watch the spinning indicators next to each cell)

### Step 3: Use the Tool

Once all cells have run:
- Interactive sliders and controls will appear
- Adjust parameters to see calculations update
- Gauges and charts display results in real time

## Tips for Using Colab

### Save Your Work

Your changes are not saved automatically. To save:
- "File" > "Save a copy in Drive" - saves to your Google Drive
- "File" > "Download" > "Download .ipynb" - saves to your computer

### Session Timeouts

Free Colab sessions disconnect after:
- ~90 minutes of inactivity
- ~12 hours of total runtime

If disconnected, just reconnect and run all cells again.

### GPU Not Needed

This tool does not require GPU acceleration. The default CPU runtime is sufficient.

## Limitations vs Local Installation

| Feature | Google Colab | Local Install |
|---------|--------------|---------------|
| Setup time | Minutes | 15-30 minutes |
| Offline use | No | Yes |
| Session persistence | Limited | Unlimited |
| Large datasets | Limited memory | Full system resources |
| Customization | Limited | Full control |

## When to Use Local Installation Instead

Consider installing locally if you:
- Need to work offline
- Process very large datasets
- Run frequent or long analyses
- Want to customize the environment
- Need consistent, persistent sessions

See [INSTALL.md](../INSTALL.md) for local installation options.

## Troubleshooting

### Widgets Not Displaying

If sliders and interactive elements do not appear:
1. Make sure all cells have finished running
2. Try "Runtime" > "Restart and run all"
3. Clear browser cache and try again

### Session Disconnected

1. Click "Reconnect" in the top right
2. Run all cells again: "Runtime" > "Run all"

### "Module not found" Errors

The first code cell installs required packages. Make sure it runs before other cells:
1. Click the first code cell
2. Press `Shift+Enter` to run it
3. Wait for completion
4. Then run remaining cells

### Slow Performance

Free Colab resources are shared. If performance is poor:
- Try again later
- Consider local installation for heavy use

For more help, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
