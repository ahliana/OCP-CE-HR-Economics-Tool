"""
Test new distinct color scheme for pie charts
"""

import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'python'))

from python.data.loader import load_csv_files
load_csv_files()

from python.core.costs import compare_approaches
from python.ui.economics_panel import suppress_logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 80)
print("NEW COLOR SCHEME TEST")
print("=" * 80)

with suppress_logging():
    comparison = compare_approaches(1.0, 20, 10, approaches=[2, 3, 5])

approaches_data = comparison['approaches']

# NEW distinct color scheme
colors = [
    '#3498DB',  # Bright Blue - Heat Exchangers
    '#E74C3C',  # Red - Pumps
    '#2ECC71',  # Green - Piping & Fittings
    '#F39C12',  # Orange - Instrumentation
    '#9B59B6',  # Purple - Valves
    '#F1C40F'   # Yellow - I&C Subtotal
]

color_names = [
    'Bright Blue',
    'Red',
    'Green',
    'Orange',
    'Purple',
    'Yellow'
]

labels = ['Heat Exchangers', 'Pumps', 'Piping & Fittings', 'Instrumentation', 'Valves', 'I&C Subtotal']

print("\nNew Color Scheme:")
print("-" * 80)
for i, (label, color, color_name) in enumerate(zip(labels, colors, color_names)):
    print(f"  {i+1}. {label:<25} → {color_name:<15} ({color})")

# Create test chart
fig, axs = plt.subplots(1, 3, figsize=(18, 7))

approaches = [2, 3, 5]
titles = ["2°C Approach", "3°C Approach", "5°C Approach"]

for idx, (approach, title) in enumerate(zip(approaches, titles)):
    key = f"{approach}C"
    data = approaches_data.get(key, {})

    # Extract cost components
    heat_exchanger = data.get('heat_exchanger', 0)
    pumps = data.get('pumps', 0)
    pipe_fittings = data.get('pipe_fittings', 0)
    instrumentation = data.get('instrumentation', 0)
    valves = data.get('valves', 0)

    # Calculate I&C Subtotal
    ic_subtotal = sum([
        data.get('installation_cost', 0),
        data.get('engineering_cost', 0),
        data.get('contingency_cost', 0)
    ])

    # Combine into values array
    values = [heat_exchanger, pumps, pipe_fittings, instrumentation, valves, ic_subtotal]

    # Create pie chart with new colors
    wedges, texts, autotexts = axs[idx].pie(
        values,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85
    )

    # Make percentage text white and bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)

    # Add legend below the chart
    axs[idx].legend(wedges, labels, loc="upper center", bbox_to_anchor=(0.5, -0.05),
                   fontsize=9, ncol=2, frameon=False)

    axs[idx].set_title(title, fontsize=14, fontweight='bold', pad=10)

# Add figure suptitle
fig.suptitle("Equipment & Installation Cost Breakdown by Approach Temperature",
            fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()

output_file = 'new_color_scheme.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✓ Chart with new colors saved to: {output_file}")

print("\n" + "=" * 80)
print("COLOR DISTINCTIVENESS")
print("=" * 80)
print("\nThe new color scheme uses 6 distinct hue families:")
print("  • Blue (cool) - Heat Exchangers")
print("  • Red (warm) - Pumps")
print("  • Green (cool) - Piping & Fittings")
print("  • Orange (warm) - Instrumentation")
print("  • Purple (cool) - Valves")
print("  • Yellow (warm) - I&C Subtotal")
print("\nNo two colors are in the same hue range!")
print("=" * 80)
