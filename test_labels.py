"""
Test pie chart labels visibility
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

with suppress_logging():
    comparison = compare_approaches(1.0, 20, 10, approaches=[2, 3, 5])

approaches_data = comparison['approaches']

# Create a single test pie chart
fig, ax = plt.subplots(figsize=(8, 8))

data = approaches_data['2C']
colors = ['#4ECDC4', '#FFA07A', '#FF6B6B', '#45B7D1', '#98D8C8', '#FFD93D']
labels = ['Heat Exchangers', 'Pumps', 'Piping & Fittings', 'Instrumentation', 'Valves', 'I&C Subtotal']

values = [
    data.get('heat_exchanger', 0),
    data.get('pumps', 0),
    data.get('pipe_fittings', 0),
    data.get('instrumentation', 0),
    data.get('valves', 0),
    sum([data.get('installation_cost', 0), data.get('engineering_cost', 0), data.get('contingency_cost', 0)])
]

print("Creating pie chart with labels...")
print(f"Labels: {labels}")
print(f"Values: {[f'€{v:,.0f}' for v in values]}")

wedges, texts, autotexts = ax.pie(
    values,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    textprops={'fontsize': 10},
    pctdistance=0.85
)

print(f"\nNumber of label texts: {len(texts)}")
print(f"Number of percentage texts: {len(autotexts)}")

# Make percentage text white and bold
for i, autotext in enumerate(autotexts):
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)
    print(f"  Percentage {i}: {autotext.get_text()}")

# Make label text bold
for i, text in enumerate(texts):
    text.set_fontsize(9)
    text.set_fontweight('bold')
    print(f"  Label {i}: {text.get_text()}")

ax.set_title("2°C Approach - Test Labels", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('test_labels.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Test chart saved to: test_labels.png")

# Also create version with legend as backup
fig2, ax2 = plt.subplots(figsize=(10, 8))

wedges2, texts2, autotexts2 = ax2.pie(
    values,
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.85
)

for autotext in autotexts2:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)

ax2.legend(wedges2, labels, title="Components", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
ax2.set_title("2°C Approach - With Legend", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('test_labels_legend.png', dpi=150, bbox_inches='tight')
print(f"✓ Test chart with legend saved to: test_labels_legend.png")

print("\nBoth versions created. Check which one has better label visibility.")
