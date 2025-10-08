# Chart Reorganization Documentation

## Changes Made

### Main Charts Section (charts.py)

**Previous**: 2x3 grid (6 positions, 3 used)
**Current**: 1x2 grid (2 positions, both used)
**Removed**: Cost Breakdown pie chart
**Remaining**:
- System Approach Profiles (left)
- Heat Exchanger Effectiveness Gauge (right)

**Code Location**: [python/ui/charts.py:15](../charts.py#L15)

```python
# Old layout
fig, axs = plt.subplots(2, 3, figsize=(18, 10))
create_cost_breakdown_chart(axs[0, 0], costs)
create_approach_profiles_chart(axs[0, 1], system)
create_effectiveness_gauge(axs[0, 2], calculate_effectiveness(analysis))

# New layout
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
create_approach_profiles_chart(axs[0], system)
create_effectiveness_gauge(axs[1], calculate_effectiveness(analysis))
```

### Economics Analysis Section (economics_panel.py)

**Added**: Equipment Cost Breakdown by Approach
**Format**: 3 pie charts in 1x3 grid
**Data**: Order of Magnitude Estimates for all approaches
**Position**: After Cost Contrast chart

**Code Location**: [python/ui/economics_panel.py:397](../economics_panel.py#L397)

```python
def create_approach_cost_breakdown_charts(wha, T1, temp_rise, output_area):
    """
    Create pie charts showing cost breakdown for each approach temperature.

    Displays:
    - 2°C Approach (left)
    - 3°C Approach (center)
    - 5°C Approach (right)

    Each pie chart shows 6 cost components:
    1. Heat Exchangers
    2. Pumps
    3. Piping & Fittings
    4. Instrumentation
    5. Valves
    6. I&C Subtotal (Installation + Engineering + Contingency)
    """
```

## Rationale

### Why Move Cost Breakdown to Economics Panel?

1. **Multiple Approaches**: Economics panel compares all three approaches (2°C, 3°C, 5°C)
2. **Different Percentages**: Each approach has different cost distributions:
   - 2°C: Higher heat exchanger percentage (larger HX needed)
   - 3°C: Balanced distribution
   - 5°C: Higher pump percentage (smaller HX, higher pressure drop)
3. **Data Consistency**: Uses same `compare_approaches()` data as the table and contrast chart
4. **User Context**: Users viewing economics data want to see ALL approaches side-by-side

### Why Not Keep Single Pie Chart in Main Charts?

- Main charts show data for the **selected approach only**
- Economics panel shows data for **all three approaches**
- Pie chart percentages vary by approach, so a single chart would be incomplete
- Side-by-side comparison is more valuable for decision-making

## Data Flow

```
User Input → calculate_heat_reuse()
             ├→ Single approach data → Main Charts (2 charts)
             │                        ├→ System Approach Profiles
             │                        └→ Effectiveness Gauge
             │
             └→ compare_approaches() → Economics Panel
                                      ├→ Comparison Table
                                      ├→ Cost Contrast Chart
                                      └→ Cost Breakdown Charts (3 pies)
```

## Visual Layout

### Main Charts (charts.py)

```
┌─────────────────────────────────────────────────────────────┐
│  Heat Reuse System Analysis - 1.0MW System                  │
├──────────────────────────────┬──────────────────────────────┤
│                              │                              │
│   System Approach Profiles   │   HX Effectiveness Gauge     │
│                              │                              │
│   [Temperature vs Time]      │   [Gauge: 85.3%]             │
│                              │                              │
│   TCS: 20°C → 30°C           │   Performance: Excellent     │
│   FWS: 18°C → 28°C           │                              │
│                              │                              │
└──────────────────────────────┴──────────────────────────────┘
```

### Economics Panel (economics_panel.py)

```
┌──────────────────────────────────────────────────────────────┐
│  💰 Economics Analysis - Order of Magnitude Estimate         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  [Comparison Table - All Approaches]                         │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  📈 Cost Contrast Analysis                                   │
│                                                              │
│  [Line chart: Capital vs Operating costs]                    │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  🔧 Equipment Cost Breakdown by Approach                     │
│                                                              │
│  ┌──────────┬──────────┬──────────┐                         │
│  │ 2°C      │ 3°C      │ 5°C      │                         │
│  │ Approach │ Approach │ Approach │                         │
│  │          │          │          │                         │
│  │ [Pie]    │ [Pie]    │ [Pie]    │                         │
│  │          │          │          │                         │
│  └──────────┴──────────┴──────────┘                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Cost Components Breakdown

### Why 6 Components (Not 5)?

The pie charts show 6 components instead of the traditional 5 equipment items:

1. **Heat Exchangers** - Equipment cost varies significantly by approach
2. **Pumps** - Equipment cost varies by approach (2°C/3°C: €35k, 5°C: €45k)
3. **Piping & Fittings** - Equipment cost (relatively constant)
4. **Instrumentation** - Equipment cost (constant at €30k)
5. **Valves** - Equipment cost (flow-based)
6. **I&C Subtotal** - Installation + Engineering + Contingency combined

### Why Combine I&C?

- Installation, Engineering, and Contingency are percentage-based (not equipment)
- Combining them shows "overhead" as a single proportion
- Makes equipment costs vs. project overhead more visible
- Reduces visual clutter (6 slices vs. 8 slices)

## Implementation Details

### Color Scheme

```python
colors = ['#4ECDC4', '#FFA07A', '#FF6B6B', '#45B7D1', '#98D8C8', '#FFD93D']
```

- Teal (`#4ECDC4`): Heat Exchangers
- Light Orange (`#FFA07A`): Pumps
- Red (`#FF6B6B`): Piping & Fittings
- Blue (`#45B7D1`): Instrumentation
- Light Teal (`#98D8C8`): Valves
- Yellow (`#FFD93D`): I&C Subtotal

### Data Accuracy

All values come directly from `compare_approaches()`:

```python
with suppress_logging():
    comparison = compare_approaches(wha, T1, temp_rise, approaches=[2, 3, 5])

approaches_data = comparison['approaches']

for approach in [2, 3, 5]:
    key = f"{approach}C"
    data = approaches_data.get(key, {})

    # Equipment costs
    heat_exchanger = data.get('heat_exchanger', 0)
    pumps = data.get('pumps', 0)
    pipe_fittings = data.get('pipe_fittings', 0)
    instrumentation = data.get('instrumentation', 0)
    valves = data.get('valves', 0)

    # I&C Subtotal
    ic_subtotal = sum([
        data.get('installation_cost', 0),
        data.get('engineering_cost', 0),
        data.get('contingency_cost', 0)
    ])
```

### Percentage Display

```python
autopct='%1.1f%%'
textprops={'color': 'white', 'weight': 'bold'}
```

- Shows percentages with 1 decimal place (e.g., "25.3%")
- White text for visibility on colored backgrounds
- Bold font for emphasis

## Benefits

1. **Comprehensive Comparison**: Users see cost distribution for all approaches at once
2. **Decision Support**: Easy to spot which approach has higher equipment vs. overhead costs
3. **Data Consistency**: All Economics panel visualizations use same underlying data
4. **Space Efficiency**: Main charts remain focused on technical performance
5. **Visual Clarity**: Each approach gets its own dedicated pie chart

## Testing Verification

To verify the implementation:

1. Run the analysis tool with any parameters
2. Check that Main Charts show only 2 visualizations (1x2 grid)
3. Scroll to Economics Analysis panel
4. Verify 3 pie charts appear after Cost Contrast chart
5. Confirm percentages in each pie chart sum to 100%
6. Verify percentages match ratios in Comparison Table above

## Files Modified

1. **[python/ui/charts.py](../charts.py)**
   - Changed `create_system_charts()` layout from 2x3 to 1x2
   - Removed `create_cost_breakdown_chart()` call
   - Updated chart positioning

2. **[python/ui/economics_panel.py](../economics_panel.py)**
   - Added `create_approach_cost_breakdown_charts()` function
   - Updated `display_economics_analysis()` to call new function
   - Added section header with description

3. **Documentation**
   - Updated UI_CALCULATION_MAP.md
   - Updated COST_MODULE_SUMMARY.md
   - Updated Cost_Restructuring_Summary.md
   - Created this CHART_REORGANIZATION.md

---

**Date**: 2025-10-08
**Status**: ✓ Completed
**Breaking Changes**: None (backward compatible)
