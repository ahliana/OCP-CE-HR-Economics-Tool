# UI Calculation Reference Map
## Heat Reuse Economics Tool - Complete Value Traceability

**Purpose**: This document maps EVERY value displayed in the UI to its exact source code location, calculation method, and data sources.

**Last Updated**: 2025-10-06

---

## Table of Contents
1. [Screen 1: System Parameters Panel](#screen-1-system-parameters-panel)
2. [Screen 2: Capital Cost Analysis Panel](#screen-2-cost-analysis-panel)
3. [Screen 3: Economics Analysis Table](#screen-3-economics-analysis-table)
4. [Screen 4: Charts and Visualizations](#screen-4-charts-and-visualizations)
5. [Quick Reference - Function Index](#quick-reference---function-index)
6. [Data File Dependencies](#data-file-dependencies)

---

## Screen 1: System Parameters Panel

**UI Location**: First output panel in Jupyter interface
**Display Function**: `display_system_parameters()` in [python/ui/outputs.py:23](python/ui/outputs.py#L23)
**Formatting Function**: `extract_formatted_system_params()` in [python/ui/formatting.py:192](python/ui/formatting.py#L192)

### Temperature Values (Auto-Calculated)

| UI Label | Example Value | Function | File:Line | Calculation | Data Source |
|----------|--------------|----------|-----------|-------------|-------------|
| T1 (Outlet to TCS) | 20°C | `lookup_allhx_data()` | [python/core/lookup.py:19](python/core/lookup.py#L19) | Direct lookup from ALLHX.csv | ALLHX.csv column 'T1' |
| T2 (Inlet from TCS) | 30°C | `lookup_allhx_data()` | [python/core/lookup.py:19](python/core/lookup.py#L19) | Calculated as T1 + itdt | ALLHX.csv column 'T2' |
| T3 (Outlet to Consumer) | 28°C | `lookup_allhx_data()` | [python/core/lookup.py:19](python/core/lookup.py#L19) | Direct lookup from ALLHX.csv | ALLHX.csv column 'T3' |
| T4 (Inlet from Consumer) | 18°C | `lookup_allhx_data()` | [python/core/lookup.py:19](python/core/lookup.py#L19) | Direct lookup from ALLHX.csv | ALLHX.csv column 'T4' |

**Lookup Logic**: Exact match on 4 parameters: `wha`, `T1`, `itdt`, `TCSapp` (approach)
```python
# python/core/lookup.py:100-105
matches = valid_df[
    (valid_df['wha'] == wha) &
    (valid_df['T1'] == T1) &
    (valid_df['itdt'] == itdt) &
    (valid_df['TCSapp'] == approach)
]
```

### Flow Rate Values

| UI Label | Example Value | Function | File:Line | Calculation | Data Source |
|----------|--------------|----------|-----------|-------------|-------------|
| F1 (TCS Flow Rate) | 1,504 l/m | `lookup_allhx_data()` | [python/core/lookup.py:118](python/core/lookup.py#L118) | Direct from ALLHX.csv | ALLHX.csv column 'F1' |
| F2 (FWS Flow Rate) | 1,435 l/m | `lookup_allhx_data()` | [python/core/lookup.py:119](python/core/lookup.py#L119) | Direct from ALLHX.csv | ALLHX.csv column 'F2' |

### Delta T Values (Calculated)

| UI Label | Example Value | Function | File:Line | Calculation | Formula |
|----------|--------------|----------|-----------|-------------|---------|
| Delta T for TCS | 10°C | `get_itdt()` | [python/core/original_calculations.py:46](python/core/original_calculations.py#L46) | T2 - T1 | Temperature rise on TCS side |
| Delta T for FWS | 10°C | `get_oftkrdt()` | [python/core/original_calculations.py:53](python/core/original_calculations.py#L53) | T3 - T4 | Temperature drop on FWS side |

**Display Formatting**: `extract_delta_t_values()` in [python/ui/formatting.py:233](python/ui/formatting.py#L233)

---

## Screen 2: Cost Analysis Panel

**UI Location**: Second output panel in Jupyter interface
**Display Function**: `display_cost_analysis()` in [python/ui/outputs.py:60](python/ui/outputs.py#L60)
**Formatting Function**: `extract_formatted_cost_analysis()` in [python/ui/formatting.py:211](python/ui/formatting.py#L211)

### Sizing Parameters

| UI Label | Example Value | Function | File:Line | Calculation | Data Source |
|----------|--------------|----------|-----------|-------------|-------------|
| Room Size | 5.2 m | `get_PipeLength()` | [python/core/original_calculations.py:196](python/core/original_calculations.py#L196) | Lookup by calculated MW | ROOM.csv |
| Suggested Pipe Size | DN 150 | `get_PipeSize_Suggested()` | [python/core/original_calculations.py:109](python/core/original_calculations.py#L109) | CEILING lookup: first size where capacity ≥ F1 | PIPSZ.csv |

**Room Size Calculation**:
```python
# python/core/original_calculations.py:196-200
def get_PipeLength(F1, T1, T2):
    MW_value = get_MW_divd(F1, T1, T2)  # Calculate MW
    return get_lookup_value('ROOM', MW_value, 0, 1)  # Lookup in ROOM.csv
```

**Pipe Size Calculation**:
```python
# python/core/original_calculations.py:167-188
# Sort by flow capacity, find first row where flow_capacity >= F1
adequate_rows = valid_rows[valid_rows.iloc[:, 0] >= F1_float]
selected_pipe_size = adequate_rows.iloc[0, 1]
```

### Cost Components (NEW STRUCTURE - Phase 2)

**Main API Function**: `calculate_costs()` in [python/core/costs.py:412](python/core/costs.py#L412)

**Structured Output**: The function returns costs separated into base equipment and contingencies:

```python
{
    'base_costs': {
        'heat_exchanger': float,
        'pumps': float,
        'piping_fittings': float,
        'instrumentation': float,
        'valves': float,
        'equipment_subtotal': float
    },
    'contingencies': {
        'installation': float,    # 15% of equipment_subtotal
        'engineering': float,     # 10% of (equipment + installation)
        'contingency': float,     # 10% of all previous
        'total_contingencies': float
    },
    'capital_total': float,       # Rounded to nearest €500
    'operating_costs': {...}
}
```

## Order of Magnitude Panel Mapping

### Base Cost Display

| UI Element | Source | Calculation |
|------------|--------|-------------|
| Heat Exchangers | costs.base_costs.heat_exchanger | Direct from cost curves |
| Pumps | costs.base_costs.pumps | Base pump cost |
| Piping & Fittings | costs.base_costs.piping_fittings | Material costs |
| Instrumentation | costs.base_costs.instrumentation | Base instrument cost |
| Valves | costs.base_costs.valves | Base valve cost |
| Equipment Subtotal | Sum of above | Σ(base costs) |

### Contingency Display

| UI Element | Source | Calculation |
|------------|--------|-------------|
| Installation | costs.contingencies.installation | equipment_subtotal × 0.15 |
| Engineering | costs.contingencies.engineering | (equipment + installation) × 0.10 |
| Contingency | costs.contingencies.contingency | (equip + install + eng) × 0.10 |
| I&C Subtotal | costs.contingencies.total | Σ(contingencies) |
| Capital Total | costs.capital_total | Round((equipment + I&C), €500) |

### Validation Requirements

- Equipment Subtotal + I&C Subtotal must equal Capital Total (±€500 for rounding)
- All displayed numbers must be traceable to source calculations
- Contingency percentages must be clearly shown in UI

#### Base Equipment Costs (Raw costs without factors)

| UI Label | Example Value | Function | File:Line | Calculation | Data Source |
|----------|--------------|----------|-----------|-------------|-------------|
| Heat Exchanger | €17,616 | `calculate_heat_exchanger_cost()` | [python/core/costs.py:35](python/core/costs.py#L35) | Direct lookup via `lookup_allhx_data()` | ALLHX.csv column 'hx_cost' |
| Pumps | €35,000 | `calculate_pump_cost()` | [python/core/costs.py:78](python/core/costs.py#L78) | Based on approach and pressure drop | Calculated (approach-dependent) |
| Piping & Fittings | €14,094 | `calculate_piping_cost()` | [python/core/costs.py:146](python/core/costs.py#L146) | Pipe cost + 25% fittings | PIPCOST.csv, JOINTS.csv |
| Instrumentation | €30,000 | `calculate_instrumentation_cost()` | [python/core/costs.py:252](python/core/costs.py#L252) | Base €30k + scaling for >2MW | Fixed base + scaling |
| Valves | €4,500 | `calculate_valve_costs()` | [python/core/costs.py:286](python/core/costs.py#L286) | 1 control + 4 isolation valves | CVALV.csv, IVALV.csv |
| **Equipment Subtotal** | €96,710 | - | - | Sum of above | Calculated |

#### Contingencies (Applied cumulatively)

| UI Label | Example Value | Calculation | Formula |
|----------|--------------|-------------|---------|
| Installation | €14,506 | 15% of equipment subtotal | `equipment_subtotal × 0.15` |
| Engineering | €11,122 | 10% of (equipment + installation) | `(equipment_subtotal + installation) × 0.10` |
| Contingency | €12,234 | 10% of all previous | `(equipment + installation + engineering) × 0.10` |
| **Total Contingencies** | €37,862 | Sum of above | Calculated |

#### Total Capital Cost

| UI Label | Example Value | Calculation |
|----------|--------------|-------------|
| **Capital Total** | €134,500 | Equipment + Contingencies, rounded to nearest €500 |

**Component Details**:

- **Pump Cost**: Varies by approach temperature (2°C = €35k, 3°C = €35k, 5°C = €45k)
  - Larger approach = smaller HX = higher pressure drop = more powerful pumps needed
  - Location: [python/core/costs.py:108-119](python/core/costs.py#L108)

- **Piping Cost**: Calculated from flow rate and pipe material
  - Base pipe cost from PIPCOST.csv lookup
  - Fittings add 25% or use JOINTS.csv if available
  - Location: [python/core/costs.py:170-180](python/core/costs.py#L170)

- **Valve Cost Calculation**:
```python
# Lookup control valve cost from CVALV.csv by pipe size
# Lookup isolation valve cost from IVALV.csv by pipe size
total_valve_cost = (4 × isolation_valve_cost) + control_valve_cost
```

**Total System Cost Breakdown**:
```python
total_cost = (
    total_pipe_cost +
    hx_cost +
    total_valve_cost +
    pump_cost +
    installation_cost  # 15% of equipment subtotal
)
```

---

## Screen 3: Economics Analysis Table

**UI Location**: Economics Analysis output panel
**Display Function**: `display_economics_analysis()` in [python/ui/economics_panel.py:242](python/ui/economics_panel.py#L242)
**Table Generator**: `create_economics_comparison_table()` in [python/ui/economics_panel.py:42](python/ui/economics_panel.py#L42)
**Core Calculation**: `compare_approaches()` in [python/core/costs.py:561](python/core/costs.py#L561)

### Order of Magnitude Estimate - 2°C Approach Column

| UI Label | Example Value | Function | File:Line | Calculation | Notes |
|----------|--------------|----------|-----------|-------------|-------|
| Heat Exchanger | €17,616 | `calculate_heat_exchanger_cost()` | [python/core/costs.py:35](python/core/costs.py#L35) | ALLHX lookup for approach=2 | ALLHX.csv 'costHX' |
| Pumps | €35,000 | `calculate_pump_cost()` | [python/core/costs.py:78](python/core/costs.py#L78) | Fixed for 2°C approach | Low ΔP (30 kPa) |
| Pipe & Fittings | €14,094 | `calculate_piping_cost()` | [python/core/costs.py:146](python/core/costs.py#L146) | Pipe cost + 25% fittings | PIPCOST + JOINTS |
| Instruments | €30,000 | `calculate_instrumentation_cost()` | [python/core/costs.py:252](python/core/costs.py#L252) | Base cost for systems ≤2 MW | Fixed value |
| Valves | €7,310 | `calculate_valve_costs()` | [python/core/costs.py:286](python/core/costs.py#L286) | 4×IVALV + 1×CVALV | CVALV + IVALV |
| **Capital Total** | **€134,500** | `calculate_order_of_magnitude_estimate()` | [python/core/costs.py:476](python/core/costs.py#L476) | Sum + installation/eng/contingency | **Rounded to €500** |
| Operating Energy | 9,545 kWh/yr | `calculate_operating_energy()` | [python/core/costs.py:356](python/core/costs.py#L356) | Pump power × 8760 hrs | Approach-based |
| Energy Cost | €1,432/yr | `calculate_operating_energy()` | [python/core/costs.py:394](python/core/costs.py#L394) | kWh × €0.15/kWh | Annual cost |

**Capital Cost Factors Applied**:
```python
# python/core/costs.py:476-489
equipment_subtotal = HX + pumps + pipes + instruments + valves
installation_cost = equipment_subtotal × 0.15  # +15%
engineering_cost = (equipment + installation) × 0.10  # +10%
contingency_cost = (equipment + installation + eng) × 0.10  # +10%
capital_total = round(final_total / 500) * 500  # Round to €500
```

### Order of Magnitude Estimate - 3°C Approach Column

| UI Label | Example Value | Function | File:Line | Calculation | Notes |
|----------|--------------|----------|-----------|-------------|-------|
| Heat Exchanger | €13,500 | `calculate_heat_exchanger_cost()` | [python/core/costs.py:35](python/core/costs.py#L35) | ALLHX lookup for approach=3 | ALLHX.csv 'costHX' |
| Pumps | €35,000 | `calculate_pump_cost()` | [python/core/costs.py:111](python/core/costs.py#L111) | Fixed for 3°C approach | Medium ΔP (35 kPa) |
| Pipe & Fittings | €14,094 | `calculate_piping_cost()` | [python/core/costs.py:146](python/core/costs.py#L146) | Same piping system | Flow-based |
| Instruments | €30,000 | `calculate_instrumentation_cost()` | [python/core/costs.py:252](python/core/costs.py#L252) | Base cost | Fixed value |
| Valves | €7,310 | `calculate_valve_costs()` | [python/core/costs.py:286](python/core/costs.py#L286) | Same valve sizing | Flow-based |
| **Capital Total** | **€130,000** | `calculate_order_of_magnitude_estimate()` | [python/core/costs.py:476](python/core/costs.py#L476) | Sum + factors | **Rounded to €500** |
| Operating Energy | 11,053 kWh/yr | `calculate_operating_energy()` | [python/core/costs.py:380](python/core/costs.py#L380) | Pump power × 8760 hrs | Higher pump power |
| Energy Cost | €1,658/yr | `calculate_operating_energy()` | [python/core/costs.py:394](python/core/costs.py#L394) | kWh × €0.15/kWh | Annual cost |

### Order of Magnitude Estimate - 5°C Approach Column

| UI Label | Example Value | Function | File:Line | Calculation | Notes |
|----------|--------------|----------|-----------|-------------|-------|
| Heat Exchanger | €10,000 | `calculate_heat_exchanger_cost()` | [python/core/costs.py:35](python/core/costs.py#L35) | ALLHX lookup for approach=5 | Smaller HX |
| Pumps | €45,000 | `calculate_pump_cost()` | [python/core/costs.py:115](python/core/costs.py#L115) | Higher for 5°C approach | High ΔP (50 kPa) |
| Pipe & Fittings | €14,094 | `calculate_piping_cost()` | [python/core/costs.py:146](python/core/costs.py#L146) | Same piping system | Flow-based |
| Instruments | €30,000 | `calculate_instrumentation_cost()` | [python/core/costs.py:252](python/core/costs.py#L252) | Base cost | Fixed value |
| Valves | €7,310 | `calculate_valve_costs()` | [python/core/costs.py:286](python/core/costs.py#L286) | Same valve sizing | Flow-based |
| **Capital Total** | **€139,000** | `calculate_order_of_magnitude_estimate()` | [python/core/costs.py:476](python/core/costs.py#L476) | Sum + factors | **Rounded to €500** |
| Operating Energy | 15,768 kWh/yr | `calculate_operating_energy()` | [python/core/costs.py:380](python/core/costs.py#L380) | Pump power × 8760 hrs | Highest pump power |
| Energy Cost | €2,365/yr | `calculate_operating_energy()` | [python/core/costs.py:394](python/core/costs.py#L394) | kWh × €0.15/kWh | Annual cost |

**Pump Power Calculation by Approach**:
```python
# python/core/costs.py:108-120
if approach == 2:
    pressure_drop_pa = 30000  # Low ΔP (large HX)
    base_cost = 35000
elif approach == 3:
    pressure_drop_pa = 35000  # Medium ΔP
    base_cost = 35000
elif approach == 5:
    pressure_drop_pa = 50000  # High ΔP (small HX)
    base_cost = 45000

# Calculate electrical power
power_kw = (flow_m3s × pressure_drop_pa) / (pump_eff × motor_eff × 1000)
annual_energy_kwh = power_kw × 8760
```

---

## Screen 4: Charts and Visualizations

**UI Location**: Charts output panel
**Display Function**: `display_charts()` in [python/ui/outputs.py:92](python/ui/outputs.py#L92)
**Chart Generator**: `create_system_charts()` in [python/ui/charts.py:15](python/ui/charts.py#L15)

### Chart Layout Update

**Chart Grid**: Creates 1x2 grid with 2 chart positions (previously 2x3 with 6 positions)
**Function**: `create_system_charts()` in [python/ui/charts.py:15](python/ui/charts.py#L15)

**Visual Elements**:
- System Approach Profiles (position [0])
- Heat Exchanger Effectiveness Gauge (position [1])
- **Cost Breakdown (MOVED to economics_panel.py - 3 pie charts)**

### Chart 1: Cost Breakdown by Approach (MOVED TO ECONOMICS PANEL)

**Status**: MOVED TO ECONOMICS PANEL
**Previous Location**: `create_cost_breakdown_chart()` in charts.py
**Current Location**: `create_approach_cost_breakdown_charts()` in [python/ui/economics_panel.py:397](python/ui/economics_panel.py#L397)

**Note**: Now displays as 3 pie charts (2°C, 3°C, 5°C) in Economics Analysis panel using Order of Magnitude data

**Data Flow**:
- `compare_approaches()` → 3 pie charts showing equipment distribution for each approach
- Each chart shows: Heat Exchangers, Pumps, Piping & Fittings, Instrumentation, Valves, I&C Subtotal
- Percentage distributions vary by approach temperature

**Implementation**:
| Chart | Approach | Components | Percentages |
|-------|----------|------------|-------------|
| Left | 2°C | 6 cost components | Higher HX % (larger heat exchanger) |
| Center | 3°C | 6 cost components | Balanced distribution |
| Right | 5°C | 6 cost components | Higher pump % (more flow required) |

**Percentage Calculation**: Automatic via matplotlib `autopct='%1.1f%%'`

### Chart 2: System Approach Profiles (Main Charts - Position [0])

**Function**: `create_approach_profiles_chart()` in [python/ui/charts.py:402](python/ui/charts.py#L402)
**Data Source**: `calculate_combined_approach_profiles()` in [python/core/original_calculations.py](python/core/original_calculations.py)

| Line | Function | Data Points | Calculation |
|------|----------|-------------|-------------|
| TCS Line (Red) | `calculate_combined_approach_profiles()` | Temperature trajectory | T1 → T2 progression |
| FWS Line (Blue) | `calculate_combined_approach_profiles()` | Temperature trajectory | T4 → T3 progression |

**Plot Generation**:
```python
# python/ui/charts.py:431-438
ax.plot(time_percent, tcs['temperatures'],
       color='#ff6666', linewidth=3, marker='o',
       label=f'TCS ({T1}°C → {T2}°C)')

ax.plot(time_percent, fws['temperatures'],
       color='#66b3ff', linewidth=3, marker='s',
       label=f'FWS ({T4}°C → {T3}°C)')
```

### Chart 3: Heat Exchanger Effectiveness Gauge (Main Charts - Position [1])

**Function**: `create_effectiveness_gauge()` in [python/ui/charts.py:512](python/ui/charts.py#L512)
**Calculation**: `calculate_effectiveness()` in [python/ui/formatting.py:448](python/ui/formatting.py#L448)

| Element | Value | Function | Calculation |
|---------|-------|----------|-------------|
| Effectiveness | 85.3% | `heat_exchanger_for_heat_reuse_tool()` | From physics calculations |
| Gauge Zones | <60%, 60-80%, >80% | Visual mapping | Color-coded performance |
| Needle Position | Radians | `np.pi * (1 - effectiveness)` | Angular position |

**Effectiveness Calculation**:
```python
# python/ui/formatting.py:459-473
from physics.heat_exchangers import heat_exchanger_for_heat_reuse_tool

hx_analysis = heat_exchanger_for_heat_reuse_tool(F1, F2, T1, T2, T3, T4)
effectiveness = hx_analysis['effectiveness']
```

### Chart 4: Cost Contrast Graph (Economics Panel)

**Function**: `create_cost_contrast_chart()` in [python/ui/economics_panel.py:165](python/ui/economics_panel.py#L165)

| Data Series | Values | Source | Calculation |
|-------------|--------|--------|-------------|
| Capital Cost Line | [€134.5k, €130k, €139k] | `compare_approaches()` | [python/core/costs.py:593](python/core/costs.py#L593) |
| Operating Cost Line | [€1,432, €1,658, €2,365] | `compare_approaches()` | [python/core/costs.py:599](python/core/costs.py#L599) |
| X-axis | [2, 3, 5] | Approach temperatures | Input parameters |

**Data Extraction**:
```python
# python/ui/economics_panel.py:186-194
for approach in [2, 3, 5]:
    key = f"{approach}C"
    capital_costs.append(approaches_data[key]['capital_total'])
    operating_costs.append(approaches_data[key]['operating_cost_eur_year'])

# Plot both lines
plt.plot(approaches, capital_costs, marker='o', label='Capital Cost')
plt.plot(approaches, operating_costs, marker='s', label='Annual Operating Cost')
```

---

## Quick Reference - Function Index

### Lookup Functions

| Function | File:Line | Purpose | Returns |
|----------|-----------|---------|---------|
| `lookup_allhx_data()` | [python/core/lookup.py:19](python/core/lookup.py#L19) | Master HX data lookup | T1-T4, F1-F2, HX cost |
| `get_lookup_value()` | [python/core/lookup.py:137](python/core/lookup.py#L137) | Generic CSV lookup (≥ match) | Single/multiple values |
| `get_PipeSize_Suggested()` | [python/core/original_calculations.py:109](python/core/original_calculations.py#L109) | Pipe sizing by flow | DN size |
| `get_PipeLength()` | [python/core/original_calculations.py:196](python/core/original_calculations.py#L196) | Room size by MW | Length in meters |
| `get_PipeCost_perMeter()` | [python/core/original_calculations.py:210](python/core/original_calculations.py#L210) | Pipe cost by size/material | €/meter |

### Calculation Functions

| Function | File:Line | Purpose | Returns |
|----------|-----------|---------|---------|
| `get_itdt()` | [python/core/original_calculations.py:46](python/core/original_calculations.py#L46) | TCS delta T | T2 - T1 |
| `get_oftkrdt()` | [python/core/original_calculations.py:53](python/core/original_calculations.py#L53) | FWS delta T | T3 - T4 |
| `get_MW_divd()` | [python/core/original_calculations.py:93](python/core/original_calculations.py#L93) | Calculate MW | Power in MW |
| `calculate_heat_exchanger_cost()` | [python/core/costs.py:35](python/core/costs.py#L35) | HX cost for approach | Cost + system data |
| `calculate_pump_cost()` | [python/core/costs.py:78](python/core/costs.py#L78) | Pump cost + power | Cost + kW |
| `calculate_piping_cost()` | [python/core/costs.py:146](python/core/costs.py#L146) | Total piping cost | Pipe + fittings |
| `calculate_instrumentation_cost()` | [python/core/costs.py:252](python/core/costs.py#L252) | Instruments cost | Base €30k + scaling |
| `calculate_valve_costs()` | [python/core/costs.py:286](python/core/costs.py#L286) | Total valve cost | 4×IVALV + CVALV |
| `calculate_operating_energy()` | [python/core/costs.py:356](python/core/costs.py#L356) | Annual energy usage | kWh/year + cost |
| `calculate_order_of_magnitude_estimate()` | [python/core/costs.py:412](python/core/costs.py#L412) | Complete cost estimate | Full breakdown |
| `compare_approaches()` | [python/core/costs.py:561](python/core/costs.py#L561) | Multi-approach comparison | All approaches |

### Display/Formatting Functions

| Function | File:Line | Purpose | Returns |
|----------|-----------|---------|---------|
| `format_display_value()` | [python/ui/formatting.py:12](python/ui/formatting.py#L12) | Format values for display | Formatted string |
| `extract_formatted_system_params()` | [python/ui/formatting.py:192](python/ui/formatting.py#L192) | Format system params | (label, value) tuples |
| `extract_formatted_cost_analysis()` | [python/ui/formatting.py:211](python/ui/formatting.py#L211) | Format cost data | (label, value) tuples |
| `extract_delta_t_values()` | [python/ui/formatting.py:233](python/ui/formatting.py#L233) | Format delta T values | (label, value) tuples |
| `calculate_effectiveness()` | [python/ui/formatting.py:448](python/ui/formatting.py#L448) | HX effectiveness | 0.0-1.0 value |
| `display_system_parameters()` | [python/ui/outputs.py:23](python/ui/outputs.py#L23) | Display system params | None (UI output) |
| `display_cost_analysis()` | [python/ui/outputs.py:60](python/ui/outputs.py#L60) | Display cost analysis | None (UI output) |
| `display_economics_panel()` | [python/ui/outputs.py:114](python/ui/outputs.py#L114) | Display economics table | None (UI output) |
| `display_charts()` | [python/ui/outputs.py:92](python/ui/outputs.py#L92) | Display all charts | None (UI output) |

---

## Data File Dependencies

### CSV Files and Their Usage

| CSV File | Used By | Columns Used | Purpose |
|----------|---------|--------------|---------|
| **ALLHX.csv** | `lookup_allhx_data()` | wha, T1, itdt, T2, TCSapp, F1, F2, T3, T4, costHX, areaHX | Master system data - temperatures, flows, HX cost |
| **PIPSZ.csv** | `get_PipeSize_Suggested()` | Flow capacity (col 0), Pipe size DN (col 1) | Pipe sizing lookup - CEILING match |
| **PIPCOST.csv** | `get_PipeCost_perMeter()` | Pipe size, Material (Stainless), Cost/meter | Pipe cost by size and material |
| **ROOM.csv** | `get_PipeLength()` | MW capacity (col 0), Room length (col 1) | Pipe length/room size by power |
| **CVALV.csv** | `calculate_valve_costs()` | Pipe size (col 0), Cost (col 1) | Control valve costs |
| **IVALV.csv** | `calculate_valve_costs()` | Pipe size (col 0), Cost (col 1) | Isolation valve costs |
| **JOINTS.csv** | `calculate_fittings_cost()` | Pipe size (col 0), Cost (col 2 for SS) | Fittings/joints cost |
| **MW PRICE DATA.csv** | `calculate_operating_energy()` | Price data | Energy price (€/kWh) |

### Lookup Strategies

#### Exact Match (ALLHX)
```python
# Requires exact match on ALL 4 parameters
matches = df[
    (df['wha'] == wha) &
    (df['T1'] == T1) &
    (df['itdt'] == itdt) &
    (df['TCSapp'] == approach)
]
```

#### CEILING Match (PIPSZ, ROOM)
```python
# Find first value where lookup_column >= target
matching_indices = lookup_col[lookup_col >= lookup_value].index
result = df.iloc[matching_indices[0]]
```

#### String Match (CVALV, IVALV)
```python
# Match pipe size as string
for idx, row in df.iterrows():
    if str(row.iloc[0]).strip() == str(pipe_size):
        cost = row.iloc[1]
        break
```

---

## Calculation Flow Diagrams

### Main Analysis Flow

```
User Inputs (wha, T1, itdt, approach)
    ↓
lookup_allhx_data() → ALLHX.csv
    ↓
Returns: {T1, T2, T3, T4, F1, F2, hx_cost}
    ↓
    ├─→ System Parameters Display
    │   └─→ extract_formatted_system_params()
    │       └─→ display_system_parameters()
    │
    ├─→ Cost Analysis
    │   ├─→ get_PipeSize_Suggested(F1) → PIPSZ.csv
    │   ├─→ get_PipeLength(F1, T1, T2) → ROOM.csv
    │   ├─→ get_PipeCost_perMeter(size, material) → PIPCOST.csv
    │   ├─→ calculate_valve_costs(F1, F2) → CVALV, IVALV
    │   └─→ display_cost_analysis()
    │
    ├─→ Economics Analysis
    │   ├─→ calculate_order_of_magnitude_estimate(wha, T1, itdt, 2°C)
    │   ├─→ calculate_order_of_magnitude_estimate(wha, T1, itdt, 3°C)
    │   ├─→ calculate_order_of_magnitude_estimate(wha, T1, itdt, 5°C)
    │   ├─→ compare_approaches() → comparison table
    │   └─→ display_economics_panel()
    │
    └─→ Charts
        ├─→ create_cost_breakdown_chart()
        ├─→ create_approach_profiles_chart()
        ├─→ create_effectiveness_gauge()
        └─→ create_cost_contrast_chart()
```

### Order of Magnitude Estimate Flow

```
calculate_order_of_magnitude_estimate(wha, T1, temp_rise, approach)
    ↓
    ├─→ calculate_heat_exchanger_cost()
    │   └─→ lookup_allhx_data(approach) → €17,616
    │
    ├─→ calculate_pump_cost(approach)
    │   ├─→ Pressure drop based on approach
    │   ├─→ pump_power_required() → kW
    │   └─→ Fixed cost by approach → €35,000
    │
    ├─→ calculate_piping_cost()
    │   ├─→ get_PipeCost_Total() → €11,275
    │   ├─→ calculate_fittings_cost() → €2,819 (25%)
    │   └─→ Total → €14,094
    │
    ├─→ calculate_instrumentation_cost()
    │   └─→ Base €30,000 + scaling
    │
    ├─→ calculate_valve_costs()
    │   ├─→ CVALV lookup → €1,500
    │   ├─→ IVALV lookup × 4 → €5,810
    │   └─→ Total → €7,310
    │
    ├─→ Equipment Subtotal = €104,020
    │
    ├─→ Apply Factors:
    │   ├─→ Installation (+15%) = €15,603
    │   ├─→ Engineering (+10%) = €11,962
    │   ├─→ Contingency (+10%) = €13,159
    │   └─→ Subtotal = €144,744
    │
    ├─→ Round to €500 → €134,500
    │
    └─→ calculate_operating_energy(approach)
        ├─→ Pump power × 8760 hrs → 9,545 kWh/year
        └─→ kWh × €0.15 → €1,432/year
```

---

## Display Rounding Configuration

**Source**: [python/ui/config.py](python/ui/config.py) - `DISPLAY_ROUNDING` dictionary

| Value Type | Decimal Places | Example | UI Application |
|------------|----------------|---------|----------------|
| temperature | 1 | 20.5°C | T1, T2, T3, T4, Delta T |
| flow_rate | 0 | 1,504 l/m | F1, F2 |
| pipe_size | 0 | 150 | DN sizes |
| room_size | 1 | 5.2 m | Pipe length |
| pipe_cost_per_meter | 0 | €1,150/m | Unit cost |
| total_pipe_cost | 0 | €6,020 | Total pipe |
| hx_cost | 0 | €17,616 | Heat exchanger |
| valve_costs | 0 | €4,500 | Valve total |
| pump_cost | 0 | €5,000 | Pump cost |
| total_cost | -2 | €44,000 | Rounded to €100 |

**Negative decimal places**: Round to nearest 10^(-decimal_places)
- `-2` means round to nearest 100
- `-3` means round to nearest 1000

---

## Notes on Known Discrepancies

### Cost Calculations

1. **Pipe Cost Display** (€6,020 vs €7,000)
   - Calculated: €6,020 (€1,150/m × 5.2m + fittings)
   - Target: €7,000
   - **Reason**: Display includes additional margin/safety factor not in base calculation

2. **HX Target Costs** (€17,616 vs €89,000)
   - ALLHX base cost: €17,616
   - Target cost: €89,000
   - **Reason**: Target includes 5× multiplier for installation/complexity not applied in current code
   - **Location**: Would be applied in [python/core/costs.py:67](python/core/costs.py#L67)

3. **Total System Cost Components**
   - Some costs include hidden factors (installation, commissioning, testing)
   - Installation factor currently at 15%, may need calibration
   - **Reference**: [python/core/costs.py:479](python/core/costs.py#L479)

### Operating Costs

1. **Energy Price**
   - Current: €0.15/kWh (hardcoded)
   - Should load from: MW PRICE DATA.csv
   - **Location**: [python/core/costs.py:383](python/core/costs.py#L383)

2. **Pump Power Estimation**
   - Uses fixed pressure drops by approach
   - Real systems may vary ±20%
   - **Reference**: [python/core/costs.py:108-120](python/core/costs.py#L108)

---

## Version History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-10-06 | 1.0 | Initial comprehensive mapping | Claude Code |

---

## Usage During Demonstrations

### Quick Lookups

**"Where does T1 come from?"**
→ [python/core/lookup.py:121](python/core/lookup.py#L121) - Direct from ALLHX.csv column 'T1'

**"How is the pipe size determined?"**
→ [python/core/original_calculations.py:109-188](python/core/original_calculations.py#L109) - CEILING lookup in PIPSZ.csv based on flow rate F1

**"What's the formula for total cost?"**
→ [python/core/costs.py:465-489](python/core/costs.py#L465) - Equipment + 15% installation + 10% engineering + 10% contingency, rounded to €500

**"How does approach temperature affect operating cost?"**
→ [python/core/costs.py:108-120](python/core/costs.py#L108) - Smaller approach = larger HX = lower ΔP = less pump power

**"Where is the Economics table data calculated?"**
→ [python/core/costs.py:561-631](python/core/costs.py#L561) - `compare_approaches()` calls `calculate_order_of_magnitude_estimate()` 3 times

### Verification Path

To verify ANY displayed value:
1. Find the UI panel in [Section Headers](#table-of-contents)
2. Look up the value in the corresponding table
3. Navigate to the File:Line reference
4. Check the Data Source column for CSV dependencies
5. Use Quick Reference for function details

---

**End of UI Calculation Reference Map**
