# Heat Reuse Economics Tool - UI Calculation Map

**Version:** 2.2.0
**Last Updated:** 2026-01-08
**Author:** Ahliana Byrd

This document explains **every number displayed in the UI** - where it comes from, how it's calculated, and what data sources are used. Written for engineers who work with heat exchangers and CDUs, not software developers.

---

## Quick Navigation

| Section | What You'll Find |
|---------|------------------|
| [1. User Inputs](#1-user-inputs) | What the 4 dropdown selections mean |
| [2. System Parameters](#2-system-parameters-section) | T1-T4, F1-F2, Delta T calculations |
| [3. Piping Cost Analysis](#3-piping-cost-analysis-section) | Pipe sizing, costs, fittings, valves |
| [4. Economics Analysis Table](#4-economics-analysis-table) | 3-column comparison (2°C, 3°C, 5°C) |
| [5. Operating Costs](#5-operating-costs) | Annual energy consumption and costs |
| [6. Charts](#6-charts) | What each visualization shows |
| [7. Advanced Economic Analysis](#7-advanced-economic-analysis) | Unit cost, annualized costs, economy of scale |
| [8. Data Files](#8-data-sources) | CSV files and what's in them |
| [9. Limitations](#9-known-limitations) | What's NOT included |

---

## 1. User Inputs

You select 4 values from dropdown menus. These drive everything else.

### What Each Input Means

| Dropdown Label | Variable | Valid Values | Engineering Meaning |
|----------------|----------|--------------|---------------------|
| **Power/Capacity (MW)** | wha | 1, 2, 3, 4, 5 | Heat exchanger thermal capacity |
| **T1 Temperature (°C)** | T1 | 20, 30, 45 | CDU outlet temperature (cold side leaving HX) |
| **Temperature Rise (°C)** | itdt | 10, 12, 14 | How much the water heats up (T2 - T1) |
| **Approach (°C)** | approach | 2, 3, 4, 5 | Temperature difference between cold outlet and hot inlet (T4 - T1) |

### How Inputs Work Together

```
Your selections → ALLHX.csv lookup → Complete system configuration

Example:
  1 MW + 20°C + 10°C rise + 2°C approach
    ↓
  ALLHX.csv returns: T2=30°C, T3=28°C, T4=22°C, F1=1,434 L/min, F2=1,434 L/min, HX cost=€17,616
```

**Why only certain values?** The ALLHX.csv data file contains pre-calculated configurations for specific combinations. Values outside these options have no data.

---

## 2. System Parameters Section

**Green border** | Title: "System Parameters (Auto-Calculated)"

### 2.1 Temperature Points

These come directly from the ALLHX.csv lookup - no calculation needed.

| Display Label | What It Is | Source |
|---------------|------------|--------|
| **T1 (Outlet to TCS)** | Water leaving HX to data center | User input (dropdown) |
| **T2 (Inlet from TCS)** | Warm water returning from data center | ALLHX.csv (= T1 + temp rise) |
| **T3 (Outlet to Consumer)** | Hot water going to building/district heating | ALLHX.csv |
| **T4 (Inlet from Consumer)** | Return water from building | ALLHX.csv (= T1 + approach) |

**Temperature Flow Diagram:**
```
DATA CENTER SIDE (TCS)              BUILDING SIDE (FWS)
    T1 ←──────────────────────────────── T4
    (cold out)     HEAT              (cold return)
                 EXCHANGER
    T2 ──────────────────────────────→ T3
    (warm in)                         (hot out)
```

### 2.2 Flow Rates

| Display Label | What It Is | Source | Units |
|---------------|------------|--------|-------|
| **F1 (TCS Flow Rate)** | Water flow on data center side | ALLHX.csv | L/min |
| **F2 (FWS Flow Rate)** | Water flow on building side | ALLHX.csv | L/min |

**How Flow Rate Relates to Power:**
```
Power (MW) = Flow Rate (kg/s) × Specific Heat (4,186 J/kg·K) × Temperature Rise (°C)

Rearranged:
Flow Rate = Power / (4,186 × ΔT × density)
```

For a **1 MW** system with **10°C** temperature rise:
- Flow ≈ 1,000,000 / (4,186 × 10 × 995) ≈ **24 L/s** ≈ **1,440 L/min**

### 2.3 Delta T Values (Calculated)

These are calculated from the temperature points, not looked up.

| Display Label | Formula | What It Means |
|---------------|---------|---------------|
| **Delta T for TCS (IT Medium)** | T2 - T1 | Temperature RISE on data center side |
| **Delta T for FWS (Heating Medium)** | T3 - T4 | Temperature DROP on building side |

**Example:**
- T1 = 20°C, T2 = 30°C → **Delta T TCS = 10°C** (water heated by 10°C)
- T3 = 28°C, T4 = 22°C → **Delta T FWS = 6°C** (water cooled by 6°C)

---

## 3. Piping Cost Analysis Section

**Blue border** | Title: "Piping Cost Analysis"

### 3.1 Room Size

| Display | Source | Meaning |
|---------|--------|---------|
| e.g., "5.2 m²" | ROOM.csv | Pipe run length for the system |

**Lookup Method:** Find the first row where MW capacity ≥ your system MW.

**Note:** The values in ROOM.csv (5.2, 6.0, 17.5 for 1, 2, 5 MW) seem small for room floor area but reasonable for typical pipe run lengths. Clarification needed on actual meaning.

### 3.2 Suggested Pipe Size

| Display | Source | Method |
|---------|--------|--------|
| e.g., "DN150" | PIPSZ.csv | Ceiling match based on flow rate |

**How It Works (CEILING Method):**
1. Take your flow rate (F1)
2. Find the **smallest** pipe in PIPSZ.csv that can handle that flow
3. Return that DN size

**Example:**
- F1 = 1,434 L/min
- PIPSZ.csv has: 1000 L/min → DN100, 1500 L/min → DN150, 2500 L/min → DN200
- Result: **DN150** (first size that handles ≥1,434 L/min)

**Why CEILING?** Engineering safety - always size up, never down.

### 3.3 Pipe Cost per Meter

| Display | Source | Material |
|---------|--------|----------|
| e.g., "€216/m" | PIPCOST.csv | Stainless Steel |

**Lookup Steps:**
1. Get DN size from previous step
2. Convert DN to American inch equivalent (DN150 → 6")
3. Look up cost in PIPCOST.csv for that inch size

**DN to Inch Mapping Used:**
| DN Size | American Inches | Typical Cost Range |
|---------|-----------------|-------------------|
| DN100 | 4" | €150-180/m |
| DN150 | 6" | €200-250/m |
| DN200 | 8" | €280-350/m |

### 3.4 Total Pipe Cost

| Display | Formula | Rounding |
|---------|---------|----------|
| e.g., "€1,000" | Cost per meter × Pipe length | Nearest €1,000 |

```
Total Pipe Cost = €216/m × 5.2m = €1,123 → Rounded to €1,000
```

### 3.5 Fittings Cost

| Display | Calculation | Rounding |
|---------|-------------|----------|
| e.g., "€300" | 25% of pipe cost OR JOINTS.csv lookup | Nearest €100 |

**Logic:**
1. **Try first:** Look up fitting cost from JOINTS.csv by pipe size
2. **If no match:** Use 25% of total pipe cost (industry rule of thumb)

**Why 25%?** Standard engineering estimate - fittings typically add 20-30% to pipe material cost.

### 3.6 Valve Costs

| Display | Calculation | Rounding |
|---------|-------------|----------|
| e.g., "€7,300" | (1 × Control Valve) + (4 × Isolation Valve) | Nearest €100 |

**Valve Count Assumption:**
- **1 control valve** (modulating, for flow regulation)
- **4 isolation valves** (on/off, for maintenance isolation)

**Lookup Sources:**
| Valve Type | Data File | Cost Driver |
|------------|-----------|-------------|
| Control Valve | CVALV.csv | Pipe DN size |
| Isolation Valve | IVALV.csv | Pipe DN size |

---

## 4. Economics Analysis Table

**Section Title:** "Economics Analysis"
**Compares 3 scenarios:** 2°C approach, 3°C approach, 5°C approach

### 4.1 Understanding the Table Structure

```
                        │   2°C    │   3°C    │   5°C   │
─────────────────────────┼──────────┼──────────┼─────────┤
EQUIPMENT COSTS          │          │          │         │
  Heat Exchanger         │ €17,616  │ €13,500  │ €10,000 │
  Pumps                  │ €35,000  │ €35,000  │ €45,000 │
  Pipe & Fittings        │ €14,094  │ €14,094  │ €14,094 │
  Instrumentation        │ €30,000  │ €30,000  │ €30,000 │
  Valves                 │  €7,310  │  €7,310  │  €7,310 │
  Equipment Subtotal     │ €104,020 │ €99,904  │€106,404 │
─────────────────────────┼──────────┼──────────┼─────────┤
INSTALLATION & CONTINGENCY│          │          │         │
  Installation (15%)     │ €15,603  │ €14,986  │ €15,961 │
  Engineering (10%)      │ €11,962  │ €11,489  │ €12,237 │
  Contingency (10%)      │ €13,159  │ €12,638  │ €13,460 │
  I&C Subtotal           │ €40,724  │ €39,113  │ €41,658 │
─────────────────────────┼──────────┼──────────┼─────────┤
CAPITAL TOTAL            │€144,744  │€139,017  │€148,062 │
─────────────────────────┼──────────┼──────────┼─────────┤
OPERATING COSTS (Annual) │          │          │         │
  Operating Energy       │ 9,545 kWh│11,053 kWh│15,768 kWh│
  Annual Energy Cost     │  €1,432  │  €1,658  │  €2,365 │
```

### 4.2 Equipment Costs Explained

#### Heat Exchanger Cost

| Approach | Typical Cost | Why? |
|----------|--------------|------|
| 2°C | Higher (€17,616) | Smaller approach needs LARGER HX surface area |
| 3°C | Medium (€13,500) | Balanced trade-off |
| 5°C | Lower (€10,000) | Larger approach allows SMALLER HX |

**Source:** Direct lookup from ALLHX.csv column 'hx_cost'

#### Pump Cost

| Approach | Cost | Pressure Drop | Why? |
|----------|------|---------------|------|
| 2°C | €35,000 | 0.30 bar (30 kPa) | Larger HX = lower flow resistance |
| 3°C | €35,000 | 0.35 bar (35 kPa) | Medium resistance |
| 5°C | €45,000 | 0.50 bar (50 kPa) | Smaller HX = higher flow resistance |

**Hardcoded values** - not from CSV lookup.

#### Pipe & Fittings

Same for all approaches (depends on flow rate, not approach temperature):
- Pipe cost from PIPCOST.csv
- Fittings = 25% of pipe cost or JOINTS.csv lookup

#### Instrumentation

| System Size | Cost | Formula |
|-------------|------|---------|
| ≤ 2 MW | €30,000 | Base cost |
| > 2 MW | €30,000 + €5,000/MW | Base + €5K per MW above 2 |

**Example:** 5 MW system = €30,000 + (5-2) × €5,000 = **€45,000**

#### Valves

Same for all approaches (depends on pipe size, not approach):
- 1 control valve (CVALV.csv)
- 4 isolation valves (IVALV.csv)

### 4.3 Installation & Contingency (I&C)

Applied **cumulatively**, not independently:

| Factor | Percentage | Base | Formula |
|--------|------------|------|---------|
| Installation | 15% | Equipment Subtotal | Equipment × 1.15 |
| Engineering | 10% | Equipment + Installation | (Equip + Install) × 1.10 |
| Contingency | 10% | All previous | (Equip + Install + Eng) × 1.10 |

**Total multiplier:** 1.15 × 1.10 × 1.10 = **1.3915** (≈39% markup)

**Example:**
```
Equipment Subtotal = €104,020
Installation = €104,020 × 0.15 = €15,603
Engineering = (€104,020 + €15,603) × 0.10 = €11,962
Contingency = (€104,020 + €15,603 + €11,962) × 0.10 = €13,159
─────────────────────────────────────────────────────────
CAPITAL TOTAL = €144,744
```

### 4.4 The Approach Temperature Trade-off

| Factor | 2°C (small approach) | 5°C (large approach) |
|--------|----------------------|----------------------|
| Heat Exchanger | **Larger, more expensive** | Smaller, cheaper |
| Pumps | Smaller, cheaper | **Larger, more expensive** |
| Operating Cost | **Lower** (less pump power) | Higher (more pump power) |

**Key Insight:** Lower approach temperature = higher CapEx but lower OpEx. Engineering economics dictates the optimal choice based on energy costs and system lifetime.

---

## 5. Operating Costs

### 5.1 What's Included

| Cost Type | Included? | Calculation |
|-----------|-----------|-------------|
| Pump electricity | YES | Pump power × 8,760 hours × €0.15/kWh |
| Maintenance | **NO** | Should be ~3-5% of CapEx/year |
| Labor | **NO** | Not calculated |
| Water treatment | **NO** | Not calculated |

### 5.2 Annual Operating Energy Calculation

```
Annual Energy (kWh) = Pump Power (kW) × 8,760 hours/year

Where Pump Power = (Flow Rate × Pressure Drop) / (Pump Efficiency × Motor Efficiency)
```

**Efficiency Assumptions:**
- Pump hydraulic efficiency: 75%
- Motor efficiency: 92%
- Combined: 69%

**Pressure Drop by Approach:**
| Approach | Pressure Drop | Pump Power (1MW system) | Annual Energy |
|----------|---------------|-------------------------|---------------|
| 2°C | 30,000 Pa | 1.09 kW | 9,545 kWh |
| 3°C | 35,000 Pa | 1.27 kW | 11,053 kWh |
| 5°C | 50,000 Pa | 1.80 kW | 15,768 kWh |

### 5.3 Annual Energy Cost

```
Annual Cost = Annual Energy (kWh) × €0.15/kWh
```

**Current electricity price:** €0.15/kWh (European industrial average)

| Approach | Annual Energy | Annual Cost |
|----------|---------------|-------------|
| 2°C | 9,545 kWh | €1,432 |
| 3°C | 11,053 kWh | €1,658 |
| 5°C | 15,768 kWh | €2,365 |

### 5.4 What You Should Add for Total Cost of Ownership

For realistic total operating costs, add:

| Cost Type | Typical Range | For €155K CapEx |
|-----------|---------------|-----------------|
| Maintenance | 3-5% of CapEx/year | €4,650 - €7,750/year |
| Insurance | 0.5-1% of CapEx/year | €775 - €1,550/year |
| Water treatment | €500-2,000/year | €500-2,000/year |

**More realistic annual operating cost:** €7,000-12,000/year (not €1,500)

---

## 6. Charts

### 6.1 Cost Contrast Chart (Line Graph)

**Shows:** Capital cost vs. Annual operating cost across all 3 approaches

| Line | Color | Data Points |
|------|-------|-------------|
| Capital Cost | Blue | [€144K, €139K, €148K] |
| Annual Operating | Orange | [€1,432, €1,658, €2,365] |

**Key Insight:** Capital costs are relatively flat (~5% variation), but operating costs vary significantly (~65% from lowest to highest).

### 6.2 Cost Breakdown Pie Charts (3 charts)

**Shows:** How total capital cost breaks down for each approach

**Slices:**
1. Heat Exchangers (varies by approach)
2. Pumps (varies by approach)
3. Pipe & Fittings (constant)
4. Instrumentation (constant)
5. Valves (constant)
6. Installation (15% of equipment)
7. Engineering (10% cumulative)
8. Contingency (10% cumulative)

### 6.3 Temperature Profile Chart

**Shows:** Temperature progression through the heat exchanger

### 6.4 Effectiveness Gauge

**Shows:** Heat exchanger effectiveness as a percentage

**Calculation:**
```
Effectiveness = Actual Heat Transfer / Maximum Possible Heat Transfer
             = (T2 - T1) / (T3 - T1)
```

---

## 7. Advanced Economic Analysis

**Section Title:** "Advanced Economic Analysis"
**Toggle:** `SHOW_ADVANCED_ECONOMICS = True` in `python/ui/advanced_economics.py` (line 27)
**File:** `python/ui/advanced_economics.py`

This section provides deeper economic insights including annualized costs, unit heat recovery cost (€/kWh), and economy of scale comparisons.

### 7.1 Calculations

#### Annualized Capital Cost

| Formula | Example |
|---------|---------|
| CapEx ÷ Payback Period | €1,351,000 ÷ 5 years = **€270,200/year** |

**Default Payback Period:** 5 years (hardcoded in `advanced_economics.py`)

#### Total Annualized Cost

| Formula | Example |
|---------|---------|
| Annualized CapEx + Annual OpEx | €270,200 + €50,000 = **€320,200/year** |

This is the key metric for comparing different system configurations on an annual basis.

#### Normalized Capital Cost (€/MW)

| Formula | Purpose |
|---------|---------|
| CapEx ÷ Capacity (MW) | Economy of scale comparison |

**Example - Economy of Scale:**
| System Size | CapEx | Normalized Cost | Better Value? |
|-------------|-------|-----------------|---------------|
| 1 MW | €1,000,000 | €1,000,000/MW | ❌ |
| 5 MW | €3,500,000 | €700,000/MW | ✅ |

#### Unit Heat Recovery Cost (€/kWh)

| Formula | What It Tells You |
|---------|-------------------|
| Total Annualized Cost ÷ (MW × 1000 × 8760) | Cost per kWh of recovered heat |

**This is "the most interesting number"** - it lets you compare heat recovery cost directly against energy prices.

**Benchmark Comparisons:**
| Unit Cost | Interpretation |
|-----------|----------------|
| < €0.05/kWh | ✅ Competitive with natural gas |
| < €0.15/kWh | ✅ Competitive with EU electricity |
| > €0.15/kWh | ⚠️ Above typical energy benchmarks |

**Example Calculation:**
```
Total Annualized Cost = €320,200/year
Capacity = 2 MW
Annual Energy Potential = 2 MW × 1000 kW/MW × 8760 hrs = 17,520,000 kWh

Unit Cost = €320,200 / 17,520,000 kWh = €0.0183/kWh
```
Result: **€0.018/kWh** - significantly cheaper than natural gas!

### 7.2 Comparison Tables

#### Table A: Fixed Capacity, Variable Approach

Shows how approach temperature affects economics for a single system size.

| Column | Meaning |
|--------|---------|
| Capacity (MW) | Same for all rows (your selected capacity) |
| Approach (°C) | 2°C, 3°C, 5°C |
| CapEx (K€) | Total capital cost |
| OpEx (K€/yr) | Annual operating cost |
| Annualized CapEx (K€/yr) | CapEx ÷ 5 years |
| Total Ann. Cost (K€/yr) | Sum of annualized CapEx + OpEx |
| Normalized CapEx (K€/MW) | CapEx per MW capacity |
| Heat Recovery Cost (€/kWh) | **Key metric** for comparison |

**Optimal row highlighted in green** = lowest Total Annualized Cost

#### Table B: Fixed Approach, Variable Capacity

Shows economy of scale - how unit cost decreases with larger systems.

Uses 3°C approach (default) across capacities 1-5 MW.

### 7.3 Charts

#### Chart 1: Annual Costs vs. Approach Temperature

**Dual Y-Axis Chart:**
- Left axis (lines): OpEx, Annualized CapEx, Total Annualized Cost
- Right axis (bars): Normalized CapEx (K€/MW)
- Gold star marks the optimal point (lowest total cost)

**Key Insight:** Shows the trade-off between CapEx and OpEx as approach changes.

#### Chart 2: Unit Heat Recovery Cost vs. Approach Temperature

**Dual Y-Axis Chart:**
- Left axis (line): Heat Recovery Cost (€/kWh)
- Right axis (bars): Normalized CapEx (K€/MW)
- Benchmark lines at €0.05 (natural gas) and €0.15 (EU electricity)

**Key Insight:** Shows whether your heat recovery is economically competitive.

#### Chart 3: Economy of Scale

**Dual Y-Axis Chart:**
- Left axis (line): Unit Heat Recovery Cost vs. Capacity
- Right axis (bars): Normalized CapEx (K€/MW)
- Annotation shows % cost reduction from 1 MW to 5 MW

**Key Insight:** Larger systems have significantly lower unit costs.

### 7.4 Key Economic Insights Summary

Auto-generated insights at the bottom of the section:
- Optimal approach temperature for current capacity
- Unit cost at optimal point
- Economy of scale percentage (1→5 MW)
- Competitiveness vs. energy benchmarks

### 7.5 Assumptions & Limitations

| Assumption | Value | Notes |
|------------|-------|-------|
| Payback period | 5 years | Hardcoded, not adjustable in UI |
| Operating hours | 8,760/year | Assumes 100% on-stream (24/7) |
| Electricity price | €0.15/kWh | Used for OpEx calculation |

**What's NOT included in OpEx:**
- Maintenance (typically 3-5% of CapEx/year)
- Labor costs
- Water treatment
- Insurance

---

## 8. Data Sources

### 8.1 CSV Data Files

All located in `/Data/` folder:

| File | Purpose | Key Columns |
|------|---------|-------------|
| **ALLHX.csv** | Master lookup for HX configurations | wha, T1, itdt, approach → T2, T3, T4, F1, F2, hx_cost |
| **PIPSZ.csv** | Flow rate to pipe size | Flow (L/min) → DN size |
| **PIPCOST.csv** | Pipe cost by size | Size → €/meter (Stainless, Carbon) |
| **ROOM.csv** | Pipe length by MW | MW → Length (m) |
| **CVALV.csv** | Control valve costs | DN size → € |
| **IVALV.csv** | Isolation valve costs | DN size → € |
| **JOINTS.csv** | Fittings costs | DN size → € per fitting |

### 8.2 Hardcoded Values

| Value | Location | Current Setting | Notes |
|-------|----------|-----------------|-------|
| Electricity price | costs.py | €0.15/kWh | European industrial avg |
| Installation % | costs.py | 15% | Of equipment subtotal |
| Engineering % | costs.py | 10% | Cumulative |
| Contingency % | costs.py | 10% | Cumulative |
| Base instrumentation | costs.py | €30,000 | For ≤2 MW systems |
| Pump cost (2°C/3°C) | costs.py | €35,000 | Fixed estimate |
| Pump cost (5°C) | costs.py | €45,000 | Fixed estimate |
| Isolation valve count | costs.py | 4 | Per system |
| Pump efficiency | costs.py | 75% | Hydraulic |
| Motor efficiency | costs.py | 92% | Electric |
| Operating hours | costs.py | 8,760 | 24/7 operation |

---

## 9. Known Limitations

### 9.1 Data Quality Issues

| Issue | Impact | Notes |
|-------|--------|-------|
| Pipe costs | May be outdated | Original source: Dubai/India suppliers |
| Pump costs | Estimates only | Hardcoded, not from quotes |
| Room size values | Unclear units | May be m² or linear meters |

### 9.2 What's NOT in Operating Costs

| Missing Item | Typical Cost | Why It Matters |
|--------------|--------------|----------------|
| **Maintenance** | 3-5% of CapEx/year | €4,650-7,750/year for a €155K system |
| **Labor** | Varies | Inspection, monitoring |
| **Water treatment** | €500-2,000/year | Chemicals, testing |
| **Insurance** | 0.5-1% of CapEx/year | Asset protection |

### 9.3 Simplifications in Calculations

1. **Pump sizing** - Uses estimated pressure drops, not actual hydraulic calculations
2. **Fittings count** - Assumes 20 fittings per system (may vary)
3. **Pipe sizing** - CEILING method may oversize slightly
4. **Heat exchanger selection** - Based on lookup tables, not custom engineering

---

## Quick Reference: Where Each Number Comes From

| UI Display | Source Type | Data Source |
|------------|-------------|-------------|
| T1, T2, T3, T4 | Lookup | ALLHX.csv |
| F1, F2 | Lookup | ALLHX.csv |
| Delta T TCS | Formula | T2 - T1 |
| Delta T FWS | Formula | T3 - T4 |
| Room Size | Lookup | ROOM.csv |
| Pipe Size | Lookup | PIPSZ.csv |
| Pipe Cost/m | Lookup | PIPCOST.csv |
| Total Pipe Cost | Formula | €/m × length |
| Fittings | Formula | 25% of pipe OR JOINTS.csv |
| Valve Costs | Lookup | CVALV.csv + IVALV.csv |
| Heat Exchanger | Lookup | ALLHX.csv |
| Pumps | **Hardcoded** | €35K or €45K |
| Instrumentation | Formula | €30K + €5K/MW over 2 |
| Installation | Formula | Equipment × 15% |
| Engineering | Formula | (Equip+Install) × 10% |
| Contingency | Formula | (Equip+Install+Eng) × 10% |
| Operating Energy | Formula | Pump kW × 8760 hrs |
| Operating Cost | Formula | kWh × €0.15 |

---

## Glossary

| Term | Meaning |
|------|---------|
| **Approach Temperature** | Difference between cold fluid outlet (T1) and hot fluid inlet (T4). Smaller approach = larger HX needed |
| **CapEx** | Capital Expenditure - upfront equipment and installation cost |
| **CEILING lookup** | Find the first value in a table that's ≥ your target value |
| **DN** | Diameter Nominal - European pipe sizing standard (e.g., DN100 ≈ 4" pipe) |
| **FWS** | Facility Water System - the building/consumer side of the heat exchanger |
| **I&C** | Installation and Contingency - markup factors added to equipment cost |
| **itdt** | Internal Temperature Delta T - temperature rise on the TCS side (T2 - T1) |
| **OpEx** | Operating Expenditure - ongoing costs (energy, maintenance, labor) |
| **TCS** | Technology Cooling System - the data center side of the heat exchanger |
| **wha** | Power capacity in MW (Watts Heat Absorbed) |

---

## Technical Reference: Function Locations

**For developers and maintainers** - exact file and function locations for each calculation.

### System Parameters (Temperature & Flow)

| UI Value | Function | File | Line |
|----------|----------|------|------|
| T1 (user input) | Dropdown selection | `python/ui/config.py` | ~25 |
| T2, T3, T4, F1, F2 | `get_itdt()` | `python/core/original_calculations.py` | ~45-80 |
| Delta T TCS | `T2 - T1` inline | `python/ui/formatting.py` | ~180 |
| Delta T FWS | `T3 - T4` inline | `python/ui/formatting.py` | ~181 |

### Piping Costs

| UI Value | Function | File | Line |
|----------|----------|------|------|
| Room/Pipe Length | `get_PipeLength()` | `python/core/original_calculations.py` | ~95-115 |
| Suggested Pipe Size | `get_PipeSize_Suggested()` | `python/core/original_calculations.py` | ~120-145 |
| Pipe Cost per Meter | `get_PipeCost_Total()` → reads PIPCOST.csv | `python/core/original_calculations.py` | ~150-180 |
| Total Pipe Cost | `get_PipeCost_Total()` | `python/core/original_calculations.py` | ~150-180 |
| Fittings Cost | `get_PipeCost_Total()` → `fittings_cost` key | `python/core/original_calculations.py` | ~175 |
| Control Valve Cost | `calculate_valve_costs()` | `python/core/costs.py` | ~180-220 |
| Isolation Valve Cost | `calculate_valve_costs()` | `python/core/costs.py` | ~180-220 |
| Total Valve Cost | `calculate_valve_costs()` → sum | `python/core/costs.py` | ~215 |

### Equipment Costs (Economics Table)

| UI Value | Function | File | Line |
|----------|----------|------|------|
| Heat Exchanger Cost | `get_itdt()` → `hx_cost` key | `python/core/original_calculations.py` | ~75 |
| Pump Cost | `calculate_pump_cost()` | `python/core/costs.py` | ~250-300 |
| Pipe & Fittings | `get_PipeCost_Total()` | `python/core/original_calculations.py` | ~150-180 |
| Instrumentation | `calculate_instrumentation_cost()` | `python/core/costs.py` | ~140-175 |
| Valves | `calculate_valve_costs()` | `python/core/costs.py` | ~180-220 |
| Equipment Subtotal | Sum in `build_approaches_data()` | `python/ui/economics_panel.py` | ~140-150 |

### Installation & Contingency

| UI Value | Function | File | Line |
|----------|----------|------|------|
| Installation (15%) | `calculate_contingency_costs()` | `python/core/costs.py` | ~320-355 |
| Engineering (10%) | `calculate_contingency_costs()` | `python/core/costs.py` | ~320-355 |
| Contingency (10%) | `calculate_contingency_costs()` | `python/core/costs.py` | ~320-355 |
| Capital Total | `calculate_total_system_cost()` | `python/core/costs.py` | ~380-410 |

### Operating Costs

| UI Value | Function | File | Line |
|----------|----------|------|------|
| Pump Power (kW) | `calculate_pump_cost()` → `power_kw` | `python/core/costs.py` | ~285 |
| Annual Energy (kWh) | `calculate_operating_energy()` | `python/core/costs.py` | ~360-405 |
| Annual Cost (€) | `calculate_operating_energy()` → `annual_cost` | `python/core/costs.py` | ~395 |
| Electricity Price | Hardcoded `0.15` €/kWh | `python/core/costs.py` | ~390 |

### Display Functions (HTML Generation)

| UI Section | Display Function | File | Line |
|------------|------------------|------|------|
| System Parameters panel | `display_system_params()` | `python/ui/outputs.py` | ~45-90 |
| Piping Cost Analysis | `format_cost_table()` | `python/ui/formatting.py` | ~50-130 |
| Economics Analysis table | `create_economics_table()` | `python/ui/economics_panel.py` | ~50-325 |
| Charts (all) | `display_charts()` | `python/ui/outputs.py` | ~200-280 |
| Summary Cards | `display_visual_summary()` | `python/ui/outputs.py` | ~480-525 |
| Advanced Economic Analysis | `display_advanced_economics()` | `python/ui/advanced_economics.py` | ~459-550 |

### Advanced Economic Analysis Functions

| UI Value | Function | File | Line |
|----------|----------|------|------|
| Annualized CapEx | `calculate_advanced_metrics()` | `python/ui/advanced_economics.py` | ~68 |
| Total Annualized Cost | `calculate_advanced_metrics()` | `python/ui/advanced_economics.py` | ~71 |
| Normalized CapEx (€/MW) | `calculate_advanced_metrics()` | `python/ui/advanced_economics.py` | ~74 |
| Unit Heat Recovery Cost (€/kWh) | `calculate_advanced_metrics()` | `python/ui/advanced_economics.py` | ~79 |
| Approach Comparison Data | `generate_approach_comparison_data()` | `python/ui/advanced_economics.py` | ~97-130 |
| Capacity Comparison Data | `generate_capacity_comparison_data()` | `python/ui/advanced_economics.py` | ~133-166 |
| Comparison Tables | `create_comparison_table()` | `python/ui/advanced_economics.py` | ~173-248 |
| Annual Costs Chart | `create_annual_costs_chart()` | `python/ui/advanced_economics.py` | ~255-318 |
| Unit Cost Chart | `create_unit_cost_chart()` | `python/ui/advanced_economics.py` | ~321-385 |
| Economy of Scale Chart | `create_economy_of_scale_chart()` | `python/ui/advanced_economics.py` | ~388-452 |
| Key Insights Summary | `create_insights_summary()` | `python/ui/advanced_economics.py` | ~561-608 |
| Toggle Check | `should_show_advanced_economics()` | `python/ui/advanced_economics.py` | ~636 |

### Event Handlers

| Action | Handler Function | File | Line |
|--------|------------------|------|------|
| Calculate button click | `handle_calculate()` | `python/ui/handlers.py` | ~35-150 |
| Full calculation pipeline | `run_full_calculation()` | `python/ui/handlers.py` | ~155-250 |

### Data File Loaders

| Data Source | Loader Function | File |
|-------------|-----------------|------|
| ALLHX.csv | `load_allhx_data()` | `python/core/lookup.py` |
| PIPSZ.csv | `get_PipeSize_Suggested()` | `python/core/original_calculations.py` |
| PIPCOST.csv | `get_PipeCost_Total()` | `python/core/original_calculations.py` |
| ROOM.csv | `get_PipeLength()` | `python/core/original_calculations.py` |
| CVALV.csv | `calculate_valve_costs()` | `python/core/costs.py` |
| IVALV.csv | `calculate_valve_costs()` | `python/core/costs.py` |
| JOINTS.csv | `get_PipeCost_Total()` | `python/core/original_calculations.py` |

### Key Constants & Hardcoded Values

| Value | Location | Variable/Context |
|-------|----------|------------------|
| Electricity €0.15/kWh | `python/core/costs.py:390` | `energy_price = 0.15` |
| Installation 15% | `python/core/costs.py:325` | `installation_rate = 0.15` |
| Engineering 10% | `python/core/costs.py:326` | `engineering_rate = 0.10` |
| Contingency 10% | `python/core/costs.py:327` | `contingency_rate = 0.10` |
| Base instrumentation €30K | `python/core/costs.py:145` | `base_cost = 30000` |
| Pump efficiency 75% | `python/core/costs.py:275` | `pump_efficiency = 0.75` |
| Motor efficiency 92% | `python/core/costs.py:276` | `motor_efficiency = 0.92` |
| Operating hours 8760 | `python/core/costs.py:365` | `operating_hours = 8760` |
| Isolation valve count 4 | `python/core/costs.py:195` | `num_isolation = 4` |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-06 | Initial creation |
| 1.1 | 2025-10-08 | Added pie charts documentation |
| 2.0 | 2026-01-07 | Complete rewrite for engineer audience - added formulas, trade-off explanations, limitations section, glossary |
| 2.1 | 2026-01-07 | Added Technical Reference section with exact function locations for maintainers |
| 2.2 | 2026-01-08 | Added Section 7: Advanced Economic Analysis - annualized costs, unit heat recovery cost (€/kWh), economy of scale charts, benchmark comparisons |
