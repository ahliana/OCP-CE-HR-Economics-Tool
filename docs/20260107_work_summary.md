# Heat Reuse Economics Tool - Development Summary

**Dates:** January 7-8, 2026
**Developer:** Ahliana Byrd
**Repository:** [OCP-CE-HR-Economics-Tool](https://github.com/opencomputeproject/OCP-CE-HR-Economics-Tool)

---

## Executive Summary

This sprint delivered **7 major enhancements** to the Heat Reuse Economics Tool, transforming it from a basic calculator into a comprehensive economic analysis platform with export capabilities and improved accessibility.

| Category | Deliverables |
|----------|--------------|
| **New Features** | Advanced Economic Analysis, CSV/PNG Export |
| **Bug Fixes** | Energy price calculation, display consistency |
| **Accessibility** | Dark mode support (WCAG 2.1 AA compliant) |
| **Code Quality** | DRY refactoring, unified cost calculations |

---

## New Features

### 1. Advanced Economic Analysis Module

**File:** `python/ui/advanced_economics.py` (~750 lines)

Comprehensive economic analysis enabling direct comparison of heat recovery costs against energy benchmarks.

#### Key Calculations

| Metric | Formula | Purpose |
|--------|---------|---------|
| Annualized Capital Cost | CapEx / Payback Period | Spread capital over time |
| Total Annualized Cost | Ann. CapEx + OpEx | Annual cost comparison |
| Normalized CapEx | CapEx / Capacity (MW) | Economy of scale analysis |
| **Unit Heat Recovery Cost** | Total Ann. / (MW x hrs x 1000) | **Compare to energy prices** |

#### Interactive Parameters

| Parameter | Options | Default |
|-----------|---------|---------|
| Payback Period | 5, 10, 15, 20 years | 5 years |
| On-stream Hours | 8760 (100%), 8000 (91%), 6000 (68%), 4000 (46%) | 8760 hrs |

#### Comparison Tables

| Table | Description |
|-------|-------------|
| **Table A** | Fixed capacity, variable approach (2C, 3C, 5C) |
| **Table B** | Fixed approach (3C), variable capacity (1-5 MW) |

#### Charts (Dual Y-Axis)

1. **Annual Costs vs. Approach** - Trade-off visualization
2. **Unit Cost vs. Approach** - With benchmark lines at 0.05/kWh (gas) and 0.15/kWh (electricity)
3. **Economy of Scale** - Unit cost reduction as capacity increases

#### Visual Highlights

- Gold star marks optimal configuration
- Green row highlighting for lowest cost option
- Auto-generated Key Economic Insights summary

---

### 2. Export Functionality

**File:** `python/ui/export.py` (~900 lines)

Save analysis results before Colab session disconnects.

#### Export Formats

| Format | Contents |
|--------|----------|
| **CSV** | All data in spreadsheet-compatible format |
| **PNG** | Complete visual report (11 sections) |

#### Cross-Platform Support

| Environment | Download Method |
|-------------|-----------------|
| Google Colab | `google.colab.files.download()` |
| Local Jupyter | Base64-encoded download links |

#### PNG Export Layout (11 Rows)

| Row | Content |
|-----|---------|
| 0 | System Parameters / Piping Cost Analysis |
| 1 | Economics Analysis (Order of Magnitude Estimate) |
| 2 | Equipment Cost Breakdown (3 pie charts: 2C, 3C, 5C) |
| 3 | Cost Contrast Analysis (Capital vs Operating) |
| 4 | System Approach Profiles / Effectiveness Gauge |
| 5 | Table A: Approach Temperature Comparison |
| 6 | Chart 1: Annual Costs / Chart 2: Unit Cost |
| 7 | Table B: Economy of Scale |
| 8 | Chart 3: Economy of Scale |
| 9 | Key Insights Summary |
| 10 | Benchmarks Footer |

#### CSV Export Sections

1. Current System Configuration (T1-T4, F1-F2, pipe sizing)
2. Cost Breakdown (equipment, installation, contingency)
3. Approach Temperature Comparison (2C, 3C, 5C with advanced metrics)
4. Economy of Scale Analysis (1-5 MW at 3C)
5. Energy Cost Benchmarks

#### Colab Optimization

- Reduced figure size for memory compatibility
- Auto-detect DPI (100 for Colab, 150 for local)
- Progress feedback during generation

---

## Accessibility Improvements

### 3. Dark Mode Support (WCAG 2.1 AA Compliant)

**File:** `python/ui/styles.py` (NEW)

**Problem:** Text was nearly invisible when users ran the tool in Google Colab with dark mode enabled.

**Solution:** High-contrast color palette with explicit backgrounds.

#### Color Palette

| Element | Color | Purpose |
|---------|-------|---------|
| Currency values | `#00C853` (bright green) | High visibility |
| Totals | `#7C4DFF` (bright purple) | Stand out |
| Labels | `#78909C` (medium gray) | Readable on any background |
| Headers | Gradient backgrounds | White text always visible |

#### Implementation

- Global CSS injection at notebook startup
- Explicit `background-color: #f8f9fa` on all containers
- Alternating row backgrounds in tables
- Text contrast ratio >= 4.5:1 (WCAG 2.1 AA)

---

## Bug Fixes & Improvements

### 4. Energy Price Calculation Fix

**Problem:** System was reading `MW Price Data.csv` (equipment costs) as electricity prices, calculating 16/kWh instead of 0.15/kWh.

**Solution:** Now uses hardcoded 0.15/kWh (reasonable European industrial rate).

---

### 5. Display Consistency Fixes

| Issue | Before | After |
|-------|--------|-------|
| Flow rate units | `l/m` | `L/min` |
| Pipe size format | `150` | `DN150` |
| Valve rounding | Inconsistent | Nearest 100 everywhere |

---

### 6. Piping Cost Analysis Refactor

- **Renamed** section from "Capital Cost Analysis" to "Piping Cost Analysis"
- **Removed** Heat Exchanger Cost, Pump Cost, TOTAL EQUIPMENT COST (moved to Economics)
- **Added** Fittings line with proper rounding (nearest 100)

---

### 7. DRY Principle Improvements

Refactored duplicate code in `original_calculations.py` to use shared functions from `costs.py`:

| Function | Before | After |
|----------|--------|-------|
| `calculate_fittings_cost()` | Inline `pipe_cost * 0.25` | JOINTS.csv lookup with 25% fallback |
| `calculate_valve_costs()` | Duplicated CVALV/IVALV lookup | Single source of truth |

---

## Files Changed

### New Files (2)

| File | Lines | Purpose |
|------|-------|---------|
| `python/ui/advanced_economics.py` | ~750 | Advanced economic analysis |
| `python/ui/export.py` | ~900 | CSV/PNG export functionality |
| `python/ui/styles.py` | ~200 | Dark mode styling |

### Modified Files (10)

| File | Changes |
|------|---------|
| `python/ui/inputs.py` | Added `advanced_economics` and `export` output areas |
| `python/ui/outputs.py` | Added display functions for new panels |
| `python/ui/formatting.py` | High-contrast styling, unit fixes |
| `python/ui/economics_panel.py` | Valve rounding, dark mode tables |
| `python/ui/interface.py` | Global CSS injection |
| `python/ui/config.py` | Section title, fittings rounding |
| `python/ui/__init__.py` | Module exports (version 1.2.0) |
| `python/core/costs.py` | Energy price fix |
| `python/core/original_calculations.py` | DRY refactor |
| `docs/UI_CALCULATION_MAP.md` | Section 7 documentation (version 2.3) |

---

## Data Verification

All data files confirmed as **European/metric**:

- Pipe sizes: DN format
- Temperatures: Celsius
- Flow rates: L/min
- Currency: Euro
- Lengths: meters

---

## Git Configuration

Configured dual-push to both repositories:

| Remote | Repository |
|--------|------------|
| origin (push 1) | `ahliana/OCP-CE-HR-Economics-Tool` |
| origin (push 2) | `opencomputeproject/OCP-CE-HR-Economics-Tool` |
| upstream | OCP main repo (for pulling PRs) |

---

## Outstanding Questions

### 1. Pump Cost Discrepancy

Two different calculations exist:

| Location | Formula |
|----------|---------|
| `costs.py` | 35,000 or 45,000 based on approach |
| `original_calculations.py` | `wha x 5,000` |

**Question:** Which is correct? Should we unify?

### 2. Room Size / Area Ambiguity

`ROOM.csv` values (1 MW = 5.2, 2 MW = 6.0, 5 MW = 17.5):

- Is this floor area (m^2) or pipe run length (m)?
- Values seem small for floor space but reasonable for pipe lengths
- Should UI label say "Room Size" or "Pipe Length"?

### 3. Data Quality (Acknowledged)

| Data | Status |
|------|--------|
| Pipe costs | "Fictional (Dubai/India sourced)" - needs European quotes |
| Pump costs | Hardcoded estimates, not from actual data |

---

## Version Summary

| Component | Version |
|-----------|---------|
| `python/ui/__init__.py` | 1.2.0 |
| `docs/UI_CALCULATION_MAP.md` | 2.3.0 |

---

*Document prepared for OCP CE Heat Reuse Working Group meeting - January 8, 2026*
