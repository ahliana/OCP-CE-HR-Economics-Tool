# Heat Reuse Tool - Visual Styling Options for Light/Dark Mode
## Research Summary for Claude Code Implementation

**Problem:** Tool looks good on light backgrounds but is unreadable on dark mode in Google Colab. Black text renders as faint gray, totals/subtotals invisible, equipment cost lines nearly impossible to see.

**Goal:** Professional, visually appealing interface that works in BOTH light and dark modes while meeting accessibility standards.

---

## OPTION 1: CSS Injection with Theme Detection (RECOMMENDED)

### How It Works
Inject custom CSS via `IPython.display.HTML()` that uses CSS media queries to automatically adapt to the user's theme preference.

### Implementation
```python
from IPython.display import HTML, display

def inject_adaptive_styles():
    """Inject CSS that adapts to light/dark mode automatically"""
    css = """
    <style>
    /* Force our own background to ensure readability */
    .heat-reuse-output {
        background-color: #ffffff;
        color: #1a1a1a;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Dark mode override using media query */
    @media (prefers-color-scheme: dark) {
        .heat-reuse-output {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        .heat-reuse-output .highlight {
            color: #4fc3f7;
        }
        .heat-reuse-output .total-line {
            color: #81c784;
            font-weight: bold;
        }
    }
    
    /* Light mode explicit styles */
    @media (prefers-color-scheme: light) {
        .heat-reuse-output {
            background-color: #ffffff;
            color: #1a1a1a;
        }
        .heat-reuse-output .highlight {
            color: #0277bd;
        }
        .heat-reuse-output .total-line {
            color: #2e7d32;
            font-weight: bold;
        }
    }
    </style>
    """
    display(HTML(css))
```

### Pros
- Automatic detection - no user action required
- Follows system/browser preferences
- Clean, modern approach

### Cons
- Media query may not detect Colab's internal theme setting (only OS-level)
- Requires wrapping all output in custom div classes

---

## OPTION 2: Explicit Background Container (MOST RELIABLE)

### How It Works
Force a specific background color on all tool outputs, ensuring consistent contrast regardless of Colab's theme.

### Implementation
```python
from IPython.display import HTML, display

def create_styled_output(content_html):
    """Wrap content in a container with explicit light background"""
    wrapper = f"""
    <div style="
        background-color: #f8f9fa;
        color: #212529;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #dee2e6;
        margin: 10px 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        {content_html}
    </div>
    """
    return HTML(wrapper)

# For tables:
def styled_table(df, title=""):
    """Create a professionally styled HTML table"""
    table_css = """
    <style>
    .styled-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 14px;
        background: white;
    }
    .styled-table th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
    }
    .styled-table td {
        padding: 10px 15px;
        border-bottom: 1px solid #e0e0e0;
        color: #333;
    }
    .styled-table tr:nth-child(even) {
        background-color: #f8f9fa;
    }
    .styled-table tr:hover {
        background-color: #e3f2fd;
    }
    .styled-table .total-row {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        font-weight: bold;
    }
    </style>
    """
    return HTML(table_css + df.to_html(classes='styled-table'))
```

### Pros
- 100% reliable - ignores Colab's theme entirely
- Consistent appearance for all users
- Professional, polished look

### Cons
- Creates a "light island" in dark mode (may look jarring to some)
- Doesn't respect user's dark mode preference

---

## OPTION 3: User Toggle Switch (MOST FLEXIBLE)

### How It Works
Add a theme toggle widget that lets users switch between light and dark color schemes within the tool.

### Implementation
```python
import ipywidgets as widgets
from IPython.display import display, HTML

class ThemeManager:
    def __init__(self):
        self.current_theme = 'light'
        self.themes = {
            'light': {
                'bg': '#ffffff',
                'bg_secondary': '#f8f9fa',
                'text': '#212529',
                'text_secondary': '#6c757d',
                'accent': '#0d6efd',
                'success': '#198754',
                'border': '#dee2e6',
                'highlight': '#e3f2fd'
            },
            'dark': {
                'bg': '#1a1a2e',
                'bg_secondary': '#16213e',
                'text': '#eaeaea',
                'text_secondary': '#a0a0a0',
                'accent': '#4fc3f7',
                'success': '#81c784',
                'border': '#3a3a5c',
                'highlight': '#0f3460'
            }
        }
    
    def create_toggle(self):
        toggle = widgets.ToggleButtons(
            options=['☀️ Light', '🌙 Dark'],
            description='Theme:',
            button_style='info',
            tooltips=['Light mode for bright environments', 
                      'Dark mode for low-light environments']
        )
        toggle.observe(self._on_theme_change, names='value')
        return toggle
    
    def _on_theme_change(self, change):
        self.current_theme = 'dark' if '🌙' in change['new'] else 'light'
        self._inject_theme_css()
    
    def _inject_theme_css(self):
        t = self.themes[self.current_theme]
        css = f"""
        <style id="heat-reuse-theme">
        .hr-container {{
            background: {t['bg']};
            color: {t['text']};
            border: 1px solid {t['border']};
        }}
        .hr-header {{
            background: {t['bg_secondary']};
            color: {t['text']};
        }}
        .hr-total {{
            color: {t['success']};
            font-weight: bold;
        }}
        </style>
        """
        display(HTML(css))

# Usage:
theme_mgr = ThemeManager()
display(theme_mgr.create_toggle())
```

### Pros
- User has full control
- Respects user preference explicitly
- Can save preference to localStorage

### Cons
- Extra UI element
- User must take action
- More complex implementation

---

## OPTION 4: High-Contrast Color Palette (UNIVERSAL)

### How It Works
Use colors that have sufficient contrast on BOTH light and dark backgrounds - typically saturated colors with medium brightness.

### Recommended Color Palette
```python
# Colors that work on both light (#fff) and dark (#1a1a1a) backgrounds
UNIVERSAL_COLORS = {
    # Primary UI Colors
    'primary_blue': '#2196F3',      # Bright blue - visible on both
    'success_green': '#4CAF50',     # Medium green
    'warning_orange': '#FF9800',    # Bright orange
    'error_red': '#f44336',         # Bright red
    
    # Text Colors (CRITICAL)
    'text_primary': '#37474F',      # Dark blue-gray (light bg)
    'text_dark_mode': '#B0BEC5',    # Light blue-gray (dark bg)
    
    # Use these for important numbers:
    'currency_green': '#00C853',    # Bright green - costs
    'highlight_cyan': '#00BCD4',    # Cyan for emphasis
    'total_purple': '#7C4DFF',      # Purple for totals
    
    # Background accents
    'row_alt_light': '#ECEFF1',
    'row_alt_dark': '#263238',
}

# For text that MUST be visible:
# Instead of black (#000) use: #37474F (visible on light)
# Instead of white (#fff) use: #ECEFF1 (visible on dark)
```

### CSS Implementation
```css
/* Universal high-contrast styles */
.hr-value {
    color: #00C853;  /* Bright green - visible on any background */
    font-weight: 600;
    text-shadow: 0 0 1px rgba(0,0,0,0.3);  /* Subtle shadow for legibility */
}

.hr-label {
    color: #78909C;  /* Medium gray - reasonable contrast both ways */
}

.hr-total {
    color: #7C4DFF;  /* Bright purple */
    font-size: 1.1em;
    font-weight: bold;
}

/* Add subtle background to ensure readability */
.hr-data-cell {
    background: rgba(255,255,255,0.9);  /* Mostly opaque white */
    padding: 8px 12px;
    border-radius: 4px;
}

@media (prefers-color-scheme: dark) {
    .hr-data-cell {
        background: rgba(30,30,30,0.9);  /* Mostly opaque dark */
    }
}
```

### Pros
- No theme detection needed
- Works everywhere
- Simple implementation

### Cons  
- May not be as visually "pretty" as themed approach
- Limited color palette options

---

## OPTION 5: HTML Report Cards (PROFESSIONAL LOOK)

### How It Works
Render all outputs as styled HTML "cards" with explicit styling, similar to a dashboard or report.

### Implementation
```python
def create_cost_card(title, items, total, currency="€"):
    """Create a professional-looking cost breakdown card"""
    
    items_html = ""
    for label, value in items:
        items_html += f"""
        <div class="cost-item">
            <span class="cost-label">{label}</span>
            <span class="cost-value">{currency} {value:,.0f}</span>
        </div>
        """
    
    card_html = f"""
    <style>
    .cost-card {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        font-family: 'Segoe UI', system-ui, sans-serif;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        max-width: 400px;
    }}
    .cost-card-title {{
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        margin-bottom: 8px;
    }}
    .cost-item {{
        display: flex;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid rgba(255,255,255,0.2);
    }}
    .cost-label {{
        opacity: 0.9;
    }}
    .cost-value {{
        font-weight: 600;
        font-size: 16px;
    }}
    .cost-total {{
        display: flex;
        justify-content: space-between;
        padding-top: 16px;
        margin-top: 8px;
        font-size: 20px;
        font-weight: bold;
    }}
    .cost-total-value {{
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0,255,136,0.5);
    }}
    </style>
    
    <div class="cost-card">
        <div class="cost-card-title">{title}</div>
        {items_html}
        <div class="cost-total">
            <span>TOTAL</span>
            <span class="cost-total-value">{currency} {total:,.0f}</span>
        </div>
    </div>
    """
    return HTML(card_html)
```

### Pros
- Beautiful, modern appearance
- Fully self-contained styling
- Impressive for presentations
- Works on any background

### Cons
- More complex HTML generation
- May look "over-designed" for some users
- Takes more screen space

---

## RECOMMENDED APPROACH: Hybrid Solution

For the Heat Reuse Tool, I recommend combining **Option 2 + Option 4**:

1. **Wrap all outputs in explicit containers** with light/neutral backgrounds
2. **Use the universal high-contrast color palette** for all text and values
3. **Add a simple theme toggle** as an optional enhancement

### Priority Implementation Steps:

1. **IMMEDIATE FIX** - Add explicit `background-color` and `color` to all HTML outputs
2. **QUICK WIN** - Use the universal color palette for all values
3. **POLISH** - Add gradient headers and card-style containers
4. **FUTURE** - Add theme toggle widget

### Minimum Viable CSS Block (Add at notebook start):
```python
from IPython.display import HTML, display

display(HTML("""
<style>
/* Heat Reuse Tool - Universal Styles */
.output_subarea {
    background-color: #f8f9fa !important;
    padding: 16px !important;
    border-radius: 8px !important;
}

/* Force readable text colors */
.widget-label, .widget-readout {
    color: #333333 !important;
}

/* Style ipywidgets outputs */
.jupyter-widgets-output-area .output_subarea {
    background: white !important;
    color: #212529 !important;
}
</style>
"""))
```

---

## ACCESSIBILITY CHECKLIST (WCAG 2.1 AA)

- [ ] Text contrast ratio ≥ 4.5:1 for normal text
- [ ] Text contrast ratio ≥ 3:1 for large text (18pt+)
- [ ] Don't rely on color alone to convey information
- [ ] Focus indicators visible
- [ ] Text resizable up to 200%

### Contrast Checking Tools
- WebAIM Contrast Checker: https://webaim.org/resources/contrastchecker/
- Adobe Color Accessibility: https://color.adobe.com/create/color-accessibility

---

## FILES TO CREATE/MODIFY

1. **styles.py** - Theme management and CSS injection functions
2. **widgets.py** - Modified widget styling with explicit colors
3. **display.py** - HTML report card generators
4. **constants.py** - Color palette definitions

---

*Document prepared for Claude Code implementation*  
*Last updated: January 7, 2025*
