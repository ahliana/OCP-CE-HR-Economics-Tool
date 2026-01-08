"""
Author: Ahliana Byrd <ahliana.byrd@gmail.com>
Created: 2026-01-07
"""

"""
Visual Styling Module
CSS injection, color palette, and styled HTML container generators.
Ensures readability on both light and dark mode backgrounds (Google Colab compatibility).
"""

from IPython.display import HTML, display

# =============================================================================
# HIGH-CONTRAST COLOR PALETTE
# Colors that work on BOTH light (#fff) and dark (#1a1a1a) backgrounds
# =============================================================================

COLORS = {
    # Primary UI Colors
    'primary_blue': '#2196F3',
    'primary_blue_dark': '#1976D2',
    'success_green': '#4CAF50',
    'success_green_dark': '#2E7D32',
    'warning_orange': '#FF9800',
    'warning_orange_dark': '#E65100',
    'error_red': '#f44336',

    # High-contrast text colors for values
    'currency_green': '#00C853',       # Bright green for costs (works on light bg)
    'highlight_cyan': '#00838F',       # Darker cyan for emphasis on light bg
    'total_purple': '#5E35B1',         # Darker purple for totals (better contrast)

    # DARK TEXT COLORS for light backgrounds (forced containers)
    'label_dark': '#333333',           # Dark gray for labels - HIGH CONTRAST
    'text_dark': '#1a1a1a',            # Near-black for body text
    'text_secondary': '#424242',       # Medium-dark for secondary text
    'text_muted': '#616161',           # For less important text (still readable)

    # Keep for reference but prefer dark colors
    'text_light': '#ECEFF1',           # Light text for dark/colored backgrounds only

    # Background colors for containers
    'container_bg': '#f8f9fa',         # Light gray container background
    'container_border': '#dee2e6',     # Light border
    'row_alt': '#ECEFF1',              # Alternating row color
    'row_white': '#ffffff',            # White rows

    # Section header gradients
    'header_gradient_blue': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    'header_gradient_green': 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
    'header_gradient_orange': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
}

# =============================================================================
# GLOBAL CSS INJECTION
# =============================================================================

GLOBAL_CSS = """
<style id="heat-reuse-global-styles">
/* Heat Reuse Tool - Universal Styles for Light/Dark Mode Compatibility */

/* Force light background on output areas */
.output_subarea {
    background-color: #f8f9fa !important;
    padding: 16px !important;
    border-radius: 8px !important;
}

/* Force readable text colors on ipywidgets */
.widget-label, .widget-readout {
    color: #333333 !important;
}

/* Style ipywidgets outputs */
.jupyter-widgets-output-area .output_subarea {
    background: white !important;
    color: #212529 !important;
}

/* Heat Reuse Tool container class */
.hr-container {
    background-color: #f8f9fa;
    color: #212529;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dee2e6;
    margin: 10px 0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* High-contrast value styling */
.hr-value {
    color: #00C853;
    font-weight: 600;
}

.hr-value-currency {
    color: #00C853;
    font-weight: 600;
}

.hr-value-total {
    color: #7C4DFF;
    font-weight: bold;
    font-size: 1.1em;
}

.hr-label {
    color: #37474F;
    font-weight: 500;
}

/* Table styling */
.hr-table {
    width: 100%;
    border-collapse: collapse;
    background: white;
}

.hr-table th {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 12px 15px;
    text-align: left;
    font-weight: 600;
}

.hr-table td {
    padding: 10px 15px;
    border-bottom: 1px solid #e0e0e0;
    color: #333;
}

.hr-table tr:nth-child(even) {
    background-color: #f8f9fa;
}

.hr-table tr:hover {
    background-color: #e3f2fd;
}

.hr-table .total-row {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
    font-weight: bold;
}

/* Section headers */
.hr-section-header {
    color: white;
    padding: 12px 16px;
    border-radius: 8px 8px 0 0;
    margin: 0;
    font-weight: 600;
}

.hr-section-header-blue {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.hr-section-header-green {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
}

.hr-section-header-orange {
    background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
}

/* Card styling */
.hr-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 1px solid #e0e0e0;
}

/* Checkmark and validation icons with visibility */
.hr-check {
    color: #00C853;
    font-weight: bold;
}

.hr-warning {
    color: #FF9800;
    font-weight: bold;
}

/* Force visibility on all text elements */
.hr-container *, .hr-card *, .hr-table * {
    color: inherit;
}
</style>
"""


def inject_global_styles():
    """
    Inject global CSS styles at notebook startup.
    Call this once when the tool initializes.
    """
    display(HTML(GLOBAL_CSS))


# =============================================================================
# STYLED CONTAINER GENERATORS
# =============================================================================

def wrap_in_container(content_html, title=None, border_color=None, title_bg=None):
    """
    Wrap HTML content in a styled container with explicit background.
    Ensures visibility on both light and dark backgrounds.

    Args:
        content_html: HTML content to wrap
        title: Optional title for the container
        border_color: Optional border color (defaults to primary blue)
        title_bg: Optional background for title (gradient string or color)

    Returns:
        HTML string with styled container
    """
    border = border_color or COLORS['primary_blue']

    title_html = ""
    if title:
        title_style = title_bg or COLORS['header_gradient_blue']
        # Check if it's a gradient or solid color
        if 'gradient' in title_style:
            bg_style = f"background: {title_style};"
        else:
            bg_style = f"background-color: {title_style};"

        title_html = f"""
        <div style="{bg_style} color: white; padding: 14px 20px;
                    border-radius: 12px 12px 0 0; margin: -20px -20px 15px -20px;
                    font-size: 16px; font-weight: 600;">
            {title}
        </div>
        """

    return f"""
    <div style="
        background-color: {COLORS['container_bg']};
        color: {COLORS['text_dark']};
        padding: 20px;
        border-radius: 12px;
        border: 2px solid {border};
        margin: 10px 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        {title_html}
        {content_html}
    </div>
    """


def create_styled_table(data_rows, header_row=None, show_total=False):
    """
    Create a styled HTML table with high-contrast colors.

    Args:
        data_rows: List of (label, value) tuples
        header_row: Optional tuple of header labels
        show_total: If True, style the last row as a total row

    Returns:
        HTML string for the styled table
    """
    header_html = ""
    if header_row:
        cells = "".join([f"<th style='padding: 12px 15px; text-align: left; color: white;'>{h}</th>"
                        for h in header_row])
        header_html = f"""
        <thead>
            <tr style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                {cells}
            </tr>
        </thead>
        """

    rows_html = ""
    for i, (label, value) in enumerate(data_rows):
        is_last = i == len(data_rows) - 1
        is_total = show_total and is_last

        if is_total:
            # Total row styling - white text on gradient
            row_style = "background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); color: white;"
            label_style = "padding: 14px 15px; font-weight: bold; font-size: 16px; color: white;"
            value_style = "padding: 14px 15px; font-weight: bold; font-size: 16px; color: white; text-align: right;"
        else:
            # Regular row styling - DARK text on light background
            row_bg = COLORS['row_alt'] if i % 2 == 1 else "white"
            row_style = f"background-color: {row_bg};"
            border_style = "border-bottom: 1px solid #e0e0e0;" if not is_last else ""
            # Use dark text colors for high contrast
            label_style = f"padding: 10px 15px; font-weight: 600; color: {COLORS['label_dark']}; {border_style}"
            value_style = f"padding: 10px 15px; color: {COLORS['currency_green']}; font-weight: 700; text-align: right; {border_style}"

        rows_html += f"""
        <tr style="{row_style}">
            <td style="{label_style}">{label}</td>
            <td style="{value_style}">{value}</td>
        </tr>
        """

    return f"""
    <table style="width: 100%; border-collapse: collapse; background: white;
                  border-radius: 8px; overflow: hidden;">
        {header_html}
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """


def create_section_header(title, icon="", color_scheme="blue"):
    """
    Create a styled section header.

    Args:
        title: Header title text
        icon: Optional emoji/icon
        color_scheme: "blue", "green", or "orange"

    Returns:
        HTML string for the header
    """
    gradients = {
        'blue': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'green': 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
        'orange': 'linear-gradient(135deg, #FF9800 0%, #F57C00 100%)'
    }

    gradient = gradients.get(color_scheme, gradients['blue'])

    return f"""
    <div style="background: {gradient}; color: white; padding: 14px 20px;
                border-radius: 8px; margin: 15px 0 10px 0;
                font-size: 18px; font-weight: 600;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);">
        {icon} {title}
    </div>
    """


def format_currency(value, show_symbol=True):
    """
    Format a currency value with high-contrast styling.

    Args:
        value: Numeric value
        show_symbol: Whether to show currency symbol

    Returns:
        HTML-styled currency string
    """
    symbol = "€" if show_symbol else ""
    formatted = f"{value:,.0f}"
    return f'<span style="color: #00C853; font-weight: 600;">{symbol}{formatted}</span>'


def format_total(value, label="TOTAL"):
    """
    Format a total value with prominent styling.

    Args:
        value: Numeric value
        label: Label text

    Returns:
        HTML-styled total string
    """
    return f'<span style="color: #7C4DFF; font-weight: bold; font-size: 1.1em;">€{value:,.0f}</span>'


# =============================================================================
# MESSAGE STYLING
# =============================================================================

def create_message_html(message, message_type='info'):
    """
    Create a styled message box with explicit colors.

    Args:
        message: Message text
        message_type: 'success', 'error', 'warning', or 'info'

    Returns:
        HTML string for the message
    """
    styles = {
        'success': {
            'bg': '#d4edda',
            'border': '#28a745',
            'text': '#155724',
            'icon': '✅'
        },
        'error': {
            'bg': '#f8d7da',
            'border': '#dc3545',
            'text': '#721c24',
            'icon': '❌'
        },
        'warning': {
            'bg': '#fff3cd',
            'border': '#ffc107',
            'text': '#856404',
            'icon': '⚠️'
        },
        'info': {
            'bg': '#e3f2fd',
            'border': '#2196F3',
            'text': '#0d47a1',
            'icon': 'ℹ️'
        }
    }

    s = styles.get(message_type, styles['info'])

    return f"""
    <div style="background-color: {s['bg']}; color: {s['text']};
                padding: 15px 20px; border-radius: 8px; margin: 10px 0;
                border: 2px solid {s['border']};
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <strong>{s['icon']} {message}</strong>
    </div>
    """
