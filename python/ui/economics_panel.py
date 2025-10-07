"""
Economics Analysis Panel
Order of Magnitude Estimate comparison and visualization
"""

from IPython.display import display, HTML
import matplotlib.pyplot as plt
import numpy as np
import logging
import io
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from core.costs import compare_approaches

@contextmanager
def suppress_logging():
    """Context manager to suppress logging output and print statements."""
    # Save original logging levels
    loggers = {
        'core.costs': logging.getLogger('core.costs'),
        'core.lookup': logging.getLogger('core.lookup'),
        'core.original_calculations': logging.getLogger('core.original_calculations')
    }
    original_levels = {name: logger.level for name, logger in loggers.items()}

    # Suppress logging
    for logger in loggers.values():
        logger.setLevel(logging.CRITICAL)

    # Suppress stdout and stderr (print statements)
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield
    finally:
        # Restore logging levels
        for name, level in original_levels.items():
            loggers[name].setLevel(level)

# =============================================================================
# COMPARISON TABLE
# =============================================================================

def create_economics_comparison_table(wha: float, T1: float, temp_rise: float) -> str:
    """
    Create HTML table comparing all three approaches (2°C, 3°C, 5°C).

    Args:
        wha: System power in MW
        T1: Inlet temperature in °C
        temp_rise: Temperature rise in °C

    Returns:
        HTML string for the comparison table
    """
    # Get comparison data from costs module (suppress logging)
    with suppress_logging():
        comparison = compare_approaches(wha, T1, temp_rise, approaches=[2, 3, 5])

    if comparison.get('status') != 'success':
        return create_error_table("Unable to calculate economics comparison")

    approaches_data = comparison['approaches']

    # Build table HTML
    html = """
    <div style="margin: 20px 0; font-family: Arial, sans-serif;">
        <div style="background-color: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 12px; margin-bottom: 15px;">
            <strong>📊 Note:</strong> Values shown are equipment costs. Installation multipliers pending calibration.
        </div>

        <table style="width: 100%; border-collapse: collapse; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <thead>
                <tr style="background-color: #2196F3; color: white;">
                    <th style="padding: 12px; text-align: left; border: 1px solid #ddd;">Cost Component</th>
                    <th style="padding: 12px; text-align: right; border: 1px solid #ddd;">2°C Approach</th>
                    <th style="padding: 12px; text-align: right; border: 1px solid #ddd;">3°C Approach</th>
                    <th style="padding: 12px; text-align: right; border: 1px solid #ddd;">5°C Approach</th>
                </tr>
            </thead>
            <tbody>
    """

    # Capital cost rows
    rows = [
        ('Heat Exchanger', 'heat_exchanger'),
        ('Pumps', 'pumps'),
        ('Pipe & Fittings', 'pipe_fittings'),
        ('Instruments', 'instrumentation'),
    ]

    for row_label, key in rows:
        html += f"""
                <tr style="background-color: #f8f9fa;">
                    <td style="padding: 10px; border: 1px solid #ddd;">{row_label}</td>
        """
        for approach in ['2C', '3C', '5C']:
            value = approaches_data.get(approach, {}).get(key, 0)
            html += f"""
                    <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">€{value:,.0f}</td>
            """
        html += """
                </tr>
        """

    # Capital total row (highlighted)
    html += """
                <tr style="background-color: #e3f2fd; font-weight: bold;">
                    <td style="padding: 10px; border: 1px solid #ddd;">Capital Total</td>
    """
    for approach in ['2C', '3C', '5C']:
        value = approaches_data.get(approach, {}).get('capital_total', 0)
        html += f"""
                    <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">€{value:,.0f}</td>
        """
    html += """
                </tr>
    """

    # Operating cost rows
    html += """
                <tr style="background-color: #fff9c4;">
                    <td style="padding: 10px; border: 1px solid #ddd;">Annual Operating Energy</td>
    """
    for approach in ['2C', '3C', '5C']:
        value = approaches_data.get(approach, {}).get('operating_energy_kwh_year', 0)
        html += f"""
                    <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">{value:,.0f} kWh</td>
        """
    html += """
                </tr>
    """

    html += """
                <tr style="background-color: #fff9c4; font-weight: bold;">
                    <td style="padding: 10px; border: 1px solid #ddd;">Annual Energy Cost</td>
    """
    for approach in ['2C', '3C', '5C']:
        value = approaches_data.get(approach, {}).get('operating_cost_eur_year', 0)
        html += f"""
                    <td style="padding: 10px; text-align: right; border: 1px solid #ddd;">€{value:,.0f}</td>
        """
    html += """
                </tr>
            </tbody>
        </table>
    </div>
    """

    return html


def create_error_table(error_message: str) -> str:
    """Create an error message table."""
    return f"""
    <div style="margin: 20px 0; padding: 15px; background-color: #f8d7da;
                border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24;">
        <strong>⚠️ Error:</strong> {error_message}
    </div>
    """


# =============================================================================
# COST CONTRAST GRAPH
# =============================================================================

def create_cost_contrast_chart(wha: float, T1: float, temp_rise: float, output_area):
    """
    Create cost contrast chart comparing Capital vs Operating costs across approaches.

    Args:
        wha: System power in MW
        T1: Inlet temperature in °C
        temp_rise: Temperature rise in °C
        output_area: Output widget to display the chart
    """
    # Get comparison data (suppress logging)
    with suppress_logging():
        comparison = compare_approaches(wha, T1, temp_rise, approaches=[2, 3, 5])

    if comparison.get('status') != 'success':
        with output_area:
            display(HTML("<p style='color: red;'>Unable to generate cost contrast chart</p>"))
        return

    approaches_data = comparison['approaches']

    # Extract data for plotting
    approaches = [2, 3, 5]
    capital_costs = []
    operating_costs = []

    for approach in approaches:
        key = f"{approach}C"
        capital_costs.append(approaches_data.get(key, {}).get('capital_total', 0))
        operating_costs.append(approaches_data.get(key, {}).get('operating_cost_eur_year', 0))

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot both lines
    plt.plot(approaches, capital_costs, marker='o', linewidth=2, markersize=8,
             label='Capital Cost', color='#2196F3')
    plt.plot(approaches, operating_costs, marker='s', linewidth=2, markersize=8,
             label='Annual Operating Cost', color='#FF9800')

    # Formatting
    plt.xlabel('Approach Temperature (°C)', fontsize=12, fontweight='bold')
    plt.ylabel('Cost (€)', fontsize=12, fontweight='bold')
    plt.title('Cost Contrast: Capital vs Operating Cost', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='--')

    # Format y-axis with thousands separator
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'€{x:,.0f}'))

    # Set x-axis to only show our approach values
    plt.xticks(approaches)

    # Add value labels on points
    for i, approach in enumerate(approaches):
        plt.annotate(f'€{capital_costs[i]:,.0f}',
                    (approach, capital_costs[i]),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        plt.annotate(f'€{operating_costs[i]:,.0f}',
                    (approach, operating_costs[i]),
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9)

    plt.tight_layout()

    # Display in output area
    with output_area:
        plt.show()

    # Close to free memory
    plt.close()


# =============================================================================
# MAIN DISPLAY FUNCTION
# =============================================================================

def display_economics_analysis(output_area, wha: float, T1: float, temp_rise: float):
    """
    Display complete economics analysis panel.

    Args:
        output_area: Output widget to display in
        wha: System power in MW
        T1: Inlet temperature in °C
        temp_rise: Temperature rise in °C
    """
    # Clear output first
    output_area.clear_output(wait=True)

    with output_area:
        try:
            # Section header
            header_html = """
            <div style="margin: 20px 0;">
                <h2 style="color: #2196F3; border-bottom: 3px solid #2196F3;
                           padding-bottom: 10px; margin-bottom: 20px;">
                    💰 Economics Analysis - Order of Magnitude Estimate
                </h2>
            </div>
            """
            display(HTML(header_html))

            # Display comparison table
            table_html = create_economics_comparison_table(wha, T1, temp_rise)
            display(HTML(table_html))

            # Display cost contrast chart
            chart_title_html = """
            <div style="margin: 30px 0 15px 0;">
                <h3 style="color: #1976D2;">📈 Cost Contrast Analysis</h3>
            </div>
            """
            display(HTML(chart_title_html))

            create_cost_contrast_chart(wha, T1, temp_rise, output_area)

        except Exception as e:
            error_html = f"""
            <div style="margin: 20px 0; padding: 15px; background-color: #f8d7da;
                        border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24;">
                <strong>⚠️ Error displaying economics analysis:</strong> {str(e)}
            </div>
            """
            display(HTML(error_html))
