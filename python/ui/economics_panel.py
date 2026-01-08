"""
Author: Ahliana Byrd <ahliana.byrd@gmail.com>
Created: 2025-10-08
"""

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
    Create HTML table comparing all three approaches (2°C, 3°C, 5°C) with transparent breakdown.

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

    # Build table HTML with new transparent structure
    html = """
    <div style="margin: 20px 0; font-family: 'Segoe UI', Arial, sans-serif;">
        <table style="width: 100%; border-collapse: collapse; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <thead>
                <tr style="background-color: #2196F3; color: white;">
                    <th style="padding: 14px; text-align: left; border: 1px solid #1976D2; font-size: 14px;">Cost Component</th>
                    <th style="padding: 14px; text-align: right; border: 1px solid #1976D2; font-size: 14px;">2°C Approach</th>
                    <th style="padding: 14px; text-align: right; border: 1px solid #1976D2; font-size: 14px;">3°C Approach</th>
                    <th style="padding: 14px; text-align: right; border: 1px solid #1976D2; font-size: 14px;">5°C Approach</th>
                </tr>
            </thead>
            <tbody>
    """

    # Section header: EQUIPMENT COSTS (Base)
    html += """
                <tr style="background-color: #E3F2FD;">
                    <td colspan="4" style="padding: 10px; border: 1px solid #BBDEFB; font-weight: bold;
                                         color: #1565C0; font-size: 13px; letter-spacing: 0.5px;">
                        EQUIPMENT COSTS (Base)
                    </td>
                </tr>
    """

    # Base equipment cost rows
    base_rows = [
        ('Heat Exchangers', 'heat_exchanger', 'Industrial plate heat exchangers sized for system capacity'),
        ('Pumps', 'pumps', 'Circulation pumps with motor and VFD controls'),
        ('Piping & Fittings', 'pipe_fittings', 'Stainless steel pipes, elbows, tees, and connections'),
        ('Instrumentation', 'instrumentation', 'Temperature sensors, flow meters, and control systems'),
        ('Valves', 'valves', 'Control valves and isolation valves'),
    ]

    for row_label, key, tooltip in base_rows:
        html += f"""
                <tr style="background-color: #FAFAFA;"
                    title="{tooltip}">
                    <td style="padding: 10px; border: 1px solid #E0E0E0; padding-left: 20px;">{row_label}</td>
        """
        for approach in ['2C', '3C', '5C']:
            value = approaches_data.get(approach, {}).get(key, 0)
            # Round valves to nearest 100 to match Valve Costs display in Piping Cost Analysis
            if key == 'valves':
                value = round(value / 100) * 100
            html += f"""
                    <td style="padding: 10px; text-align: right; border: 1px solid #E0E0E0;
                               font-family: 'Segoe UI', Arial, sans-serif;">€{value:>10,.0f}</td>
            """
        html += """
                </tr>
        """

    # Equipment Subtotal row
    html += """
                <tr style="background-color: #E8EAF6; font-weight: bold;">
                    <td style="padding: 11px; border: 1px solid #C5CAE9; padding-left: 20px;">Equipment Subtotal</td>
    """

    # Calculate and validate equipment subtotals
    for approach in ['2C', '3C', '5C']:
        data = approaches_data.get(approach, {})

        # Calculate subtotal from base costs
        subtotal = sum([
            data.get('heat_exchanger', 0),
            data.get('pumps', 0),
            data.get('pipe_fittings', 0),
            data.get('instrumentation', 0),
            data.get('valves', 0)
        ])

        # Validation check
        validation_icon = "✓" if abs(subtotal - data.get('heat_exchanger', 0) - data.get('pumps', 0) -
                                      data.get('pipe_fittings', 0) - data.get('instrumentation', 0) -
                                      data.get('valves', 0)) < 1 else "⚠"

        html += f"""
                    <td style="padding: 11px; text-align: right; border: 1px solid #C5CAE9;
                               font-family: 'Segoe UI', Arial, sans-serif;">
                        €{subtotal:>10,.0f} <span style="color: #4CAF50; font-size: 11px;">{validation_icon}</span>
                    </td>
        """
    html += """
                </tr>
    """

    # Section header: INSTALLATION & CONTINGENCY
    html += """
                <tr style="background-color: #FFF3E0;">
                    <td colspan="4" style="padding: 10px; border: 1px solid #FFE0B2; font-weight: bold;
                                         color: #E65100; font-size: 13px; letter-spacing: 0.5px;">
                        INSTALLATION & CONTINGENCY
                    </td>
                </tr>
    """

    # Contingency rows
    contingency_rows = [
        ('Installation (15%)', 'installation_cost', 'Labor and materials for equipment installation'),
        ('Engineering (10%)', 'engineering_cost', 'Design, engineering, and project management'),
        ('Contingency (10%)', 'contingency_cost', 'Unforeseen costs and scope changes'),
    ]

    for row_label, key, tooltip in contingency_rows:
        html += f"""
                <tr style="background-color: #FFF8E1;"
                    title="{tooltip}">
                    <td style="padding: 10px; border: 1px solid #E0E0E0; padding-left: 20px; color: #F57C00;">{row_label}</td>
        """
        for approach in ['2C', '3C', '5C']:
            value = approaches_data.get(approach, {}).get(key, 0)
            html += f"""
                    <td style="padding: 10px; text-align: right; border: 1px solid #E0E0E0;
                               font-family: 'Segoe UI', Arial, sans-serif; color: #F57C00;">€{value:>10,.0f}</td>
            """
        html += """
                </tr>
        """

    # I&C Subtotal row (Installation & Contingency subtotal)
    html += """
                <tr style="background-color: #FFE0B2; font-weight: bold;">
                    <td style="padding: 11px; border: 1px solid #FFCC80; padding-left: 20px;">I&C Subtotal</td>
    """

    for approach in ['2C', '3C', '5C']:
        data = approaches_data.get(approach, {})
        ic_subtotal = sum([
            data.get('installation_cost', 0),
            data.get('engineering_cost', 0),
            data.get('contingency_cost', 0)
        ])

        html += f"""
                    <td style="padding: 11px; text-align: right; border: 1px solid #FFCC80;
                               font-family: 'Segoe UI', Arial, sans-serif;">€{ic_subtotal:>10,.0f}</td>
        """
    html += """
                </tr>
    """

    # Separator row
    html += """
                <tr style="background-color: #FFFFFF;">
                    <td colspan="4" style="padding: 2px; border: none; border-top: 3px double #2196F3;"></td>
                </tr>
    """

    # CAPITAL TOTAL row (highlighted and bold)
    html += """
                <tr style="background-color: #1976D2; color: white; font-weight: bold; font-size: 15px;">
                    <td style="padding: 14px; border: 1px solid #1565C0; text-transform: uppercase;">CAPITAL TOTAL</td>
    """

    for approach in ['2C', '3C', '5C']:
        data = approaches_data.get(approach, {})
        capital_total = data.get('capital_total', 0)

        # Calculate expected total for validation
        equipment_subtotal = sum([
            data.get('heat_exchanger', 0),
            data.get('pumps', 0),
            data.get('pipe_fittings', 0),
            data.get('instrumentation', 0),
            data.get('valves', 0)
        ])
        ic_subtotal = sum([
            data.get('installation_cost', 0),
            data.get('engineering_cost', 0),
            data.get('contingency_cost', 0)
        ])
        expected_total = equipment_subtotal + ic_subtotal

        # Validation (allow for rounding to nearest 500)
        validation_icon = "✓" if abs(capital_total - expected_total) <= 500 else "⚠"

        html += f"""
                    <td style="padding: 14px; text-align: right; border: 1px solid #1565C0;
                               font-family: 'Segoe UI', Arial, sans-serif;">
                        €{capital_total:>10,.0f} <span style="font-size: 12px;">{validation_icon}</span>
                    </td>
        """
    html += """
                </tr>
    """

    # Empty row for spacing
    html += """
                <tr style="background-color: #FFFFFF;">
                    <td colspan="4" style="padding: 8px; border: none;"></td>
                </tr>
    """

    # Section header: OPERATING COSTS
    html += """
                <tr style="background-color: #E8F5E9;">
                    <td colspan="4" style="padding: 10px; border: 1px solid #C8E6C9; font-weight: bold;
                                         color: #2E7D32; font-size: 13px; letter-spacing: 0.5px;">
                        OPERATING COSTS (Annual)
                    </td>
                </tr>
    """

    # Operating cost rows
    html += """
                <tr style="background-color: #F1F8E9;">
                    <td style="padding: 10px; border: 1px solid #E0E0E0; padding-left: 20px;">Annual Operating Energy</td>
    """
    for approach in ['2C', '3C', '5C']:
        value = approaches_data.get(approach, {}).get('operating_energy_kwh_year', 0)
        html += f"""
                    <td style="padding: 10px; text-align: right; border: 1px solid #E0E0E0;
                               font-family: 'Segoe UI', Arial, sans-serif;">{value:>10,.0f} kWh</td>
        """
    html += """
                </tr>
    """

    html += """
                <tr style="background-color: #F1F8E9; font-weight: bold;">
                    <td style="padding: 10px; border: 1px solid #E0E0E0; padding-left: 20px;">Annual Energy Cost</td>
    """
    for approach in ['2C', '3C', '5C']:
        value = approaches_data.get(approach, {}).get('operating_cost_eur_year', 0)
        html += f"""
                    <td style="padding: 10px; text-align: right; border: 1px solid #E0E0E0;
                               font-family: 'Segoe UI', Arial, sans-serif;">€{value:>10,.0f}</td>
        """
    html += """
                </tr>
            </tbody>
        </table>
    """

    # Legend section
    html += """
        <div style="margin-top: 15px; padding: 12px; background-color: #F5F5F5;
                    border-radius: 4px; font-size: 12px; color: #616161;">
            <strong>Legend:</strong>
            <span style="color: #4CAF50; font-weight: bold;">✓</span> = Calculations verified |
            <span style="color: #FF9800; font-weight: bold;">⚠</span> = Rounding adjustments applied |
            <span style="font-style: italic;">Hover over items for details</span>
        </div>
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


def create_approach_cost_breakdown_charts(wha: float, T1: float, temp_rise: float, output_area):
    """
    Create pie charts showing cost breakdown for each approach temperature.

    Args:
        wha: System power in MW
        T1: Inlet temperature in °C
        temp_rise: Temperature rise in °C
        output_area: Output widget to display the chart
    """
    try:
        # Get comparison data (suppress logging)
        with suppress_logging():
            comparison = compare_approaches(wha, T1, temp_rise, approaches=[2, 3, 5])

        if comparison.get('status') != 'success':
            with output_area:
                display(HTML("<p style='color: red;'>Unable to generate cost breakdown charts</p>"))
            return

        approaches_data = comparison['approaches']

        # Create figure with 1 row, 3 columns (increased height for legends)
        fig, axs = plt.subplots(1, 3, figsize=(18, 7))

        # Approach temperatures and titles
        approaches = [2, 3, 5]
        titles = ["2°C Approach", "3°C Approach", "5°C Approach"]
        # Rainbow spectrum color scheme
        colors = [
            '#E74C3C',  # Red - Heat Exchangers
            '#FF9F43',  # Orange - Pumps
            '#F1C40F',  # Yellow - Piping & Fittings
            '#2ECC71',  # Green - Instrumentation
            '#3498DB',  # Blue - Valves
            '#9B59B6'   # Purple - I&C Subtotal
        ]
        labels = ['Heat Exchangers', 'Pumps', 'Piping & Fittings', 'Instrumentation', 'Valves', 'I&C Subtotal']

        # Create pie chart for each approach
        for idx, (approach, title) in enumerate(zip(approaches, titles)):
            key = f"{approach}C"
            data = approaches_data.get(key, {})

            # Extract cost components
            heat_exchanger = data.get('heat_exchanger', 0)
            pumps = data.get('pumps', 0)
            pipe_fittings = data.get('pipe_fittings', 0)
            instrumentation = data.get('instrumentation', 0)
            valves = data.get('valves', 0)

            # Calculate I&C Subtotal (installation + engineering + contingency)
            ic_subtotal = sum([
                data.get('installation_cost', 0),
                data.get('engineering_cost', 0),
                data.get('contingency_cost', 0)
            ])

            # Combine into values array
            values = [heat_exchanger, pumps, pipe_fittings, instrumentation, valves, ic_subtotal]

            # Create pie chart with percentages only (labels in legend for clarity)
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

        # Display in output area
        with output_area:
            plt.show()

        # Close to free memory
        plt.close()

    except Exception as e:
        with output_area:
            display(HTML(f"<p style='color: red;'>Error creating cost breakdown charts: {str(e)}</p>"))


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

            # Display equipment cost breakdown charts
            breakdown_title_html = """
            <div style="margin: 30px 0 15px 0;">
                <h3 style="color: #1976D2;">🔧 Equipment Cost Breakdown by Approach</h3>
                <p style="color: #616161; font-size: 14px; margin-top: 5px;">
                    Comparison of cost distribution across equipment categories for different approach temperatures
                </p>
            </div>
            """
            display(HTML(breakdown_title_html))

            create_approach_cost_breakdown_charts(wha, T1, temp_rise, output_area)

        except Exception as e:
            error_html = f"""
            <div style="margin: 20px 0; padding: 15px; background-color: #f8d7da;
                        border: 1px solid #f5c6cb; border-radius: 4px; color: #721c24;">
                <strong>⚠️ Error displaying economics analysis:</strong> {str(e)}
            </div>
            """
            display(HTML(error_html))
