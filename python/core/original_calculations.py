"""
Original calculation functions for heat reuse system analysis.

This module contains the core calculation functions that were originally
in the Jupyter notebook.
"""


# =============================================================================
# IMPORT DEPENDENCIES
# =============================================================================
try:
    from typing import Dict, Optional, List, Tuple, Union

    from physics.constants import WATER_PROPERTIES
    from physics.units import (liters_per_minute_to_m3_per_second, m3_per_second_to_liters_per_minute)
    from physics.thermodynamics import sensible_heat_transfer

    # Import data access functions
    from data.loader import get_csv_data, is_csv_loaded
    from data.converter import universal_float_convert

    # Import lookup functions - now in separate module
    from core.lookup import lookup_allhx_data, get_lookup_value
    
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
    
    import logging
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)  

    
except ImportError as e:
    # Don't define functions if imports fail
    raise ImportError(f"Cannot import required physics modules: {e}")

# =============================================================================
# DEFINE FUNCTIONS
# =============================================================================
def quick_power_calculation(flow_lpm: float, temp_rise_c: float, fluid: str = 'water') -> float:
    """Quick power calculation compatible with existing code."""
    if fluid == 'water':
        props = WATER_PROPERTIES['30C']  # Representative properties
        mass_flow = liters_per_minute_to_m3_per_second(flow_lpm) * props['density']
        return mass_flow * props['specific_heat'] * temp_rise_c
    else:
        raise ValueError("Only water implemented")


def get_MW(F1: float, T1: float, T2: float) -> float:
    return quick_power_calculation(F1, T2 - T1)


def get_MW_divd(F1: float, T1: float, T2: float) -> float:
    return get_MW(F1, T1, T2) / 1_000_000

def get_DeltaT_TCS(T1, T2):
    try:
        return float(T2) - float(T1)
    except Exception as e:
        return 0.0
    
def get_DeltaT_FWS(T3, T4):
    try:
        return float(T3) - float(T4)
    except Exception as e:
        return 0.0

def get_Approach(T1, T4):
    """Calculate approach temperature"""
    try:
        return float(T4) - float(T1)
    except Exception as e:
        print(f"❌ Error in get_Approach: {e}")
        return 0.0


# =============================================================================
# DEFINE PIPE FUNCTIONS THAT USE  LOOKUPS
# =============================================================================


def get_PipeSize_Suggested(F1):
    """
    Get suggested pipe size using CEILING lookup (engineering safe)
    Returns the SMALLEST pipe size that can handle the flow (size >= required)
    """
    msg="get_PipeSize_Suggested"
    logger.info(msg)
    print(msg)
    msg = f"Parameter F1={F1}"
    logger.info(msg)
    print(msg)
        
    try:
        F1_float = float(F1)
        msg = f"F1_float={F1_float}"
        logger.info(msg)
        print(msg)        
        
        # Check if PIPSZ data exists using the data module
        if not is_csv_loaded('PIPSZ'):
            msg="PIPSZ CSV not found"
            logger.info(msg)
            print(msg)
            return 0  # Default fallback
        
        # Get and examine PIPSZ data using the data module
        pipsz_df = get_csv_data('PIPSZ')
        if pipsz_df is None:
            return 0
        
        pipsz_df = pipsz_df.copy()

        msg = f"PIPSZ data shape: {pipsz_df.shape}"
        logger.info(msg)
        print(msg)
        
        # Convert columns to numeric
        pipsz_df = pipsz_df.astype(float)  # Convert DataFrame to float
        pipsz_df.iloc[:, 0] = pipsz_df.iloc[:, 0].apply(universal_float_convert)  # Flow capacity
        pipsz_df.iloc[:, 1] = pipsz_df.iloc[:, 1].apply(universal_float_convert)  # Pipe size
        
        # Remove invalid rows
        valid_rows = pipsz_df.dropna()
        
        # Sort by flow capacity to ensure we find the smallest adequate size
        valid_rows = valid_rows.sort_values(by=valid_rows.columns[0])

        # Debug info
        flow_capacities = valid_rows.iloc[:, 0].values
        pipe_sizes = valid_rows.iloc[:, 1].values
        msg = f"Available flow capacities: min={min(flow_capacities)}, max={max(flow_capacities)}"
        logger.info(msg)
        print(msg)
        msg = f" First few flow/size pairs: {list(zip(flow_capacities[:5], pipe_sizes[:5]))}"
        logger.info(msg)
        print(msg)
        
        # Find the CEILING - first flow capacity >= required flow
        adequate_rows = valid_rows[valid_rows.iloc[:, 0] >= F1_float]
        msg = f"adequate_rows: {adequate_rows}"
        logger.info(msg)
        print(msg)
                
        if adequate_rows.empty:
            msg = f"❌ No pipe size available for flow {F1_float} l/m. Max available: {max(flow_capacities)}"
            logger.info(msg)
            print(msg)
            return max(pipe_sizes)  # Return largest available as fallback
        
        # Get the first (smallest) adequate pipe size
        selected_row = adequate_rows.iloc[0]
        selected_flow_capacity = selected_row.iloc[0]
        selected_pipe_size = selected_row.iloc[1]

        msg = f"selected_pipe_size: {selected_pipe_size}"
        logger.info(msg)
        print(msg)
                
        # msg = f"✅ CEILING match found: Flow capacity {selected_flow_capacity} >= {F1_float} → Pipe Size {selected_pipe_size}"
        # logger.info(msg)
        # print(msg)
        # msg = f"🔧 Engineering validation: Pipe can handle {selected_flow_capacity} l/m >= required {F1_float} l/m ✓"
        # logger.info(msg)
        # print(msg)
        
        return selected_pipe_size
        
    except Exception as e:
        msg = f"❌ Error in get_PipeSize_Suggested: {e}"
        logger.info(msg)
        print(msg)
        return 0  # Default fallback

def get_PipeLength(F1, T1, T2):
    """
    Calculate pipe length based on power requirement using ROOM lookup.
    """
    try:
        # Calculate power requirement
        power_mw = get_MW_divd(F1, T1, T2)
        
        # Check if ROOM data exists
        if not is_csv_loaded('ROOM'):
            print("❌ ROOM CSV not found")
            return 0
        
        # Use lookup function to find room size/length
        room_df = get_csv_data('ROOM')
        if room_df is None:
            return 0
            
        # Convert to numeric and find appropriate length
        room_df = room_df.copy()
        room_df = room_df.astype(float)  # Convert DataFrame to float
        room_df.iloc[:, 0] = room_df.iloc[:, 0].apply(universal_float_convert)  # Power capacity
        room_df.iloc[:, 1] = room_df.iloc[:, 1].apply(universal_float_convert)  # Length
        
        # Find ceiling match
        adequate_rows = room_df[room_df.iloc[:, 0] >= power_mw]
        
        if adequate_rows.empty:
            print(f"❌ No room size available for power {power_mw} MW")
            return 0
        
        # Get the first (smallest) adequate room
        length = adequate_rows.iloc[0, 1]
        # print(f"✅ Room length: {length} m for {power_mw} MW")
        
        return length
        
    except Exception as e:
        print(f"❌ Error in get_PipeLength: {e}")
        return 0

def get_PipeCost_perMeter(flow_rate, pipe_type):
    """
    Get pipe cost per meter based on flow rate and pipe type.
    European-first approach using DN sizes with conversion to match cost data.
    """
    try:
        # First get European DN pipe size from PIPSZ
        dn_size = get_PipeSize_Suggested(flow_rate)
        if dn_size == 0:
            print("❌ No suitable pipe size found")
            return 0
        msg=f"dn_size = {dn_size}"
        logger.info(msg)
        print(msg)
                
        # print(f"🔍 European pipe sizing: DN{dn_size} for flow {flow_rate} L/min")
        
        # Check if PIPCOST data exists
        if not is_csv_loaded('PIPCOST'):
            print("❌ PIPCOST CSV not found")
            return 0
        
        # Get cost data
        pipcost_df = get_csv_data('PIPCOST')
        if pipcost_df is None:
            return 0
            
        # Convert to numeric
        pipcost_df = pipcost_df.copy()
        pipcost_df.iloc[:, 0] = pipcost_df.iloc[:, 0].apply(universal_float_convert)  # Pipe size
        
        # Determine column index based on pipe type
        col_index = 1 if pipe_type.lower() == "stainless" else 2
        pipcost_df.iloc[:, col_index] = pipcost_df.iloc[:, col_index].apply(universal_float_convert)
        
        # Convert European DN size to match PIPCOST data format
        # Option 1: Try direct DN match first
        matching_rows = pipcost_df[pipcost_df.iloc[:, 0] == dn_size]
        
        if not matching_rows.empty:
            # Direct DN match found
            cost = matching_rows.iloc[0, col_index]
            msg=f"cost = {cost}"
            logger.info(msg)
            print(msg)
            return cost
        
        # Option 2: Convert DN to American inches using units.py conversion
        try:
            from units import dn_to_nominal_inches
            american_inches = dn_to_nominal_inches(dn_size)
            
            if american_inches:
                # Try to find the converted size in PIPCOST
                matching_rows = pipcost_df[pipcost_df.iloc[:, 0] == american_inches]
                
                if not matching_rows.empty:
                    cost = matching_rows.iloc[0, col_index]
                    # print(f"✅ Converted match: DN{dn_size} → {american_inches}\" → €{cost}/m")
                    return cost
                    
        except ImportError:
            # print("⚠️ Units conversion module not available")
            print('')
        
        # Option 3: European engineering fallback mapping
        # Based on standard DN to inch conversions
        european_to_cost_mapping = {
            100: 4,    # DN100 ≈ 4" (102.3mm inner diameter)
            160: 6,    # DN160 ≈ 6" (closest to 154.1mm)
            200: 8,    # DN200 ≈ 8" (202.7mm inner diameter)
            250: 10,   # DN250 ≈ 10" (254.5mm inner diameter)
            315: 12,   # DN315 ≈ 12" (closest to 303.2mm)
            350: 14,   # DN350 ≈ 14" (closest to 333.3mm)
            400: 16,   # DN400 ≈ 16" (closest to 381.0mm)
        }
        
        mapped_size = european_to_cost_mapping.get(dn_size)
        if mapped_size:
            matching_rows = pipcost_df[pipcost_df.iloc[:, 0] == mapped_size]
            
            if not matching_rows.empty:
                cost = matching_rows.iloc[0, col_index]
                # print(f"✅ Engineering mapping: DN{dn_size} → {mapped_size}\" → €{cost}/m")
                return cost
        
        # Option 4: Find closest available size as engineering fallback
        available_sizes = pipcost_df.iloc[:, 0].values
        
        # Use European DN standards to find closest match
        try:
            from units import european_dn_pipe_sizes
            dn_inner_diameter = european_dn_pipe_sizes().get(dn_size, dn_size)
            
            # Convert available inch sizes to mm for comparison
            from units import american_nominal_pipe_sizes
            american_sizes = american_nominal_pipe_sizes()
            
            closest_size = None
            min_difference = float('inf')
            
            for inch_size in available_sizes:
                if inch_size in american_sizes:
                    inch_diameter_mm = american_sizes[inch_size]
                    difference = abs(dn_inner_diameter - inch_diameter_mm)
                    
                    if difference < min_difference:
                        min_difference = difference
                        closest_size = inch_size
            
            if closest_size:
                matching_rows = pipcost_df[pipcost_df.iloc[:, 0] == closest_size]
                cost = matching_rows.iloc[0, col_index]
                # print(f"✅ Closest engineering match: DN{dn_size} ({dn_inner_diameter}mm) → {closest_size}\" → €{cost}/m")
                return cost
                
        except ImportError:
            print("⚠️ European pipe standards not available")
        
        # Final fallback - use median cost
        median_cost = pipcost_df.iloc[:, col_index].median()
        print(f"⚠️ Using median cost fallback for DN{dn_size}: €{median_cost}/m")
        return median_cost
        
    except Exception as e:
        print(f"❌ Error in get_PipeCost_perMeter: {e}")
        return 0



def get_PipeCost_Total(F1, T1, T2, pipe_type="Stainless"):
    """
    Calculate total pipe cost based on flow, temperatures, and pipe type.
    """
    try:
        # Get cost per meter
        cost_per_meter = get_PipeCost_perMeter(F1, pipe_type)
        if cost_per_meter == 0:
            return 0
        
        # Get total length
        length = get_PipeLength(F1, T1, T2)
        if length == 0:
            return 0
        
        # Calculate total cost
        total_cost = cost_per_meter * length
        return total_cost
        
    except Exception as e:
        print(f"❌ Error in get_PipeCost_Total: {e}")
        return 0


# =============================================================================
# DEFINE SYSTEM FUNCTIONS THAT USE LOOKUPS
# =============================================================================

def get_system_sizing(system_data):

    msg = f"get_system_sizing"
    logger.info(msg)
    print(msg)
    
    msg = f"Input: system_data: {system_data}"
    logger.info(msg)
    print(msg)
    
    if not system_data:
        return None
    
    # Use formula functions for pipe sizing
    pipe_size_f1 = get_PipeSize_Suggested(system_data['F1'])
    pipe_size_f2 = get_PipeSize_Suggested(system_data['F2'])
    
    # Get room size based on power using existing ROOM lookup with data module
    room_size = None
    if is_csv_loaded('ROOM'):
        room_df = get_csv_data('ROOM')
        if room_df is not None:
            room_df = room_df.copy()
            # Convert columns to numeric
            room_df = room_df.astype(float)  # Convert DataFrame to float
            room_df.iloc[:, 0] = room_df.iloc[:, 0].apply(universal_float_convert)  # Power column
            room_df.iloc[:, 1] = room_df.iloc[:, 1].apply(universal_float_convert)  # Room size column
            
            for idx, row in room_df.iterrows():
                power_val = row.iloc[0]
                if power_val >= system_data['power']:
                    room_size = row.iloc[1]
                    break
    
    sizing_data = {
        'pipe_size_f1': pipe_size_f1 or 100,  # Default fallback
        'pipe_size_f2': pipe_size_f2 or 100,
        'room_size': room_size or 12.5,
        'primary_pipe_size': max(pipe_size_f1 or 100, pipe_size_f2 or 100)
    }

    msg = f"get_system_sizing: sizing_data: {sizing_data}"
    logger.info(msg)
    print(msg)
    
    return sizing_data

def calculate_system_costs(system_data, sizing_data):

    msg = f"calculate_system_costs"
    logger.info(msg)
    print(msg)
    msg = f"Inputs: system_data={system_data}, sizing_data={sizing_data}"
    logger.info(msg)
    print(msg)
        
    if not system_data or not sizing_data:
        return None
    
    # Get flow rates for formula calculations
    F1 = system_data['F1']
    F2 = system_data['F2']
    T1 = system_data['T1']
    T2 = system_data['T2']
    
    total_pipe_length = get_PipeLength(F1, T1, T2)
    msg = f"total_pipe_length: {total_pipe_length}"
    logger.info(msg)
    print(msg)
    
    pipe_cost_per_meter = get_PipeCost_perMeter(F1, "Stainless")
    msg = f"pipe_cost_per_meter: {pipe_cost_per_meter}"
    logger.info(msg)
    print(msg)
    
    # Use get_PipeCost_Total formula for total pipe cost
    total_pipe_cost = get_PipeCost_Total(F1, T1, T2, "Stainless")
    msg = f"total_pipe_cost: {total_pipe_cost}"
    logger.info(msg)
    print(msg)
    
    # Calculate valve costs using formula-determined pipe size
    primary_pipe_size = get_PipeSize_Suggested(max(F1, F2))  # Use formula function
    msg = f"primary_pipe_size: {primary_pipe_size}"
    logger.info(msg)
    print(msg)
    
    control_valve_cost = 0
    isolation_valve_cost = 0
    
    # Use data module for CVALV access
    if is_csv_loaded('CVALV'):
        cvalv_df = get_csv_data('CVALV')
        if cvalv_df is not None:
            cvalv_df = cvalv_df.copy()
            # Convert cost column to numeric
            cvalv_df.iloc[:, 1] = cvalv_df.iloc[:, 1].apply(universal_float_convert)
            msg = f"cvalv_df: {cvalv_df}"
            logger.info(msg)
            print(msg)
            
            # Look for exact match on pipe size
            pipe_size_str = str(universal_float_convert(int(primary_pipe_size)))
            msg = f"pipe_size_str: {pipe_size_str}"
            logger.info(msg)
            print(msg)

            for idx, row in cvalv_df.iterrows():
                if str(row.iloc[0]).strip() == pipe_size_str:
                    control_valve_cost = row.iloc[1]
                    msg = f"control_valve_cost: {control_valve_cost}"
                    logger.info(msg)
                    print(msg)
                    break
    
    # Use data module for IVALV access
    if is_csv_loaded('IVALV'):
        ivalv_df = get_csv_data('IVALV')
        if ivalv_df is not None:
            ivalv_df = ivalv_df.copy()
            # Convert cost column to numeric
            ivalv_df.iloc[:, 1] = ivalv_df.iloc[:, 1].apply(universal_float_convert)
            

            for idx, row in ivalv_df.iterrows():
                if str(row.iloc[0]).strip() == pipe_size_str:
                    isolation_valve_cost = row.iloc[1]
                    msg = f"isolation_valve_cost: {isolation_valve_cost}"
                    logger.info(msg)
                    print(msg)
                    break
    
    total_valve_cost = (4 * isolation_valve_cost) + control_valve_cost
    msg = f"total_valve_cost: {total_valve_cost}"
    logger.info(msg)
    print(msg)
    
    # Other costs
    hx_cost = system_data['hx_cost']
    pump_cost = system_data['power'] * 5000  # Estimated
    installation_cost = 10000  # Placeholder
    total_cost = total_pipe_cost + total_valve_cost + hx_cost + pump_cost + installation_cost
    
    cost_data = {
        'pipe_cost_per_meter': pipe_cost_per_meter,
        'total_pipe_length': total_pipe_length,
        'total_pipe_cost': total_pipe_cost,
        'control_valve_cost': control_valve_cost,
        'isolation_valve_cost': isolation_valve_cost,
        'total_valve_cost': total_valve_cost,
        'hx_cost': hx_cost,
        'pump_cost': pump_cost,
        'installation_cost': installation_cost,
        'total_cost': total_cost
    }
    msg = f"cost_data: {cost_data}"
    logger.info(msg)
    print(msg)
    
    return cost_data

def get_complete_system_analysis(power, t1, temp_diff, approach):

    msg = f"get_complete_system_analysis"
    logger.info(msg)
    print(msg)
    
    msg = f"Input: {power}MW, {t1}°C, +{temp_diff}°C, approach {approach}"
    logger.info(msg)
    print(msg)
    
    # Get system data from ALLHX
    system_data = lookup_allhx_data(power, t1, temp_diff, approach)
    if not system_data:
        msg = "❌ ALLHX lookup failed"
        logger.info(msg)
        print(msg)
        return None
    
    # Calculate sizing using corrected formula functions
    sizing_data = get_system_sizing(system_data)
    if not sizing_data:
        msg = "❌ System sizing failed"
        logger.info(msg)
        print(msg)
        return None
    
    # Calculate costs using corrected formula functions
    cost_data = calculate_system_costs(system_data, sizing_data)
    if not cost_data:
        msg = "❌ Cost calculation failed"
        logger.info(msg)
        print(msg)
        return None
    
    
    # Additional validation using formulas
    F1 = system_data['F1']
    F2 = system_data['F2']
    T1 = system_data['T1']
    T2 = system_data['T2']
    T3 = system_data['T3']
    T4 = system_data['T4']
    msg = f"system_data, F1 = {F1}, F2 = {F2}, T1 = {T1}, T2 = {T2}, T3 = {T3}, T4 = {T4}"
    logger.info(msg)
    print(msg)
    
    # Validate calculations using formula functions
    calculated_mw = get_MW_divd(F1, T1, T2)
    delta_t_tcs = get_DeltaT_TCS(T1, T2)
    delta_t_fws = get_DeltaT_FWS(T3, T4)
    approach_calc = get_Approach(T1, T4)

    msg = f"calculated_mw = {calculated_mw}"
    logger.info(msg)
    print(msg)
    msg = f"delta_t_tcs = {delta_t_tcs}"
    logger.info(msg)
    print(msg)
    msg = f"delta_t_fws = {delta_t_fws}"
    logger.info(msg)
    print(msg)
    msg = f"approach_calc = {approach_calc}"
    logger.info(msg)
    print(msg)
    
    # print(f"🔬 Formula validation:")
    # print(f"  Calculated MW: {calculated_mw}")
    # print(f"  Delta T TCS: {delta_t_tcs}°C")
    # print(f"  Delta T FWS: {delta_t_fws}°C")
    # print(f"  Approach: {approach_calc}")
    
    # Combine all data
    complete_analysis = {
        'system': system_data,
        'sizing': sizing_data,
        'costs': cost_data,
        'validation': {
            'calculated_mw': calculated_mw,
            'delta_t_tcs': delta_t_tcs,
            'delta_t_fws': delta_t_fws,
            'approach_calculated': approach_calc
        },
        'summary': {
            'power_mw': system_data['power'],
            't1_celsius': system_data['T1'],
            't2_celsius': system_data['T2'],
            't3_celsius': system_data['T3'],
            't4_celsius': system_data['T4'],
            'f1_flow': system_data['F1'],
            'f2_flow': system_data['F2'],
            'pipe_size': sizing_data['primary_pipe_size'],
            'room_size': sizing_data['room_size'],
            'total_cost_eur': round(cost_data['total_cost'])
        }
    }
    
    # print(f"Complete system analysis finished successfully")
    # print(f"Summary: {system_data['power']}MW system, €{round(cost_data['total_cost']):,} total cost")
    
    return complete_analysis


# Approach calculations

def calculate_tcs_approach_profile(system_data, time_points=10):
    """
    Calculate TCS (internal) approach profile based on system parameters.
    
    Args:
        system_data: System analysis data dictionary
        time_points: Number of points to calculate along the profile
    
    Returns:
        Dictionary with time and temperature/efficiency approach data
    """
    try:
        T1 = float(system_data['T1'])
        T2 = float(system_data['T2'])
        F1 = float(system_data['F1'])
        
        # Create time progression (0 to 100% of process)
        time_progression = [i / (time_points - 1) for i in range(time_points)]
        
        # TCS approach: Temperature reduction profile (descending)
        tcs_temperatures = []
        tcs_efficiency = []
        
        for t in time_progression:
            # Temperature descends from T2 to T1
            temp = T2 - (T2 - T1) * t
            tcs_temperatures.append(temp)
            
            # Efficiency calculation (increases as temperature approaches target)
            efficiency = 50 + 45 * t  # Efficiency improves from 50% to 95%
            tcs_efficiency.append(efficiency)
        
        return {
            'time_progression': time_progression,
            'temperatures': tcs_temperatures,
            'efficiency': tcs_efficiency,
            'flow_rate': F1,
            'start_temp': T2,
            'target_temp': T1,
            'profile_type': 'TCS_Internal'
        }
        
    except Exception as e:
        print(f"Error calculating TCS approach profile: {e}")
        return None

def calculate_fws_approach_profile(system_data, time_points=10):
    """
    Calculate FWS (external) approach profile based on system parameters.
    
    Args:
        system_data: System analysis data dictionary
        time_points: Number of points to calculate along the profile
    
    Returns:
        Dictionary with time and temperature/efficiency approach data
    """
    try:
        T3 = float(system_data['T3'])
        T4 = float(system_data['T4'])
        F2 = float(system_data['F2'])
        
        # Create time progression
        time_progression = [i / (time_points - 1) for i in range(time_points)]
        
        # FWS approach: Temperature reduction profile (descending)
        fws_temperatures = []
        fws_efficiency = []
        
        for t in time_progression:
            # Temperature descends from T3 to T4
            temp = T3 - (T3 - T4) * t
            fws_temperatures.append(temp)
            
            # Different efficiency curve for external system
            efficiency = 40 + 50 * t  # Efficiency improves from 40% to 90%
            fws_efficiency.append(efficiency)
        
        return {
            'time_progression': time_progression,
            'temperatures': fws_temperatures,
            'efficiency': fws_efficiency,
            'flow_rate': F2,
            'start_temp': T3,
            'target_temp': T4,
            'profile_type': 'FWS_External'
        }
        
    except Exception as e:
        print(f"Error calculating FWS approach profile: {e}")
        return None

def calculate_combined_approach_profiles(system_data):
    """
    Calculate both TCS and FWS approach profiles for comparison.
    
    Args:
        system_data: System analysis data dictionary
    
    Returns:
        Dictionary with both approach profiles
    """
    tcs_profile = calculate_tcs_approach_profile(system_data)
    fws_profile = calculate_fws_approach_profile(system_data)
    
    return {
        'tcs_profile': tcs_profile,
        'fws_profile': fws_profile,
        'analysis_type': 'approach_comparison'
    }