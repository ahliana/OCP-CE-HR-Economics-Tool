"""
CSV Data Loading and Management

This module handles loading and accessing CSV data files.
The csv_data dictionary is the central repository for all CSV files.
"""

import pandas as pd
import os
from typing import Dict, Optional, Any
from .converter import universal_float_convert

# Global CSV data storage - accessible from all modules
csv_data: Dict[str, pd.DataFrame] = {}

def load_csv_files(data_dir: str = "Data") -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from the specified directory.
    
    Parameters:
    data_dir (str): Path to the directory containing CSV files
    
    Returns:
    dict: Dictionary of dataframes with normalized names as keys
    """
    global csv_data
    
    # List of CSV files to skip (not data files)
    skip_files = [
        'HeatReuse_Data_DictionaryR16_DataDictionaryTypes.csv',
        'readme.csv',
        'config.csv',
        # Add any other non-data CSV files here
    ]
    
    try:
        # Get all CSV files in the directory, excluding skip list
        csv_files = [f for f in os.listdir(data_dir) 
                    if f.endswith('.csv') and f not in skip_files]
        
        # Load each CSV file into a dataframe
        for file in csv_files:
            # Create a normalized name for the dataframe (without .csv extension, uppercase)
            df_name = os.path.splitext(file)[0].upper()
            file_path = os.path.join(data_dir, file)
            
            try:
                # Try to read the CSV file
                csv_data[df_name] = pd.read_csv(file_path)
                # print(f"✅ Loaded: {file} as {df_name}")
            except Exception as e:
                # Try with different separators if automatic detection fails
                try:
                    csv_data[df_name] = pd.read_csv(file_path, sep=';')
                    # print(f"✅ Loaded: {file} as {df_name} (using semicolon separator)")
                except:
                    try:
                        csv_data[df_name] = pd.read_csv(file_path, sep='\\t')
                        # print(f"✅ Loaded: {file} as {df_name} (using tab separator)")
                    except Exception as e2:
                        print(f"❌ Failed to load {file}: {e2}")
        
        return csv_data
    
    except Exception as e:
        print(f"❌ Error loading CSV files: {e}")
        return {}

def get_csv_data(csv_name: str) -> Optional[pd.DataFrame]:
    """
    Get a specific CSV dataframe by name.
    
    Parameters:
    csv_name (str): Name of the CSV file (case-insensitive)
    
    Returns:
    pd.DataFrame or None: The dataframe if found, None otherwise
    """
    global csv_data
    
    # Normalize the CSV name
    csv_name = csv_name.upper()
    
    # Check if the CSV has been loaded
    if csv_name not in csv_data:
        available = list(csv_data.keys())
        raise ValueError(f"❌ CSV '{csv_name}' not loaded. Available CSVs: {available}")
    return csv_data[csv_name]

def is_csv_loaded(csv_name: str) -> bool:
    """Check if a CSV file has been loaded."""
    return csv_name.upper() in csv_data

def list_loaded_csvs() -> list:
    """Get list of all loaded CSV names."""
    return list(csv_data.keys())

def validate_required_csvs(required_csvs: list) -> bool:
    """Validate that all required CSV files are loaded."""
    missing = []
    for csv_name in required_csvs:
        if not is_csv_loaded(csv_name):
            missing.append(csv_name)
    
    if missing:
        available = list_loaded_csvs()
        raise ValueError(f"❌ Missing required CSV files: {missing}. Available: {available}")
    
    return True

# ---

def get_variable_definition(var_name: str) -> dict:
    """
    Get variable definition from data dictionary CSV.
    
    Args:
        var_name: Variable name to look up (e.g., 'T1', 'wha', 'F1')
    
    Returns:
        Dictionary with variable info or None if not found
    """
    try:
        if not is_csv_loaded('HeatReuse_Data_DictionaryR16_DataDictionaryTypes'):
            load_csv_files()  # Try to load if not already loaded
        
        df = get_csv_data('HeatReuse_Data_DictionaryR16_DataDictionaryTypes')
        match = df[df['Variable Name'] == var_name]
        
        if match.empty:
            return None
        
        row = match.iloc[0]
        return {
            'variable_name': row['Variable Name'],
            'data_point_name': row['Data Point Name'],
            'description': row['Data Point Description'],
            'data_type': row['Data Type'],
            'units': row['Units'],
            'default': row.get('Default', ''),
            'range': row.get('Anticipated Range of Values', ''),
            'source': row.get('Source of Input Data', '')
        }
    except Exception as e:
        print(f"Error getting variable definition for {var_name}: {e}")
        return None

def get_variable_units(var_name: str) -> str:
    """Get units for a variable from data dictionary"""
    var_def = get_variable_definition(var_name)
    return var_def['units'] if var_def else ''

def get_variable_description(var_name: str) -> str:
    """Get description for a variable from data dictionary"""
    var_def = get_variable_definition(var_name)
    return var_def['data_point_name'] if var_def else var_name

def validate_variable_type(var_name: str, value) -> bool:
    """Validate that a value matches the expected data type"""
    var_def = get_variable_definition(var_name)
    if not var_def:
        return True  # Unknown variable, assume valid
    
    expected_type = var_def['data_type'].lower()
    
    if expected_type == 'float':
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    elif expected_type == 'int':
        try:
            int(value)
            return True
        except (ValueError, TypeError):
            return False
    elif expected_type == 'string':
        return isinstance(value, str)
    elif expected_type == 'date':
        # Handle date validation (date only, no time)
        if isinstance(value, str):
            import re
            from datetime import datetime
            
            # ISO format (unambiguous) - try first
            iso_patterns = [
                (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),           # 2023-12-31
                (r'^\d{4}/\d{2}/\d{2}$', '%Y/%m/%d'),           # 2023/12/31
            ]
            
            for pattern, fmt in iso_patterns:
                if re.match(pattern, value):
                    try:
                        datetime.strptime(value, fmt)
                        return True
                    except ValueError:
                        continue
            
            # Ambiguous DD/MM or MM/DD patterns - try European first, then US
            if re.match(r'^\d{2}/\d{2}/\d{4}$', value):  # Pattern like 31/12/2023 or 12/31/2023
                # Try European format first (DD/MM/YYYY)
                try:
                    datetime.strptime(value, '%d/%m/%Y')
                    return True
                except ValueError:
                    # If European fails, try US format (MM/DD/YYYY)
                    try:
                        datetime.strptime(value, '%m/%d/%Y') 
                        return True
                    except ValueError:
                        pass
            
            if re.match(r'^\d{2}-\d{2}-\d{4}$', value):  # Pattern like 31-12-2023 or 12-31-2023
                # Try European format first (DD-MM-YYYY)
                try:
                    datetime.strptime(value, '%d-%m-%Y')
                    return True
                except ValueError:
                    # If European fails, try US format (MM-DD-YYYY)
                    try:
                        datetime.strptime(value, '%m-%d-%Y')
                        return True
                    except ValueError:
                        pass
            
            return False
        return False
    elif expected_type == 'datetime':
        # Handle datetime validation (date + time)
        if isinstance(value, str):
            import re
            from datetime import datetime
            
            # Unambiguous ISO datetime patterns
            iso_datetime_patterns = [
                (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$', '%Y-%m-%dT%H:%M:%S'),                    # 2023-12-31T12:30:45
                (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}$', '%Y-%m-%dT%H:%M:%S.%f'),         # 2023-12-31T12:30:45.123
                (r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', '%Y-%m-%d %H:%M:%S'),                   # 2023-12-31 12:30:45
                (r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$', '%Y-%m-%dT%H:%M:%SZ'),                 # 2023-12-31T12:30:45Z (UTC)
            ]
            
            # Try unambiguous ISO formats first
            for pattern, fmt in iso_datetime_patterns:
                if re.match(pattern, value):
                    try:
                        datetime.strptime(value, fmt)
                        return True
                    except ValueError:
                        continue
            
            # Handle ambiguous DD/MM vs MM/DD datetime patterns
            if re.match(r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$', value):  # 31/12/2023 12:30:45 or 12/31/2023 12:30:45
                # Try European format first (DD/MM/YYYY HH:MM:SS)
                try:
                    datetime.strptime(value, '%d/%m/%Y %H:%M:%S')
                    return True
                except ValueError:
                    # If European fails, try US format (MM/DD/YYYY HH:MM:SS)
                    try:
                        datetime.strptime(value, '%m/%d/%Y %H:%M:%S')
                        return True
                    except ValueError:
                        pass
            
            if re.match(r'^\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2}$', value):  # 31-12-2023 12:30:45 or 12-31-2023 12:30:45
                # Try European format first (DD-MM-YYYY HH:MM:SS)
                try:
                    datetime.strptime(value, '%d-%m-%Y %H:%M:%S')
                    return True
                except ValueError:
                    # If European fails, try US format (MM-DD-YYYY HH:MM:SS)
                    try:
                        datetime.strptime(value, '%m-%d-%Y %H:%M:%S')
                        return True
                    except ValueError:
                        pass
            
            # Also accept date-only formats for datetime fields (try both European and US)
            if re.match(r'^\d{2}/\d{2}/\d{4}$', value):  # Date only, no time
                # Try European format first
                try:
                    datetime.strptime(value, '%d/%m/%Y')
                    return True
                except ValueError:
                    try:
                        datetime.strptime(value, '%m/%d/%Y')
                        return True
                    except ValueError:
                        pass
            
            if re.match(r'^\d{2}-\d{2}-\d{4}$', value):  # Date only, no time
                # Try European format first
                try:
                    datetime.strptime(value, '%d-%m-%Y')
                    return True
                except ValueError:
                    try:
                        datetime.strptime(value, '%m-%d-%Y')
                        return True
                    except ValueError:
                        pass
            
            # ISO date fallback
            if re.match(r'^\d{4}-\d{2}-\d{2}$', value):  # 2023-12-31
                try:
                    datetime.strptime(value, '%Y-%m-%d')
                    return True
                except ValueError:
                    pass
            
            return False
        return False
    elif expected_type == 'boolean':
        return isinstance(value, bool) or value in ['true', 'false', 'True', 'False', '1', '0', 1, 0]
    elif expected_type == 'hyperlink':
        # Validate URL format
        if isinstance(value, str):
            import re
            url_pattern = r'^https?://[\w\-._~:/?#[\]@!$&\'()*+,;=%]+$'
            return re.match(url_pattern, value) is not None
        return False
    
    return True  # Default to true for unknown types

def format_hyperlink_for_display(url: str, display_text: str = None) -> str:
    """
    Format a hyperlink for display in UI.
    
    Args:
        url: The URL string
        display_text: Optional display text (defaults to URL)
    
    Returns:
        HTML anchor tag string for display
    """
    if not url:
        return ""
    
    # Ensure URL has protocol
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    display = display_text or url
    return f'<a href="{url}" target="_blank">{display}</a>'

def extract_url_from_hyperlink(hyperlink_value: str) -> str:
    """
    Extract URL from various hyperlink formats.
    
    Args:
        hyperlink_value: Could be plain URL, HTML anchor, markdown, etc.
    
    Returns:
        Clean URL string
    """
    if not hyperlink_value:
        return ""
    
    import re
    
    # If it's already a plain URL
    if hyperlink_value.startswith(('http://', 'https://')):
        return hyperlink_value
    
    # Extract from HTML anchor tag
    html_match = re.search(r'href=["\']([^"\']+)["\']', hyperlink_value)
    if html_match:
        return html_match.group(1)
    
    # Extract from markdown format
    md_match = re.search(r'\[([^\]]+)\]\(([^)]+)\)', hyperlink_value)
    if md_match:
        return md_match.group(2)
    
    # If none of the above, treat as plain URL
    return hyperlink_value


def get_hyperlink_variables() -> dict:
    """Get all hyperlink variables from data dictionary"""
    try:
        df = get_csv_data('HeatReuse_Data_DictionaryR16_DataDictionaryTypes')
        hyperlink_rows = df[df['Data Type'] == 'hyperlink']
        
        hyperlink_vars = {}
        for _, row in hyperlink_rows.iterrows():
            var_name = row['Variable Name']
            description = row['Data Point Name']
            hyperlink_vars[var_name] = description
            
        return hyperlink_vars
    except:
        # Fallback to hardcoded list
        return {
            'soweblink': 'Site Owner Website or Links',
            'itweblink': 'Tenant Website or Links',
            'howeblink': 'Heat Offtaker Owner Website or Links', 
            'huweblink': 'Heat Offtaker Website or Links'
        }
