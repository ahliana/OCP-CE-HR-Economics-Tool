import json
import sys
import os

def safe_analysis():
    try:
        # Try reading with UTF-8 first
        with open('variable_analysis_report.json', 'r', encoding='utf-8') as f:
            results = json.load(f)
    except UnicodeDecodeError:
        # If UTF-8 fails, try with latin-1 (more permissive)
        try:
            with open('variable_analysis_report.json', 'r', encoding='latin-1') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            print("Try regenerating the scanner results.")
            return
    except FileNotFoundError:
        print("variable_analysis_report.json not found")
        print("Run: python variable_scanner.py")
        return
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return

    print("=== VARIABLE ANALYSIS RESULTS ===")
    
    # Key variables that need mapping
    key_mappings = {
        'power': 'wha (Total Heat Available)',
        'temp_diff': 'itdt (Temperature difference across IT side)', 
        't1': 'T1 (consistent capitalization)',
        't2': 'T2 (consistent capitalization)',
        'f1': 'F1 (consistent capitalization)',
        'f2': 'F2 (consistent capitalization)'
    }
    
    print("\nVariables that need updating:")
    total_updates = 0
    
    for old_var, description in key_mappings.items():
        if old_var in results.get('variable_usage', {}):
            count = len(results['variable_usage'][old_var])
            total_updates += count
            print(f"  {old_var} -> {description}")
            print(f"    {count} usages found")
            
            # Show which files use this variable
            files = set()
            for usage in results['variable_usage'][old_var]:
                files.add(usage['file'])
            print(f"    Files: {', '.join(list(files)[:3])}{'...' if len(files) > 3 else ''}")
        else:
            print(f"  {old_var} -> {description}")
            print(f"    0 usages found")
    
    print(f"\nTotal updates needed: {total_updates}")
    
    # Summary stats
    if 'summary' in results:
        summary = results['summary']
        print(f"\nScan Summary:")
        print(f"  Files scanned: {summary.get('total_files', 'unknown')}")
        print(f"  Variables found: {summary.get('total_variables', 'unknown')}")
        print(f"  Functions found: {summary.get('total_functions', 'unknown')}")
        print(f"  Widget names: {summary.get('total_widget_names', 'unknown')}")

if __name__ == "__main__":
    safe_analysis()
