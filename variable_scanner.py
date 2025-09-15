"""
Variable Usage Scanner
Automated tool to scan all Python files and identify variable usage patterns
"""

import os
import re
import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

class VariableUsageScanner:
    """Scan Python files for variable usage patterns"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.variable_usage = defaultdict(list)
        self.function_signatures = []
        self.string_literals = []
        self.dictionary_keys = []
        self.widget_names = []
        
    def scan_project(self) -> Dict:
        """Scan entire project for variable usage"""
        python_files = list(self.project_root.rglob("*.py"))
        
        results = {
            'files_scanned': [],
            'variable_usage': {},
            'function_parameters': {},
            'dictionary_keys': set(),
            'string_literals': set(),
            'widget_names': set(),
            'data_type_usage': {},
            'unit_references': set()
        }
        
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
                
            file_results = self.scan_file(file_path)
            results['files_scanned'].append(str(file_path))
            
            # Merge results
            for var_name, locations in file_results['variables'].items():
                if var_name not in results['variable_usage']:
                    results['variable_usage'][var_name] = []
                results['variable_usage'][var_name].extend(locations)
            
            results['function_parameters'].update(file_results['functions'])
            results['dictionary_keys'].update(file_results['dict_keys'])
            results['string_literals'].update(file_results['strings'])
            results['widget_names'].update(file_results['widgets'])
            results['unit_references'].update(file_results['units'])
        
        return results
    
    def scan_file(self, file_path: Path) -> Dict:
        """Scan a single Python file for variable patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return self._empty_results()
        
        results = {
            'variables': defaultdict(list),
            'functions': {},
            'dict_keys': set(),
            'strings': set(),
            'widgets': set(),
            'units': set()
        }
        
        # Parse the file with AST for accurate analysis
        try:
            tree = ast.parse(content)
            visitor = VariableVisitor(file_path, results)
            visitor.visit(tree)
        except SyntaxError as e:
            print(f"Syntax error in {file_path}: {e}")
        
        # Also use regex for patterns AST might miss
        self._scan_with_regex(content, file_path, results)
        
        return results
    
    def _scan_with_regex(self, content: str, file_path: Path, results: Dict):
        """Use regex to find patterns that AST might miss"""
        
        # Find dictionary key patterns like df['column'] or data['key']
        dict_key_pattern = r"[\w\[\]\.]+\['([^']+)'\]"
        for match in re.finditer(dict_key_pattern, content):
            key = match.group(1)
            results['dict_keys'].add(key)
            results['variables'][key].append({
                'file': str(file_path),
                'type': 'dictionary_key',
                'line': content[:match.start()].count('\n') + 1,
                'context': match.group(0)
            })
        
        # Find widget patterns like widgets_dict['power_widget']
        widget_pattern = r"widget.*['\"]([\w_]+)['\"]"
        for match in re.finditer(widget_pattern, content, re.IGNORECASE):
            widget = match.group(1)
            results['widgets'].add(widget)
            if widget.endswith('_widget'):
                var_name = widget.replace('_widget', '')
                results['variables'][var_name].append({
                    'file': str(file_path),
                    'type': 'widget_name',
                    'line': content[:match.start()].count('\n') + 1,
                    'context': match.group(0)
                })
        
        # Find unit references
        unit_patterns = [
            r'\b(°C|C|celsius|fahrenheit|F)\b',
            r'\b(MW|kW|W|watts?|megawatts?|kilowatts?)\b',
            r'\b(L/min|l/m|GPM|gallons?|liters?)\b',
            r'\b(bar|psi|pascal|Pa)\b',
            r'\b(€|EUR|USD|\$|dollars?|euros?)\b'
        ]
        for pattern in unit_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                results['units'].add(match.group(0))
        
        # Find string literals that might contain variable references
        string_pattern = r'["\']([^"\']*(?:T1|T2|T3|T4|F1|F2|power|temp|flow|cost|heat)[^"\']*)["\']'
        for match in re.finditer(string_pattern, content, re.IGNORECASE):
            results['strings'].add(match.group(1))
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped"""
        skip_patterns = [
            '__pycache__',
            '.git',
            '.venv',
            'node_modules',
            '.pytest_cache'
        ]
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _empty_results(self) -> Dict:
        """Return empty results structure"""
        return {
            'variables': defaultdict(list),
            'functions': {},
            'dict_keys': set(),
            'strings': set(),
            'widgets': set(),
            'units': set()
        }

class VariableVisitor(ast.NodeVisitor):
    """AST visitor to find variable usage patterns"""
    
    def __init__(self, file_path: Path, results: Dict):
        self.file_path = file_path
        self.results = results
        self.current_line = 1
    
    def visit_Name(self, node):
        """Visit variable names"""
        var_name = node.id
        if self._is_relevant_variable(var_name):
            self.results['variables'][var_name].append({
                'file': str(self.file_path),
                'type': 'variable_name',
                'line': getattr(node, 'lineno', 0),
                'context': var_name
            })
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Visit function definitions to capture parameters"""
        func_name = node.name
        params = [arg.arg for arg in node.args.args]
        
        self.results['functions'][func_name] = {
            'file': str(self.file_path),
            'parameters': params,
            'line': getattr(node, 'lineno', 0)
        }
        
        # Add parameters to variable usage
        for param in params:
            if self._is_relevant_variable(param):
                self.results['variables'][param].append({
                    'file': str(self.file_path),
                    'type': 'function_parameter',
                    'line': getattr(node, 'lineno', 0),
                    'context': f'def {func_name}({", ".join(params)})'
                })
        
        self.generic_visit(node)
    
    def visit_Subscript(self, node):
        """Visit dictionary/list subscripts"""
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            key = node.slice.value
            self.results['dict_keys'].add(key)
            if self._is_relevant_variable(key):
                self.results['variables'][key].append({
                    'file': str(self.file_path),
                    'type': 'subscript_key',
                    'line': getattr(node, 'lineno', 0),
                    'context': f"['{key}']"
                })
        self.generic_visit(node)
    
    def _is_relevant_variable(self, var_name: str) -> bool:
        """Check if variable name is relevant for our analysis"""
        relevant_patterns = [
            r'^[Tt][1-4]$',  # T1, T2, T3, T4, t1, t2, etc.
            r'^[Ff][1-2]$',  # F1, F2, f1, f2
            r'power',
            r'temp',
            r'flow',
            r'cost',
            r'heat',
            r'wha',
            r'itdt',
            r'approach',
            r'effectiveness'
        ]
        return any(re.search(pattern, var_name, re.IGNORECASE) for pattern in relevant_patterns)

def generate_analysis_report(scan_results: Dict, output_file: str = "variable_analysis_report.json"):
    """Generate a comprehensive analysis report"""
    
    # Convert sets to lists for JSON serialization
    serializable_results = {}
    for key, value in scan_results.items():
        if isinstance(value, set):
            serializable_results[key] = list(value)
        elif isinstance(value, dict):
            serializable_results[key] = dict(value)
        else:
            serializable_results[key] = value
    
    # Add summary statistics
    serializable_results['summary'] = {
        'total_files': len(scan_results['files_scanned']),
        'total_variables': len(scan_results['variable_usage']),
        'total_functions': len(scan_results['function_parameters']),
        'total_dict_keys': len(scan_results['dictionary_keys']),
        'total_string_literals': len(scan_results['string_literals']),
        'total_widget_names': len(scan_results['widget_names']),
        'total_unit_references': len(scan_results['unit_references'])
    }
    
    # Save to file with explicit UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis report saved to {output_file}")
    return serializable_results

# Usage example
if __name__ == "__main__":
    # Scan the project
    scanner = VariableUsageScanner("./")  # Current directory
    results = scanner.scan_project()
    
    # Generate report
    report = generate_analysis_report(results)
    
    # Print summary
    print("\n=== VARIABLE USAGE ANALYSIS SUMMARY ===")
    for key, value in report['summary'].items():
        print(f"{key}: {value}")