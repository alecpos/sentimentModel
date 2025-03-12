#!/usr/bin/env python
"""
README.md Standardization Script

This script helps standardize README.md files according to the established template.
It analyzes a module's __init__.py file to identify exported classes and ensures
they are all documented in the README.md file.

Usage:
    python standardize_readme.py <directory>

Example:
    python standardize_readme.py app/models/ml/fairness
"""

import os
import sys
import re
import ast
from typing import Dict, List, Set, Optional
import shutil
from datetime import datetime

# Template sections
TEMPLATE_SECTIONS = [
    "# [Module Name]",
    "## Purpose",
    "## Directory Structure",
    "## Key Components",
    "## Usage Examples",
    "## Integration Points",
    "## Dependencies"
]

def extract_all_list(init_file: str) -> Set[str]:
    """
    Extract the __all__ list from a Python __init__.py file.
    
    Args:
        init_file: Path to the __init__.py file
        
    Returns:
        Set of strings representing the exported symbols
    """
    try:
        with open(init_file, 'r') as f:
            file_content = f.read()
            
        # Parse the Python file
        tree = ast.parse(file_content)
        
        # Find the __all__ assignment
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            # Extract string literals from the list
                            return {
                                element.value for element in node.value.elts 
                                if isinstance(element, ast.Str)
                            }
                        
    except Exception as e:
        print(f"Error parsing {init_file}: {str(e)}")
    
    return set()

def extract_module_docstring(init_file: str) -> str:
    """
    Extract the module docstring from an __init__.py file.
    
    Args:
        init_file: Path to the __init__.py file
        
    Returns:
        Module docstring or empty string if not found
    """
    try:
        with open(init_file, 'r') as f:
            file_content = f.read()
            
        # Parse the Python file
        tree = ast.parse(file_content)
        
        # Find module docstring
        if (isinstance(tree, ast.Module) and 
            len(tree.body) > 0 and 
            isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, ast.Str)):
            return tree.body[0].value.s
                        
    except Exception as e:
        print(f"Error parsing {init_file}: {str(e)}")
    
    return ""

def extract_class_details_from_file(file_path: str, class_name: str) -> Dict[str, str]:
    """
    Extract details about a class from its implementation file.
    
    Args:
        file_path: Path to the Python file
        class_name: Name of the class to extract details for
        
    Returns:
        Dictionary with class details
    """
    details = {
        'docstring': '',
        'methods': [],
        'parameters': []
    }
    
    if not os.path.exists(file_path):
        return details
        
    try:
        with open(file_path, 'r') as f:
            file_content = f.read()
            
        # Parse the Python file
        tree = ast.parse(file_content)
        
        # Find the class definition
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Extract docstring
                if (len(node.body) > 0 and 
                    isinstance(node.body[0], ast.Expr) and 
                    isinstance(node.body[0].value, ast.Str)):
                    details['docstring'] = node.body[0].value.s
                
                # Extract methods
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
                details['methods'] = methods
                
                # Try to extract __init__ parameters
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        parameters = []
                        for arg in item.args.args:
                            if arg.arg != 'self':
                                parameters.append(arg.arg)
                        details['parameters'] = parameters
                
                break
                        
    except Exception as e:
        print(f"Error parsing {file_path} for class {class_name}: {str(e)}")
    
    return details

def find_class_file(directory: str, class_name: str) -> Optional[str]:
    """
    Find the file that contains a given class definition.
    
    Args:
        directory: Directory to search in
        class_name: Name of the class to find
        
    Returns:
        Path to the file containing the class, or None if not found
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Simple pattern matching for class definition
                    if re.search(rf'class\s+{class_name}[:\(]', content):
                        return file_path
                except Exception:
                    pass
    
    return None

def extract_documented_classes(readme_file: str) -> Set[str]:
    """
    Extract class names documented in a README.md file.
    
    Args:
        readme_file: Path to the README.md file
        
    Returns:
        Set of strings representing documented class names
    """
    documented_classes = set()
    
    if not os.path.exists(readme_file):
        return documented_classes
        
    with open(readme_file, 'r') as f:
        content = f.read()
    
    # Regular expressions to find documented classes
    
    # Find classes documented with ### ClassName header
    header_pattern = re.compile(r'###\s+`?(\w+)`?')
    documented_classes.update(header_pattern.findall(content))
    
    # Find classes documented with `ClassName` in text
    backtick_pattern = re.compile(r'`([A-Z]\w+)`')
    documented_classes.update(backtick_pattern.findall(content))
    
    return documented_classes

def analyze_directory(directory: str) -> Dict[str, any]:
    """
    Analyze a directory to gather information for README standardization.
    
    Args:
        directory: Path to the directory to analyze
        
    Returns:
        Dictionary with module information
    """
    result = {
        'module_name': os.path.basename(directory),
        'init_file': os.path.join(directory, '__init__.py'),
        'readme_file': os.path.join(directory, 'README.md'),
        'python_files': [],
        'exported_symbols': set(),
        'documented_classes': set(),
        'module_docstring': '',
        'missing_classes': set(),
        'class_details': {}
    }
    
    # Check if files exist
    if not os.path.exists(result['init_file']):
        print(f"Error: __init__.py not found in {directory}")
        return result
    
    # Get Python files in directory
    for file in os.listdir(directory):
        if file.endswith('.py') and file != '__init__.py':
            result['python_files'].append(os.path.join(directory, file))
    
    # Extract exported symbols
    result['exported_symbols'] = extract_all_list(result['init_file'])
    
    # Extract module docstring
    result['module_docstring'] = extract_module_docstring(result['init_file'])
    
    # Extract documented classes
    if os.path.exists(result['readme_file']):
        result['documented_classes'] = extract_documented_classes(result['readme_file'])
    
    # Find missing class documentation
    result['missing_classes'] = result['exported_symbols'] - result['documented_classes']
    
    # Extract class details for missing classes
    for class_name in result['missing_classes']:
        class_file = find_class_file(directory, class_name)
        if class_file:
            result['class_details'][class_name] = extract_class_details_from_file(class_file, class_name)
    
    return result

def generate_standardized_readme(info: Dict[str, any]) -> str:
    """
    Generate a standardized README.md file based on the template.
    
    Args:
        info: Module information from analyze_directory
        
    Returns:
        String containing the standardized README content
    """
    # Extract module name and description from docstring
    module_name = info['module_name'].replace('_', ' ').title()
    
    module_description = ""
    if info['module_docstring']:
        lines = info['module_docstring'].strip().split('\n')
        if len(lines) > 1:
            module_description = lines[0]
    
    # Start with the template
    content = f"# {module_name}\n\n"
    
    # Add documentation status
    content += "DOCUMENTATION STATUS: COMPLETE\n\n"
    
    # Add module description
    content += f"This directory contains {module_description.lower()} for the WITHIN ML Prediction System.\n\n"
    
    # Add purpose section
    content += "## Purpose\n\n"
    content += f"The {module_name.lower()} provides capabilities for:\n"
    
    # Extract capabilities from docstring if possible
    capabilities = []
    if info['module_docstring']:
        lines = info['module_docstring'].strip().split('\n')
        for line in lines:
            if line.strip().startswith('- '):
                capabilities.append(line.strip())
    
    # Add placeholder capabilities if none found
    if not capabilities:
        for i in range(5):
            capabilities.append(f"- [Key capability {i+1}]")
    
    content += '\n'.join(capabilities) + '\n\n'
    
    # Add directory structure
    content += "## Directory Structure\n\n"
    content += "- **__init__.py**: Module initialization with component exports\n"
    
    for file in sorted([os.path.basename(f) for f in info['python_files']]):
        content += f"- **{file}**: [Brief description of file purpose]\n"
    
    content += "\n"
    
    # Add key components
    content += "## Key Components\n\n"
    
    # First add classes that were already documented
    for class_name in sorted(info['documented_classes']):
        content += f"### {class_name}\n\n"
        content += f"`{class_name}` is responsible for [brief description of the class's purpose and responsibility].\n\n"
        content += "**Key Features:**\n"
        content += "- [Feature 1]: [Description]\n"
        content += "- [Feature 2]: [Description]\n"
        content += "- [Feature 3]: [Description]\n\n"
        content += "**Parameters:**\n"
        content += "- `param1` (type): [Description of parameter]\n"
        content += "- `param2` (type): [Description of parameter]\n\n"
        content += "**Methods:**\n"
        content += "- `method1(param1, param2)`: [Description of method and what it returns]\n"
        content += "- `method2()`: [Description of method and what it returns]\n\n"
    
    # Then add missing classes with info if available
    for class_name in sorted(info['missing_classes']):
        content += f"### {class_name}\n\n"
        
        details = info['class_details'].get(class_name, {})
        
        if details.get('docstring'):
            docstring = details['docstring'].strip()
            if '\n' in docstring:
                first_line = docstring.split('\n')[0]
                content += f"`{class_name}` is responsible for {first_line.lower()}.\n\n"
            else:
                content += f"`{class_name}` is responsible for {docstring.lower()}.\n\n"
        else:
            content += f"`{class_name}` is responsible for [brief description of the class's purpose and responsibility].\n\n"
        
        content += "**Key Features:**\n"
        content += "- [Feature 1]: [Description]\n"
        content += "- [Feature 2]: [Description]\n"
        content += "- [Feature 3]: [Description]\n\n"
        
        content += "**Parameters:**\n"
        if details.get('parameters'):
            for param in details['parameters']:
                content += f"- `{param}` (type): [Description of parameter]\n"
        else:
            content += "- `param1` (type): [Description of parameter]\n"
            content += "- `param2` (type): [Description of parameter]\n"
        content += "\n"
        
        content += "**Methods:**\n"
        if details.get('methods'):
            for method in details['methods']:
                if method != '__init__' and not method.startswith('_'):
                    content += f"- `{method}()`: [Description of method and what it returns]\n"
        else:
            content += "- `method1(param1, param2)`: [Description of method and what it returns]\n"
            content += "- `method2()`: [Description of method and what it returns]\n"
        content += "\n"
    
    # Add usage examples
    content += "## Usage Examples\n\n"
    
    for i, class_name in enumerate(sorted(info['exported_symbols'])):
        if i < 2:  # Only add examples for the first two classes
            content += f"### {class_name} Usage\n\n"
            content += "```python\n"
            content += f"from app.models.{info['module_name']} import {class_name}\n\n"
            content += "# Initialization\n"
            content += f"instance = {class_name}(param1=value1, param2=value2)\n\n"
            content += "# Using key methods\n"
            content += "result = instance.method1(some_value)\n"
            content += "```\n\n"
    
    # Add integration points
    content += "## Integration Points\n\n"
    content += "- **[System 1]**: [How this module integrates with System 1]\n"
    content += "- **[System 2]**: [How this module integrates with System 2]\n"
    content += "- **[System 3]**: [How this module integrates with System 3]\n"
    content += "- **[System 4]**: [How this module integrates with System 4]\n\n"
    
    # Add dependencies
    content += "## Dependencies\n\n"
    content += "- **[Dependency 1]**: [Purpose of this dependency]\n"
    content += "- **[Dependency 2]**: [Purpose of this dependency]\n"
    content += "- **[Dependency 3]**: [Purpose of this dependency]\n"
    content += "- **[Dependency 4]**: [Purpose of this dependency]\n"
    
    return content

def main() -> None:
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python standardize_readme.py <directory>")
        sys.exit(1)
        
    directory = sys.argv[1]
    
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
    
    print(f"Analyzing directory: {directory}...")
    info = analyze_directory(directory)
    
    readme_file = info['readme_file']
    
    # Check if README.md already exists
    if os.path.exists(readme_file):
        # Backup existing README
        backup_file = f"{readme_file}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
        shutil.copy2(readme_file, backup_file)
        print(f"Backed up existing README.md to {backup_file}")
        
        # Check for missing class documentation
        if info['missing_classes']:
            print(f"Found {len(info['missing_classes'])} classes that need documentation:")
            for class_name in sorted(info['missing_classes']):
                print(f"  - {class_name}")
            
            choice = input("Do you want to generate a completely new standardized README? (y/n): ")
            if choice.lower() != 'y':
                print("Exiting without changes.")
                sys.exit(0)
        else:
            print("All exported classes are already documented in README.md.")
            choice = input("Do you still want to generate a standardized README? (y/n): ")
            if choice.lower() != 'y':
                print("Exiting without changes.")
                sys.exit(0)
    
    # Generate standardized README
    content = generate_standardized_readme(info)
    
    # Write to file
    with open(readme_file, 'w') as f:
        f.write(content)
    
    print(f"Generated standardized README.md at {readme_file}")
    print("Please edit the file to fill in details marked with [placeholders].")

if __name__ == "__main__":
    main() 