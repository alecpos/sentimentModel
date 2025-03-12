#!/usr/bin/env python3
"""
Ad Score Router Check Script

This script checks for the existence of ad_score_router in the routes module
without importing the entire application.

DOCUMENTATION STATUS: COMPLETE
"""

import os
import sys
from pathlib import Path
import importlib.util


def check_file_for_router(file_path: Path) -> bool:
    """
    Check if a file contains a router definition.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        bool: True if ad_score_router is defined in the file
    """
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
        
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Check for ad_score_router definition
    if "ad_score_router" in content:
        print(f"Found 'ad_score_router' reference in {file_path}")
        
        # Check if it's defined as a router
        router_patterns = [
            "ad_score_router = APIRouter",
            "ad_score_router=APIRouter",
            "ad_score_router =APIRouter",
            "ad_score_router=FastAPI",
            "ad_score_router = FastAPI",
        ]
        
        for pattern in router_patterns:
            if pattern in content:
                print(f"  ✓ Found router definition: '{pattern}'")
                return True
                
        print("  ✗ 'ad_score_router' is referenced but not defined as a router in this file")
    return False
    

def check_routes_directory(routes_dir: Path) -> None:
    """
    Check all files in the routes directory for ad_score_router.
    
    Args:
        routes_dir: Path to the routes directory
    """
    print(f"Checking routes directory: {routes_dir}")
    print("=" * 80)
    
    if not routes_dir.exists() or not routes_dir.is_dir():
        print(f"Directory not found: {routes_dir}")
        return
        
    found = False
    for file_path in routes_dir.glob("*.py"):
        print(f"Checking file: {file_path}")
        if check_file_for_router(file_path):
            found = True
            
    if not found:
        print("\nNo ad_score_router definition found in routes directory.")
        print("Possible fixes:")
        print("1. Create an ad_score_router in a file in the routes directory.")
        print("2. Update the import statement in __init__.py to match the actual router name.")
        print("3. Check for typos in the router name.")


def check_init_file(init_file: Path) -> None:
    """
    Check the __init__.py file for router imports and exports.
    
    Args:
        init_file: Path to the __init__.py file
    """
    print(f"\nChecking init file: {init_file}")
    print("=" * 80)
    
    if not init_file.exists():
        print(f"File not found: {init_file}")
        return
        
    with open(init_file, 'r') as f:
        content = f.read()
        
    print("\nImport statements in __init__.py:")
    import_lines = [line for line in content.split("\n") if "import" in line]
    for line in import_lines:
        print(f"  {line}")
        
    print("\nRouter references in __init__.py:")
    router_lines = [line for line in content.split("\n") if "router" in line.lower()]
    for line in router_lines:
        print(f"  {line}")
        
    # Check for ad_score_router specifically
    if "ad_score_router" in content:
        print("\n✓ 'ad_score_router' is referenced in __init__.py")
    else:
        print("\n✗ 'ad_score_router' is NOT referenced in __init__.py")


def main():
    """Main function."""
    # Get the app directory
    app_dir = Path("app")
    if not app_dir.exists():
        print("App directory not found. Please run this script from the project root.")
        return 1
        
    # Check routes directory
    routes_dir = app_dir / "api" / "v1" / "routes"
    check_routes_directory(routes_dir)
    
    # Check __init__.py file
    init_file = app_dir / "api" / "v1" / "__init__.py"
    check_init_file(init_file)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 