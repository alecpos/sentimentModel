#!/usr/bin/env python3
"""Test file for docstring indentation issues."""

import numpy as np
import torch

# This class has proper indentation
class ProperClass:
    """
    This docstring has proper indentation.
    
    It should not be modified by the script.
    """
    
    def __init__(self):
        pass

# This class has improper indentation
class ImproperClass:
    """
    This docstring has improper indentation.
    
    It should be fixed by the script.
    """

    def __init__(self):
        pass

# This class has a nested class with improper indentation
class OuterClass:
    """
    This is an outer class with proper indentation.
    """
    
    def __init__(self):
        pass
        
    # Nested class with improper indentation
    class NestedClass:
        """
        This nested class docstring has improper indentation.
        
        It should be fixed by the script.
        """
    
        def __init__(self):
            pass

# Indentation with different spacing
class WeirdIndentationClass:
    """
    This docstring has weird indentation (2 spaces instead of 4).
    
    It should be fixed to have 4 spaces.
    """
  
    def __init__(self):
        pass 