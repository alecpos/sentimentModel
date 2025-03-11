#!/usr/bin/env python3
"""
Test script for the Implementation Status Verification Engine.
This script tests the key improvements to verify they are working correctly.
"""

import os
import sys
import unittest
import tempfile
import json
from verify_implementation_status import ImplementationVerifier

class TestImplementationVerifier(unittest.TestCase):
    """Tests for the enhanced Implementation Verifier."""
    
    def setUp(self):
        """Set up test environment."""
        self.base_dir = "/Users/alecposner/WITHIN/docs"
        self.verifier = ImplementationVerifier(self.base_dir)
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_classify_section_type(self):
        """Test section type classification."""
        test_cases = [
            ("Introduction", "introduction"),
            ("Overview of the API", "introduction"),
            ("Getting Started", "usage"),
            ("API Reference", "api"),
            ("Code Examples", "examples"),
            ("Configuration Options", "configuration"),
            ("Troubleshooting Common Issues", "troubleshooting"),
            ("Related Documentation", "related_docs"),
            ("See Also", "related_docs"),
            ("Random Section", "other")
        ]
        
        for section_name, expected_type in test_cases:
            with self.subTest(section_name=section_name):
                result = self.verifier.classify_section_type(section_name)
                self.assertEqual(result, expected_type, 
                                f"Section '{section_name}' classified as '{result}', expected '{expected_type}'")

    def test_is_meaningful_placeholder(self):
        """Test context-aware placeholder detection."""
        test_cases = [
            # Text, section_type, expected_is_placeholder, expected_confidence_threshold
            ("This is normal text.", "introduction", False, 0.0),
            ("TODO: Add more examples", "introduction", True, 0.7),
            ("This will be implemented in a future release", "api", True, 0.6),
            ("See the TODO items in other files", "related_docs", False, 0.0), # Should ignore in related docs
            ("FIXME: This needs work", "examples", True, 0.8)
        ]
        
        for text, section_type, expected_placeholder, expected_confidence in test_cases:
            with self.subTest(text=text, section_type=section_type):
                is_placeholder, confidence, placeholders = self.verifier.is_meaningful_placeholder(text, section_type)
                
                self.assertEqual(is_placeholder, expected_placeholder, 
                                f"Expected is_placeholder={expected_placeholder}, got {is_placeholder}")
                
                if expected_placeholder:
                    self.assertGreaterEqual(confidence, expected_confidence, 
                                         f"Confidence {confidence} below threshold {expected_confidence}")
                    self.assertTrue(len(placeholders) > 0, "Expected placeholders to be detected")

    def test_detect_document_type(self):
        """Test document type detection."""
        # Test with path-based detection
        api_path = "api/reference/endpoints.md"
        api_content = "# API Reference\n\n## Endpoints\n\nThis is the API documentation."
        
        guide_path = "user_guides/getting_started.md"
        guide_content = "# Getting Started\n\n## Installation\n\nInstall the package."
        
        arch_path = "architecture/decisions/auth_system.md"
        arch_content = "# Authentication System Design\n\n## Overview\n\nThis document describes architecture decisions."
        
        # Test cases
        test_cases = [
            (api_path, api_content, "api_reference"),
            (guide_path, guide_content, "user_guide"),
            (arch_path, arch_content, "architecture_decision")
        ]
        
        for path, content, expected_type in test_cases:
            with self.subTest(path=path):
                result = self.verifier.detect_document_type(path, content)
                self.assertEqual(result, expected_type,
                               f"Document at {path} detected as {result}, expected {expected_type}")

    def test_get_scoring_weights(self):
        """Test adaptive scoring weights."""
        test_cases = [
            ("api_reference", (0.15, 0.25, 0.60)),
            ("user_guide", (0.20, 0.30, 0.50)),
            ("architecture_decision", (0.25, 0.40, 0.35)),
            ("general_documentation", (0.20, 0.30, 0.50)),
            ("unknown_type", (0.20, 0.30, 0.50))  # Should fall back to default
        ]
        
        for doc_type, expected_weights in test_cases:
            with self.subTest(doc_type=doc_type):
                weights = self.verifier.get_scoring_weights(doc_type)
                self.assertEqual(weights, expected_weights,
                               f"Weights for {doc_type} are {weights}, expected {expected_weights}")
    
    def test_placeholder_detection_in_suspicious_patterns(self):
        """Test placeholder detection in suspicious patterns."""
        content = """# Test Document
        
## Introduction
This is an introduction. TODO: Improve this.

## API Reference
This needs more details and will be implemented later.

## See Also
There are TODOs in other documents but they shouldn't count here.
"""
        suspicious_patterns = self.verifier.detect_suspicious_patterns(content)
        
        # Should find placeholders in non-reference sections but not in reference sections
        self.assertTrue(any("Placeholder text in 'Introduction'" in pattern for pattern in suspicious_patterns))
        self.assertTrue(any("Placeholder text in 'API Reference'" in pattern for pattern in suspicious_patterns))
        self.assertFalse(any("Placeholder text in 'See Also'" in pattern for pattern in suspicious_patterns))

def main():
    unittest.main()

if __name__ == "__main__":
    main() 