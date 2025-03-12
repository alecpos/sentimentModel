#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Suite for Model Card Generator

This module contains tests for the ModelCardGenerator to ensure it meets
regulatory requirements and correctly documents fairness considerations.
"""

import os
import json
import pytest
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import the module to test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_card_generator import ModelCardGenerator, generate_model_card_for_ad_score_predictor

class TestModelCardGenerator:
    """Test suite for the ModelCardGenerator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model_info(self):
        """Create sample model information for testing."""
        return {
            'name': 'Test Ad Score Predictor',
            'version': '1.0.0',
            'type': 'Regression/Classification',
            'description': 'A test model for predicting ad performance.',
            'use_cases': ['Testing', 'Validation'],
            'developers': [{'name': 'Test Team', 'role': 'Testing'}],
            'date_created': '2023-11-01',
            'model_architecture': {
                'framework': 'PyTorch',
                'type': 'Neural Network',
                'hidden_layers': 64,
                'activation': 'ReLU',
                'dropout': 0.2
            }
        }
    
    @pytest.fixture
    def performance_metrics(self):
        """Create sample performance metrics for testing."""
        return {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85
        }
    
    @pytest.fixture
    def fairness_metrics(self):
        """Create sample fairness metrics for testing."""
        return {
            'metrics': {
                'gender_demographic_parity': {
                    'difference': 0.05,
                    'passes_threshold': True
                },
                'gender_equal_opportunity': {
                    'difference': 0.03,
                    'passes_threshold': True
                }
            },
            'group_metrics': {
                'gender': {
                    'male': {
                        'count': 600,
                        'accuracy': 0.86,
                        'positive_rate': 0.75
                    },
                    'female': {
                        'count': 400,
                        'accuracy': 0.83,
                        'positive_rate': 0.70
                    }
                }
            },
            'threshold': 0.1
        }
    
    def test_init(self, temp_dir):
        """Test initialization of ModelCardGenerator."""
        generator = ModelCardGenerator(output_dir=temp_dir)
        
        assert generator.output_dir == temp_dir
        assert generator.organization == 'WITHIN'
        assert isinstance(generator.regulatory_frameworks, list)
        assert len(generator.regulatory_frameworks) > 0
        
        # Check if directories were created
        assert os.path.exists(temp_dir)
        assert os.path.exists(os.path.join(temp_dir, 'assets'))
    
    def test_generate_model_card_basic(self, temp_dir, model_info, performance_metrics):
        """Test generating a basic model card without fairness metrics."""
        generator = ModelCardGenerator(output_dir=temp_dir)
        
        model_card_path = generator.generate_model_card(
            model_info=model_info,
            performance_metrics=performance_metrics,
            export_formats=['md']
        )
        
        # Check if the model card was created
        assert os.path.exists(model_card_path)
        
        # Read the model card and check content
        with open(model_card_path, 'r') as f:
            content = f.read()
        
        # Check that key information is included
        assert model_info['name'] in content
        assert model_info['version'] in content
        assert model_info['description'] in content
        
        # Check that performance metrics are included
        for metric, value in performance_metrics.items():
            assert metric in content
            assert str(value) in content
    
    def test_generate_model_card_with_fairness(self, temp_dir, model_info, 
                                              performance_metrics, fairness_metrics):
        """Test generating a model card with fairness metrics."""
        generator = ModelCardGenerator(output_dir=temp_dir)
        
        model_card_path = generator.generate_model_card(
            model_info=model_info,
            performance_metrics=performance_metrics,
            fairness_metrics=fairness_metrics,
            export_formats=['md']
        )
        
        # Check if the model card was created
        assert os.path.exists(model_card_path)
        
        # Read the model card and check content
        with open(model_card_path, 'r') as f:
            content = f.read()
        
        # Check that fairness metrics are included
        assert 'Fairness Evaluation' in content
        assert 'demographic_parity' in content
        assert 'equal_opportunity' in content
        
        # Check group metrics
        assert 'male' in content
        assert 'female' in content
    
    def test_generate_model_card_with_mitigations(self, temp_dir, model_info, 
                                                performance_metrics, fairness_metrics):
        """Test generating a model card with fairness mitigations."""
        generator = ModelCardGenerator(output_dir=temp_dir)
        
        mitigation_strategies = {
            'Reweighing': {
                'description': 'Assigns weights to examples to ensure fairness.',
                'implementation': 'ReweighingMitigation',
                'parameters': {'protected_attribute': 'gender'},
                'effectiveness': 'Reduced disparity by 50%'
            }
        }
        
        model_card_path = generator.generate_model_card(
            model_info=model_info,
            performance_metrics=performance_metrics,
            fairness_metrics=fairness_metrics,
            mitigation_strategies=mitigation_strategies,
            export_formats=['md']
        )
        
        # Check if the model card was created
        assert os.path.exists(model_card_path)
        
        # Read the model card and check content
        with open(model_card_path, 'r') as f:
            content = f.read()
        
        # Check that mitigation strategies are included
        assert 'Fairness Mitigations' in content
        assert 'Reweighing' in content
        assert 'Assigns weights' in content
        assert 'Reduced disparity' in content
    
    def test_generate_from_evaluation_results(self, temp_dir, model_info):
        """Test generating a model card from evaluation results."""
        generator = ModelCardGenerator(output_dir=temp_dir)
        
        evaluation_results = {
            'overall': {
                'accuracy': 0.85,
                'precision': 0.82
            },
            'fairness_metrics': {
                'gender_demographic_parity': {
                    'difference': 0.05,
                    'passes_threshold': True
                }
            },
            'group_metrics': {
                'gender': {
                    'male': {'count': 600, 'accuracy': 0.86},
                    'female': {'count': 400, 'accuracy': 0.83}
                }
            },
            'fairness_threshold': 0.1
        }
        
        model_card_path = generator.generate_from_evaluation_results(
            model_info=model_info,
            evaluation_results=evaluation_results,
            export_formats=['md']
        )
        
        # Check if the model card was created
        assert os.path.exists(model_card_path)
        
        # Read the model card and check content
        with open(model_card_path, 'r') as f:
            content = f.read()
        
        # Check that evaluation results are properly included
        assert 'accuracy' in content
        assert 'demographic_parity' in content
        assert 'male' in content
        assert 'female' in content
    
    def test_regulatory_compliance(self, temp_dir, model_info, 
                                  performance_metrics, fairness_metrics):
        """Test regulatory compliance aspects of the model card."""
        # Use specific regulatory frameworks
        regulatory_frameworks = [
            'EU AI Act', 
            'NYC Local Law 144', 
            'NIST AI Risk Management Framework'
        ]
        
        generator = ModelCardGenerator(
            output_dir=temp_dir,
            regulatory_frameworks=regulatory_frameworks,
            organization='TestOrg'
        )
        
        model_card_path = generator.generate_model_card(
            model_info=model_info,
            performance_metrics=performance_metrics,
            fairness_metrics=fairness_metrics,
            export_formats=['md']
        )
        
        # Check if the model card was created
        assert os.path.exists(model_card_path)
        
        # Read the model card and check content
        with open(model_card_path, 'r') as f:
            content = f.read()
        
        # Check that regulatory frameworks are included
        assert 'Regulatory Compliance' in content
        for framework in regulatory_frameworks:
            assert framework in content
        
        # Check organization information
        assert 'TestOrg' in content
    
    def test_intersectional_fairness(self, temp_dir, model_info, 
                                    performance_metrics):
        """Test model card with intersectional fairness metrics."""
        generator = ModelCardGenerator(output_dir=temp_dir)
        
        # Create fairness metrics with intersectional analysis
        fairness_metrics = {
            'metrics': {
                'gender_demographic_parity': {
                    'difference': 0.05,
                    'passes_threshold': True
                }
            },
            'intersectional': {
                'fairness_metrics': {
                    'gender+location_demographic_parity': {
                        'difference': 0.07,
                        'passes_threshold': True
                    }
                },
                'group_metrics': {
                    'gender+location': {
                        'male+urban': {'positive_rate': 0.76},
                        'female+urban': {'positive_rate': 0.70}
                    }
                }
            },
            'threshold': 0.1
        }
        
        model_card_path = generator.generate_model_card(
            model_info=model_info,
            performance_metrics=performance_metrics,
            fairness_metrics=fairness_metrics,
            export_formats=['md']
        )
        
        # Check if the model card was created
        assert os.path.exists(model_card_path)
        
        # Read the model card and check content
        with open(model_card_path, 'r') as f:
            content = f.read()
        
        # Check that intersectional analysis is included
        assert 'Intersectional Analysis' in content
        assert 'gender+location' in content
    
    def test_export_html(self, temp_dir, model_info, performance_metrics):
        """Test exporting the model card to HTML format."""
        generator = ModelCardGenerator(output_dir=temp_dir)
        
        model_card_path = generator.generate_model_card(
            model_info=model_info,
            performance_metrics=performance_metrics,
            export_formats=['md', 'html']
        )
        
        # Check if both MD and HTML files were created
        assert os.path.exists(model_card_path)
        assert os.path.exists(model_card_path.replace('.md', '.html'))
        
        # Read the HTML file and check content
        with open(model_card_path.replace('.md', '.html'), 'r') as f:
            content = f.read()
        
        # Check that HTML formatting is included
        assert '<!DOCTYPE html>' in content
        assert '<html>' in content
        assert '<body>' in content
        assert '</html>' in content
        
        # Check that key information is included
        assert model_info['name'] in content
    
    @patch('weasyprint.HTML')
    def test_export_pdf(self, mock_weasyprint, temp_dir, model_info, performance_metrics):
        """Test exporting the model card to PDF format."""
        # Mock weasyprint.HTML().write_pdf()
        mock_html_instance = MagicMock()
        mock_weasyprint.return_value = mock_html_instance
        
        generator = ModelCardGenerator(output_dir=temp_dir)
        
        model_card_path = generator.generate_model_card(
            model_info=model_info,
            performance_metrics=performance_metrics,
            export_formats=['md', 'pdf']
        )
        
        # Check if MD file was created
        assert os.path.exists(model_card_path)
        
        # Verify that weasyprint was called (would create PDF in real scenario)
        mock_weasyprint.assert_called_once()
        mock_html_instance.write_pdf.assert_called_once()
    
    def test_metadata_generation(self, temp_dir, model_info, 
                               performance_metrics, fairness_metrics):
        """Test generation of metadata alongside model card."""
        generator = ModelCardGenerator(output_dir=temp_dir)
        
        model_card_path = generator.generate_model_card(
            model_info=model_info,
            performance_metrics=performance_metrics,
            fairness_metrics=fairness_metrics,
            export_formats=['md']
        )
        
        # Check if metadata file was created
        metadata_path = model_card_path.replace('.md', '_metadata.json')
        assert os.path.exists(metadata_path)
        
        # Read metadata and verify content
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        assert 'model_info' in metadata
        assert 'performance_metrics' in metadata
        assert 'fairness_metrics' in metadata
        assert 'generation_date' in metadata
        
        # Verify specific content
        assert metadata['model_info']['name'] == model_info['name']
        assert metadata['performance_metrics']['accuracy'] == performance_metrics['accuracy']
    
    def test_ad_score_predictor_helper(self, temp_dir):
        """Test the helper function for generating Ad Score Predictor model cards."""
        # Create a mock model
        class MockModel:
            def __init__(self):
                self.hidden_dim = 64
                self.dropout = 0.2
                self.learning_rate = 0.01
                self.batch_size = 32
                self.epochs = 100
        
        model = MockModel()
        
        # Mock evaluation results
        evaluation_results = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85
        }
        
        # Mock fairness evaluation results
        fairness_evaluation_results = {
            'overall': evaluation_results,
            'fairness_metrics': {
                'gender_demographic_parity': {
                    'difference': 0.05,
                    'passes_threshold': True
                }
            },
            'group_metrics': {
                'gender': {
                    'male': {'count': 600, 'accuracy': 0.86},
                    'female': {'count': 400, 'accuracy': 0.83}
                }
            },
            'fairness_threshold': 0.1
        }
        
        # Mock mitigation info
        mitigation_info = {
            'Reweighing': {
                'description': 'Assigns weights to ensure fairness.',
                'implementation': 'ReweighingMitigation',
                'parameters': {'protected_attribute': 'gender'},
                'effectiveness': 'Reduced disparity by 50%'
            }
        }
        
        with patch('model_card_generator.ModelCardGenerator.generate_from_evaluation_results') as mock_generate:
            mock_generate.return_value = os.path.join(temp_dir, 'test_model_card.md')
            
            # Call the helper function
            model_card_path = generate_model_card_for_ad_score_predictor(
                model=model,
                evaluation_results=evaluation_results,
                model_name="TestAdScorePredictor",
                model_version="1.0.0",
                output_dir=temp_dir,
                fairness_evaluation_results=fairness_evaluation_results,
                mitigation_info=mitigation_info
            )
            
            # Verify the helper function called generate_from_evaluation_results
            mock_generate.assert_called_once()
            
            # Check arguments
            args, kwargs = mock_generate.call_args
            assert kwargs['model_info']['name'] == "TestAdScorePredictor"
            assert kwargs['model_info']['version'] == "1.0.0"
            assert kwargs['evaluation_results'] == fairness_evaluation_results
            assert kwargs['mitigation_info'] == mitigation_info

if __name__ == "__main__":
    pytest.main() 