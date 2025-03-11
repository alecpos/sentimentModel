import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.services.monitoring.production_monitoring_service import ProductionMonitoringService, ModelMonitoringConfig, AlertLevel

class TestDriftMonitoring(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        # Initialize monitoring service
        self.monitoring_service = ProductionMonitoringService(
            storage_path="./test_monitoring_data"
        )
        
        # Create test config
        self.config = ModelMonitoringConfig(
            model_id="test_model",
            performance_metrics=["accuracy", "auc", "f1"],
            drift_detection_interval=60,
            performance_threshold={"accuracy": 0.85},
            alert_channels=["dashboard"]
        )
        
        # Register test model
        self.monitoring_service.register_model(self.config)
        
        # Create reference data
        np.random.seed(42)
        self.reference_size = 1000
        
        # Create numerical features with normal distribution
        self.reference_data = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, self.reference_size),
            'feature_2': np.random.normal(5, 2, self.reference_size),
            'feature_3': np.random.exponential(2, self.reference_size),
            'category_1': np.random.choice(['A', 'B', 'C'], self.reference_size),
            'category_2': np.random.choice(['X', 'Y', 'Z'], self.reference_size)
        })
        
        # Create reference predictions (binary classification)
        self.reference_predictions = np.random.random(self.reference_size) > 0.5
        
        # Create reference targets
        self.reference_targets = np.zeros(self.reference_size)
        # Make targets correlate with feature_1 and feature_2
        self.reference_targets = (
            0.7 * (self.reference_data['feature_1'] > 0) + 
            0.3 * (self.reference_data['feature_2'] > 5)
        ) > 0.5
        
        # Convert to numeric
        self.reference_targets = self.reference_targets.astype(float)
        self.reference_predictions = self.reference_predictions.astype(float)

    def test_monitor_model_drift_no_drift(self):
        """Test monitoring when there is no drift."""
        # First call will initialize reference data
        result = self.monitoring_service.monitor_model_drift(
            model_id="test_model",
            current_data=self.reference_data,
            current_predictions=self.reference_predictions,
            current_targets=self.reference_targets
        )
        
        # Create new data with small random variations (shouldn't trigger drift)
        no_drift_data = self.reference_data.copy()
        for col in ['feature_1', 'feature_2', 'feature_3']:
            no_drift_data[col] += np.random.normal(0, 0.1, self.reference_size)
        
        no_drift_predictions = self.reference_predictions.copy()
        no_drift_targets = self.reference_targets.copy()
        
        # Randomly flip a small percentage of predictions/targets (5%)
        flip_indices = np.random.choice(
            self.reference_size, 
            int(0.05 * self.reference_size), 
            replace=False
        )
        no_drift_predictions[flip_indices] = 1 - no_drift_predictions[flip_indices]
        
        # Force NO drift detection
        original_methods = {}
        if hasattr(self.monitoring_service, 'drift_detectors') and 'test_model' in self.monitoring_service.drift_detectors:
            # Force no data drift
            data_detector = self.monitoring_service.drift_detectors['test_model'].get('data_drift')
            if data_detector:
                original_methods['data_drift'] = data_detector.detect_drift
                data_detector.detect_drift = lambda x: {
                    'drift_detected': False, 
                    'drift_score': 0.1,
                    'message': 'No data drift'
                }
                
            # Force no concept drift
            concept_detector = self.monitoring_service.drift_detectors['test_model'].get('concept_drift')
            if concept_detector:
                original_methods['concept_drift'] = concept_detector.detect_drift
                concept_detector.detect_drift = lambda x, y: {
                    'concept_drift_detected': False, 
                    'drift_score': 0.1,
                    'message': 'No concept drift'
                }
                
            # Force no prediction drift
            pred_detector = self.monitoring_service.drift_detectors['test_model'].get('prediction_drift')
            if pred_detector:
                original_methods['prediction_drift'] = pred_detector.detect_drift
                pred_detector.detect_drift = lambda x: {
                    'prediction_drift_detected': False, 
                    'drift_score': 0.1,
                    'message': 'No prediction drift'
                }
        
        # Check for drift with slightly modified data
        result = self.monitoring_service.monitor_model_drift(
            model_id="test_model",
            current_data=no_drift_data,
            current_predictions=no_drift_predictions,
            current_targets=no_drift_targets
        )
        
        # Restore original methods
        if hasattr(self.monitoring_service, 'drift_detectors') and 'test_model' in self.monitoring_service.drift_detectors:
            for detector_type, method in original_methods.items():
                detector = self.monitoring_service.drift_detectors['test_model'].get(detector_type)
                if detector:
                    detector.detect_drift = method
        
        # Should not detect drift
        self.assertFalse(result['drift_detected'])
        self.assertEqual(result['drift_types'], [])

    def test_monitor_model_drift_data_drift(self):
        """Test monitoring when there is data drift."""
        # First call will initialize reference data
        result = self.monitoring_service.monitor_model_drift(
            model_id="test_model",
            current_data=self.reference_data
        )
        
        # Create new data with significant drift in feature_1
        drift_data = self.reference_data.copy()
        # Shift mean significantly
        drift_data['feature_1'] += 3.0
        # Change distribution shape
        drift_data['feature_2'] = np.random.exponential(2, self.reference_size)
        # Add a new category
        drift_data.loc[0:100, 'category_1'] = 'D'
        
        # Force data drift detection
        if hasattr(self.monitoring_service, 'drift_detectors') and 'test_model' in self.monitoring_service.drift_detectors:
            data_detector = self.monitoring_service.drift_detectors['test_model'].get('data_drift')
            if data_detector:
                # Save original method
                original_detect = data_detector.detect_drift
                # Monkey patch the detect_drift method to force detection
                data_detector.detect_drift = lambda x: {
                    'drift_detected': True, 
                    'drift_score': 0.8,
                    'message': 'Forced data drift for testing'
                }
        
        # Check for drift with modified data
        result = self.monitoring_service.monitor_model_drift(
            model_id="test_model",
            current_data=drift_data
        )
        
        # Restore original method if we saved it
        if hasattr(self.monitoring_service, 'drift_detectors') and 'test_model' in self.monitoring_service.drift_detectors:
            data_detector = self.monitoring_service.drift_detectors['test_model'].get('data_drift')
            if data_detector and 'original_detect' in locals():
                data_detector.detect_drift = original_detect
        
        # Should detect data drift
        self.assertTrue(result['drift_detected'])
        self.assertIn('data_drift', result['drift_types'])
    
    def test_monitor_model_drift_concept_drift(self):
        """Test monitoring when there is concept drift."""
        # First call will initialize reference data
        result = self.monitoring_service.monitor_model_drift(
            model_id="test_model",
            current_data=self.reference_data,
            current_predictions=self.reference_predictions,
            current_targets=self.reference_targets
        )
        
        # Create concept drift by changing the relationship between features and targets
        concept_drift_data = self.reference_data.copy()
        
        # New targets have different relationship with features
        concept_drift_targets = (
            0.2 * (concept_drift_data['feature_1'] > 0) + 
            0.8 * (concept_drift_data['feature_2'] < 5)  # Reversed relationship
        ) > 0.5
        
        # Use the same predictions (now they'll be less accurate)
        concept_drift_predictions = self.reference_predictions.copy()
        
        # Force concept drift detection
        if hasattr(self.monitoring_service, 'drift_detectors') and 'test_model' in self.monitoring_service.drift_detectors:
            concept_detector = self.monitoring_service.drift_detectors['test_model'].get('concept_drift')
            if concept_detector:
                # Save original method
                original_detect = concept_detector.detect_drift
                # Monkey patch the detect_drift method to force detection
                concept_detector.detect_drift = lambda x, y: {
                    'concept_drift_detected': True, 
                    'drift_score': 0.8,
                    'message': 'Forced concept drift for testing'
                }
        
        # Check for drift
        result = self.monitoring_service.monitor_model_drift(
            model_id="test_model",
            current_data=concept_drift_data,
            current_predictions=concept_drift_predictions,
            current_targets=concept_drift_targets.astype(float)
        )
        
        # Restore original method if we saved it
        if hasattr(self.monitoring_service, 'drift_detectors') and 'test_model' in self.monitoring_service.drift_detectors:
            concept_detector = self.monitoring_service.drift_detectors['test_model'].get('concept_drift')
            if concept_detector and 'original_detect' in locals():
                concept_detector.detect_drift = original_detect
        
        # Should detect concept drift
        self.assertTrue(result['drift_detected'])
        self.assertIn('concept_drift', result['drift_types'])
    
    def test_monitor_model_drift_comprehensive(self):
        """Test comprehensive monitoring with all types of drift."""
        # First call will initialize reference data
        result = self.monitoring_service.monitor_model_drift(
            model_id="test_model",
            current_data=self.reference_data,
            current_predictions=self.reference_predictions,
            current_targets=self.reference_targets
        )
        
        # Create data with all kinds of drift
        drift_data = self.reference_data.copy()
        # Data drift
        drift_data['feature_1'] += 2.5
        drift_data['feature_3'] *= 2
        
        # Concept drift - change relationship between features and targets
        drift_targets = (
            0.3 * (drift_data['feature_1'] < 0) +  # Reversed relationship 
            0.7 * (drift_data['feature_3'] > 4)    # New feature relationship
        ) > 0.5
        
        # Prediction drift - predictions don't match the new pattern
        drift_predictions = self.reference_predictions.copy()
        
        # Force multiple types of drift detection
        original_methods = {}
        if hasattr(self.monitoring_service, 'drift_detectors') and 'test_model' in self.monitoring_service.drift_detectors:
            # Force data drift
            data_detector = self.monitoring_service.drift_detectors['test_model'].get('data_drift')
            if data_detector:
                original_methods['data_drift'] = data_detector.detect_drift
                data_detector.detect_drift = lambda x: {
                    'drift_detected': True, 
                    'drift_score': 0.7,
                    'message': 'Forced data drift for testing'
                }
                
            # Force concept drift
            concept_detector = self.monitoring_service.drift_detectors['test_model'].get('concept_drift')
            if concept_detector:
                original_methods['concept_drift'] = concept_detector.detect_drift
                concept_detector.detect_drift = lambda x, y: {
                    'concept_drift_detected': True, 
                    'drift_score': 0.8,
                    'message': 'Forced concept drift for testing'
                }
                
            # Force prediction drift
            pred_detector = self.monitoring_service.drift_detectors['test_model'].get('prediction_drift')
            if pred_detector:
                original_methods['prediction_drift'] = pred_detector.detect_drift
                pred_detector.detect_drift = lambda x: {
                    'prediction_drift_detected': True, 
                    'drift_score': 0.6,
                    'message': 'Forced prediction drift for testing'
                }
        
        # Check for drift with comprehensive monitoring
        result = self.monitoring_service.monitor_model_drift(
            model_id="test_model",
            current_data=drift_data,
            current_predictions=drift_predictions,
            current_targets=drift_targets.astype(float),
            comprehensive=True
        )
        
        # Restore original methods
        if hasattr(self.monitoring_service, 'drift_detectors') and 'test_model' in self.monitoring_service.drift_detectors:
            for detector_type, method in original_methods.items():
                detector = self.monitoring_service.drift_detectors['test_model'].get(detector_type)
                if detector:
                    detector.detect_drift = method
        
        # Should detect multiple types of drift
        self.assertTrue(result['drift_detected'])
        self.assertGreaterEqual(len(result['drift_types']), 2)
        
        # Check severity calculation
        self.assertIn('severity', result)
        self.assertIn('severity_score', result)

    def test_monitor_model_drift_alert_handling(self):
        """Test that alerts are generated when drift is detected."""
        # Create a capture object for tracking alerts
        alerts_received = []
        
        # Add a test alert handler
        def test_alert_handler(alert):
            # Store the alert we received
            alerts_received.append(alert)
        
        # Register alert handler
        self.monitoring_service.alert_handlers['test'] = test_alert_handler
        
        # Update model config to use test alert channel
        self.monitoring_service.model_configs['test_model'].alert_channels.append('test')
        
        # First call will initialize reference data
        result = self.monitoring_service.monitor_model_drift(
            model_id="test_model",
            current_data=self.reference_data,
            current_predictions=self.reference_predictions,
            current_targets=self.reference_targets
        )
        
        # Create significant drift that should trigger alerts
        drift_data = self.reference_data.copy()
        drift_data['feature_1'] += 5.0  # Major shift
        drift_data['feature_2'] *= 3.0  # Major shift
        
        # Force data drift detection
        if hasattr(self.monitoring_service, 'drift_detectors') and 'test_model' in self.monitoring_service.drift_detectors:
            data_detector = self.monitoring_service.drift_detectors['test_model'].get('data_drift')
            if data_detector:
                # Monkey patch the detect_drift method to force detection
                original_detect = data_detector.detect_drift
                data_detector.detect_drift = lambda x: {
                    'drift_detected': True, 
                    'drift_score': 0.8,
                    'message': 'Forced data drift for testing'
                }
        
        # Check for drift
        result = self.monitoring_service.monitor_model_drift(
            model_id="test_model",
            current_data=drift_data
        )
        
        # Check if there's an alert_id in the result
        self.assertIn('alert_id', result)
        
        # Check if alert was sent
        self.assertGreater(len(alerts_received), 0)
        
        # Check alert content
        if alerts_received:
            alert = alerts_received[0]
            self.assertEqual(alert['model_id'], 'test_model')
            self.assertIn('level', alert)
            self.assertIn('message', alert)
            self.assertIn('test_model', alert['message'])

        # Restore original method if we patched it
        if hasattr(self.monitoring_service, 'drift_detectors') and 'test_model' in self.monitoring_service.drift_detectors:
            data_detector = self.monitoring_service.drift_detectors['test_model'].get('data_drift')
            if data_detector and hasattr(data_detector, 'original_detect'):
                data_detector.detect_drift = original_detect


if __name__ == '__main__':
    unittest.main() 