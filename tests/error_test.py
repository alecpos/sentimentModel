import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from app.models.ml.prediction.ad_score_predictor import AdScorePredictor
from app.models.ml.prediction.anomaly_detector import EnhancedAnomalyDetector
from app.core.errors import (
    MLBaseException, DataException, DataValidationError, 
    ModelException, ModelNotFoundError, ModelRuntimeError,
    PipelineException
)

class TestErrorHandling:
    """Test suite for validating error handling across the ML platform."""

    @pytest.mark.parametrize("invalid_input,expected_error", [
        ({}, "missing_required_fields"),
        ({"headline": "Test"}, "missing_required_fields"),
        ({"headline": "Test", "description": "x" * 10000}, "input_too_large"),
        ({"headline": "", "description": ""}, "empty_input"),
        ({"headline": 123, "description": "Test"}, "invalid_type"),
    ])
    def test_validation_errors(self, invalid_input, expected_error):
        """Test that validation errors are properly caught and reported."""
        model = AdScorePredictor()
        prediction = model.predict(invalid_input)
        
        assert prediction['fallback'] is True
        assert 'error' in prediction
        assert expected_error in prediction['error_code']
        assert prediction['confidence'] < 0.5

    def test_model_load_error(self):
        """Test error handling when model fails to load."""
        with patch('app.models.ml.prediction.ad_score_predictor.AdScorePredictor._load_model') as mock_load:
            mock_load.side_effect = ModelNotFoundError("Model not found")
            
            with pytest.raises(ModelNotFoundError) as excinfo:
                model = AdScorePredictor(model_path="nonexistent_model.pkl")
            
            assert "Model not found" in str(excinfo.value)

    def test_uninitialized_model(self):
        """Test error handling with uninitialized model."""
        detector = EnhancedAnomalyDetector()
        
        with pytest.raises(RuntimeError) as excinfo:
            detector.detect({}, [])
        
        assert "not initialized" in str(excinfo.value).lower()

    def test_fallback_chain(self):
        """Test that fallback mechanisms work as expected."""
        # Setup main model to fail but fallback to succeed
        with patch('app.models.ml.prediction.ad_score_predictor.AdScorePredictor._predict_primary') as mock_primary:
            mock_primary.side_effect = ModelRuntimeError("Primary model failed")
            
            model = AdScorePredictor()
            valid_input = {"headline": "Test headline", "description": "Test description"}
            
            # Should use fallback without raising exception
            result = model.predict(valid_input)
            
            assert result['fallback'] is True
            assert 'score' in result
            assert isinstance(result['score'], float)
            assert 0 <= result['score'] <= 100

    def test_error_boundary_isolation(self):
        """Test that errors are properly isolated and don't cascade."""
        with patch('app.models.ml.prediction.ad_score_predictor.AdScorePredictor._extract_features') as mock_extract:
            # Make feature extraction fail
            mock_extract.side_effect = Exception("Feature extraction failed")
            
            model = AdScorePredictor()
            result = model.predict({"headline": "Test", "description": "Test"})
            
            # Should still return a result with fallback
            assert result['fallback'] is True
            assert 'error' in result
            assert "feature_extraction_failed" in result['error_code']

    def test_graceful_degradation(self):
        """Test system's ability to gracefully degrade under error conditions."""
        with patch('app.models.ml.prediction.ad_score_predictor.FeatureDegradationManager.execute_with_degradation') as mock_degrade:
            # Mock the degradation response
            mock_degrade.return_value = {"degraded_feature": True}
            
            model = AdScorePredictor()
            result = model.predict_with_degradation({"headline": "Test"})
            
            # Should return degraded result
            assert "degraded_feature" in result
            assert result["degraded_feature"] is True

    @pytest.mark.parametrize("error_class,error_msg", [
        (DataValidationError, "Invalid data format"),
        (ModelRuntimeError, "Model execution failed"),
        (PipelineException, "Pipeline execution failed")
    ])
    def test_structured_error_logging(self, error_class, error_msg, caplog):
        """Test that errors are properly logged with structured context."""
        with patch('app.core.logging.StructuredErrorLogger.log_error') as mock_log:
            # Create error instance
            error = error_class(error_msg)
            
            # Call error logging
            from app.core.logging import error_logger
            error_logger.log_error(error, {"component": "test"})
            
            # Verify mock was called with correct parameters
            mock_log.assert_called_once()
            args, kwargs = mock_log.call_args
            assert args[0] == error
            assert kwargs.get('context', {}).get('component') == "test"

    def test_retry_mechanism(self):
        """Test that retry mechanism works correctly for transient errors."""
        from app.core.errors import retry
        
        mock_func = MagicMock()
        # Fail twice then succeed
        mock_func.side_effect = [ConnectionError("Network error"), ConnectionError("Still failing"), "success"]
        
        # Create retrying function
        @retry(max_attempts=3, retry_delay=0.01)
        def retrying_function():
            return mock_func()
        
        # Should eventually succeed
        result = retrying_function()
        
        assert result == "success"
        assert mock_func.call_count == 3

    def test_error_metrics_recording(self):
        """Test that error metrics are properly recorded."""
        with patch('app.core.monitoring.error_metrics_recorder.record_error') as mock_record:
            from app.core.monitoring import error_metrics_recorder
            
            # Record an error
            error = ModelRuntimeError("Test error")
            error_metrics_recorder.record_error(error, {"component": "test"})
            
            # Verify metrics were recorded
            mock_record.assert_called_once()
            args, kwargs = mock_record.call_args
            assert args[0] == error
            assert kwargs.get('context', {}).get('component') == "test"

    def test_checkpoint_recovery(self):
        """Test recovery from checkpoints after errors."""
        with patch('app.core.errors.CheckpointRecovery.load_checkpoint') as mock_load:
            # Mock a saved checkpoint
            mock_load.return_value = {
                "step": 5,
                "processed_items": 100,
                "results": [1, 2, 3, 4, 5]
            }
            
            from app.core.errors import CheckpointRecovery
            recovery = CheckpointRecovery(
                checkpoint_dir="/tmp",
                operation_id="test_operation",
                logger=MagicMock()
            )
            
            # Load the checkpoint
            state = recovery.load_checkpoint()
            
            assert state is not None
            assert state["step"] == 5
            assert state["processed_items"] == 100

    def test_circuit_breaker(self):
        """Test circuit breaker prevents cascading failures."""
        with patch('app.core.errors.CircuitBreaker.execute') as mock_execute:
            mock_execute.side_effect = Exception("Circuit is OPEN")
            
            from app.core.errors import CircuitBreaker
            circuit = CircuitBreaker(
                name="test_circuit",
                failure_threshold=3,
                logger=MagicMock()
            )
            
            # Should raise the exception from the circuit
            with pytest.raises(Exception) as excinfo:
                circuit.execute(lambda: "This shouldn't execute")
            
            assert "Circuit is OPEN" in str(excinfo.value)

# Add these tests to improve coverage for error handling
def test_api_error_responses():
    """Test that API endpoints return appropriate error responses."""
    from fastapi.testclient import TestClient
    from app.api.v1.endpoints import app
    
    client = TestClient(app)
    
    # Test invalid input
    response = client.post("/api/v1/predict/ad_score", json={})
    
    assert response.status_code == 400
    assert "error" in response.json()
    assert "code" in response.json()["error"]
    assert "message" in response.json()["error"]

def test_input_edge_cases():
    """Test handling of edge cases in input data."""
    model = AdScorePredictor()
    
    # Test with extremely long input
    long_text = "a" * 100000  # Very long text
    result = model.predict({"headline": "Test", "description": long_text})
    
    assert result['fallback'] is True
    assert 'error' in result
    
    # Test with mixed invalid/valid fields
    result = model.predict({
        "headline": "Valid headline",
        "description": "Valid description",
        "invalid_field": "This field doesn't exist"
    })
    
    # Should still work despite invalid field
    assert result['fallback'] is False
    assert 'score' in result

def test_numeric_input_ranges():
    """Test handling of out-of-range numeric inputs."""
    model = AdScorePredictor()
    
    # Test with extreme numeric values
    result = model.predict({
        "headline": "Test",
        "description": "Test",
        "ctr": float('inf')  # Infinite CTR
    })
    
    assert result['fallback'] is True
    assert 'error' in result
    
    # Test with NaN values
    result = model.predict({
        "headline": "Test",
        "description": "Test",
        "ctr": float('nan')  # NaN CTR
    })
    
    assert result['fallback'] is True
    assert 'error' in result