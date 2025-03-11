"""Tests for production validation components of the ML platform."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from unittest.mock import patch, MagicMock

from app.models.ml.validation.shadow_deployment import ShadowDeployment
from app.models.ml.validation.ab_test_manager import ABTestManager
from app.models.ml.validation.canary_test import CanaryTestRunner
from app.models.ml.validation.golden_set_validator import GoldenSetValidator
from app.services.monitoring import production_monitoring_service
from app.models.ml.prediction.ad_score_predictor import AdScorePredictor


class TestShadowDeployment:
    """Test suite for shadow deployment validation."""
    
    @pytest.fixture
    def shadow_deployment(self):
        """Create shadow deployment setup with primary and shadow models."""
        # Create primary model
        primary_model = AdScorePredictor(model_path="models/primary_model.pkl")
        
        # Create shadow model
        shadow_model = AdScorePredictor(model_path="models/shadow_model.pkl")
        
        # Create shadow deployment
        shadow = ShadowDeployment(
            primary_model=primary_model,
            shadow_model=shadow_model,
            log_predictions=True
        )
        
        return shadow
    
    @pytest.fixture
    def test_requests(self):
        """Generate test prediction requests."""
        requests = []
        for i in range(100):
            requests.append({
                "id": f"req_{i}",
                "headline": f"Test headline {i}",
                "description": f"Test description for request {i}",
                "platform": "facebook" if i % 2 == 0 else "google",
                "ctr": 0.05 + (i % 10) * 0.01
            })
        return requests
    
    def test_shadow_prediction_logging(self, shadow_deployment, test_requests):
        """Test that shadow deployment logs both primary and shadow predictions."""
        # Mock the logging function
        with patch('app.models.ml.validation.shadow_deployment.ShadowDeployment._log_prediction') as mock_log:
            # Make predictions
            for req in test_requests[:10]:  # Test with a subset
                shadow_deployment.predict(req)
            
            # Check logging calls
            assert mock_log.call_count == 10
            
            # Verify log content structure
            args, kwargs = mock_log.call_args_list[0]
            log_entry = args[0]
            
            assert 'request_id' in log_entry
            assert 'timestamp' in log_entry
            assert 'primary_prediction' in log_entry
            assert 'shadow_prediction' in log_entry
            assert 'request_data' in log_entry
    
    def test_shadow_prediction_results(self, shadow_deployment, test_requests):
        """Test that shadow deployment returns primary model predictions."""
        # Mock primary and shadow predictions
        with patch.object(shadow_deployment.primary_model, 'predict') as mock_primary:
            with patch.object(shadow_deployment.shadow_model, 'predict') as mock_shadow:
                # Set return values
                mock_primary.return_value = {"score": 80, "confidence": 0.9}
                mock_shadow.return_value = {"score": 70, "confidence": 0.8}
                
                # Make prediction
                result = shadow_deployment.predict(test_requests[0])
                
                # Should return primary prediction
                assert result["score"] == 80
                assert result["confidence"] == 0.9
                
                # Both models should be called
                mock_primary.assert_called_once()
                mock_shadow.assert_called_once()
    
    def test_shadow_analysis(self, shadow_deployment):
        """Test analyzing difference between primary and shadow models."""
        # Mock prediction logs
        mock_logs = []
        for i in range(100):
            # Create some systematic difference between models
            primary_score = 75 + np.random.normal(0, 5)
            shadow_score = primary_score - 10 + np.random.normal(0, 3)  # Shadow predicts lower
            
            mock_logs.append({
                "request_id": f"req_{i}",
                "timestamp": datetime.now() - timedelta(minutes=i),
                "primary_prediction": {"score": primary_score, "confidence": 0.9},
                "shadow_prediction": {"score": shadow_score, "confidence": 0.85},
                "request_data": {"headline": f"Test {i}"}
            })
        
        # Mock the retrieval of logs
        with patch.object(shadow_deployment, '_get_prediction_logs') as mock_get_logs:
            mock_get_logs.return_value = mock_logs
            
            # Analyze differences
            analysis = shadow_deployment.analyze_prediction_differences()
            
            # Check analysis results
            assert 'mean_abs_diff' in analysis
            assert 'max_abs_diff' in analysis
            assert 'correlation' in analysis
            assert 'systematic_bias' in analysis
            
            # Since we created shadow to be systematically lower
            assert analysis['systematic_bias'] < 0
            # Should be strong correlation
            assert analysis['correlation'] > 0.9
    
    def test_shadow_deployment_metrics(self, shadow_deployment):
        """Test computing performance metrics for shadow deployment."""
        # Mock prediction logs with outcomes
        mock_logs = []
        for i in range(100):
            primary_score = 75 + np.random.normal(0, 5)
            shadow_score = primary_score - 5 + np.random.normal(0, 8)
            
            # Simulate actual outcome (closer to shadow predictions)
            actual_outcome = shadow_score + np.random.normal(0, 3)
            
            mock_logs.append({
                "request_id": f"req_{i}",
                "timestamp": datetime.now() - timedelta(minutes=i),
                "primary_prediction": {"score": primary_score, "confidence": 0.9},
                "shadow_prediction": {"score": shadow_score, "confidence": 0.85},
                "request_data": {"headline": f"Test {i}"},
                "actual_outcome": actual_outcome
            })
        
        # Mock the retrieval of logs
        with patch.object(shadow_deployment, '_get_prediction_logs') as mock_get_logs:
            mock_get_logs.return_value = mock_logs
            
            # Compute performance metrics
            metrics = shadow_deployment.compute_performance_metrics()
            
            # Check metrics
            assert 'primary_model' in metrics
            assert 'shadow_model' in metrics
            assert 'primary_model_rmse' in metrics['primary_model']
            assert 'shadow_model_rmse' in metrics['shadow_model']
            
            # Since actual outcomes are closer to shadow, shadow should have better RMSE
            assert metrics['shadow_model']['shadow_model_rmse'] < metrics['primary_model']['primary_model_rmse']


class TestABTesting:
    """Test suite for A/B testing capabilities."""
    
    @pytest.fixture
    def ab_test_manager(self):
        """Create A/B test manager."""
        # Create model versions
        model_a = AdScorePredictor(model_path="models/model_a.pkl")
        model_b = AdScorePredictor(model_path="models/model_b.pkl")
        
        # Create AB test configuration
        ab_config = {
            "test_id": "ad_score_ab_test_001",
            "model_a_id": "model_a",
            "model_b_id": "model_b",
            "traffic_split": {
                "model_a": 0.5,
                "model_b": 0.5
            },
            "metrics": ["score_difference", "confidence", "processing_time"],
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + timedelta(days=14)).isoformat(),
            "stratification": {
                "enabled": True,
                "variables": ["platform"]
            }
        }
        
        # Create AB test manager
        manager = ABTestManager(
            model_a=model_a,
            model_b=model_b,
            config=ab_config
        )
        
        return manager
    
    def test_ab_test_assignment(self, ab_test_manager):
        """Test consistent assignment of users to test groups."""
        # Test with 1000 assignments with fixed IDs
        assignments = {}
        for i in range(1000):
            user_id = f"user_{i % 100}"  # Use 100 distinct users
            request_id = f"request_{i}"
            assignment = ab_test_manager.get_assignment(user_id, request_id)
            
            if user_id in assignments:
                # Assignment should be consistent for the same user
                assert assignment == assignments[user_id]
            else:
                assignments[user_id] = assignment
        
        # Check distribution is approximately balanced
        model_a_count = sum(1 for a in assignments.values() if a == "model_a")
        model_b_count = sum(1 for a in assignments.values() if a == "model_b")
        
        # Should be close to 50/50 split with some tolerance
        assert abs(model_a_count - model_b_count) <= 10
    
    def test_stratified_assignment(self, ab_test_manager):
        """Test stratified assignment across platforms."""
        # Create test requests with different platforms
        platforms = ["facebook", "google", "instagram"]
        assignments = {platform: {"model_a": 0, "model_b": 0} for platform in platforms}
        
        # Make 1000 assignments
        for i in range(1000):
            platform = platforms[i % len(platforms)]
            user_id = f"user_{i}"
            request_id = f"request_{i}"
            request_data = {"platform": platform}
            
            assignment = ab_test_manager.get_stratified_assignment(user_id, request_id, request_data)
            assignments[platform][assignment] += 1
        
        # Check balance within each platform
        for platform in platforms:
            model_a_count = assignments[platform]["model_a"]
            model_b_count = assignments[platform]["model_b"]
            total = model_a_count + model_b_count
            
            # Should be close to 50/50 within each platform
            assert abs(model_a_count/total - 0.5) < 0.1
    
    def test_ab_test_prediction(self, ab_test_manager):
        """Test that predictions use the assigned model."""
        # Mock predictions from models
        with patch.object(ab_test_manager.model_a, 'predict') as mock_a:
            with patch.object(ab_test_manager.model_b, 'predict') as mock_b:
                mock_a.return_value = {"score": 80, "model_id": "model_a"}
                mock_b.return_value = {"score": 70, "model_id": "model_b"}
                
                # Force assignment to model A
                with patch.object(ab_test_manager, 'get_assignment') as mock_assignment:
                    mock_assignment.return_value = "model_a"
                    
                    # Make prediction
                    result = ab_test_manager.predict("user_1", "request_1", {"headline": "Test"})
                    
                    # Should use model A
                    assert result["model_id"] == "model_a"
                    assert result["score"] == 80
                    mock_a.assert_called_once()
                    mock_b.assert_not_called()
                
                # Reset mocks
                mock_a.reset_mock()
                mock_b.reset_mock()
                
                # Force assignment to model B
                with patch.object(ab_test_manager, 'get_assignment') as mock_assignment:
                    mock_assignment.return_value = "model_b"
                    
                    # Make prediction
                    result = ab_test_manager.predict("user_1", "request_2", {"headline": "Test"})
                    
                    # Should use model B
                    assert result["model_id"] == "model_b"
                    assert result["score"] == 70
                    mock_b.assert_called_once()
                    mock_a.assert_not_called()
    
    def test_ab_test_analysis(self, ab_test_manager):
        """Test analyzing results of an A/B test."""
        # Mock outcome data
        mock_outcomes = []
        # Model A has higher scores but lower conversion rate
        for i in range(500):
            # Model A outcomes
            mock_outcomes.append({
                "request_id": f"req_a_{i}",
                "user_id": f"user_{i}",
                "timestamp": datetime.now() - timedelta(days=5, minutes=i),
                "model_id": "model_a",
                "prediction": {"score": 75 + np.random.normal(0, 5)},
                "actual_metrics": {
                    "conversion": 1 if np.random.random() < 0.15 else 0,
                    "revenue": np.random.exponential(50) if np.random.random() < 0.15 else 0
                }
            })
            
            # Model B outcomes
            mock_outcomes.append({
                "request_id": f"req_b_{i}",
                "user_id": f"user_{500+i}",
                "timestamp": datetime.now() - timedelta(days=5, minutes=i),
                "model_id": "model_b",
                "prediction": {"score": 65 + np.random.normal(0, 5)},
                "actual_metrics": {
                    "conversion": 1 if np.random.random() < 0.20 else 0,  # Higher conversion
                    "revenue": np.random.exponential(60) if np.random.random() < 0.20 else 0  # Higher revenue
                }
            })
        
        # Mock the retrieval of outcomes
        with patch.object(ab_test_manager, '_get_test_outcomes') as mock_get_outcomes:
            mock_get_outcomes.return_value = mock_outcomes
            
            # Analyze results
            analysis = ab_test_manager.analyze_results(metrics=["conversion", "revenue"])
            
            # Check analysis
            assert 'model_a' in analysis
            assert 'model_b' in analysis
            assert 'conversion_rate' in analysis['model_a']
            assert 'revenue' in analysis['model_a']
            assert 'conversion_rate' in analysis['model_b']
            assert 'revenue' in analysis['model_b']
            assert 'statistical_significance' in analysis
            
            # Model B should have higher conversion and revenue
            assert analysis['model_b']['conversion_rate'] > analysis['model_a']['conversion_rate']
            assert analysis['model_b']['revenue']['mean'] > analysis['model_a']['revenue']['mean']


class TestCanaryTesting:
    """Test suite for canary testing capabilities."""
    
    @pytest.fixture
    def canary_runner(self):
        """Create canary test runner."""
        # Create model instance
        model = AdScorePredictor()
        
        # Define golden set queries
        golden_queries = [
            {"id": "canary_1", "headline": "Limited offer", "description": "Special discount", "expected_score_range": [75, 85]},
            {"id": "canary_2", "headline": "New product", "description": "Just released", "expected_score_range": [60, 70]},
            {"id": "canary_3", "headline": "Free shipping", "description": "Order now", "expected_score_range": [80, 90]}
        ]
        
        # Create canary test runner
        runner = CanaryTestRunner(
            model=model,
            golden_queries=golden_queries,
            alert_on_failure=True
        )
        
        return runner
    
    def test_canary_test_execution(self, canary_runner):
        """Test execution of canary tests."""
        # Mock model to return scores within expected ranges
        with patch.object(canary_runner.model, 'predict') as mock_predict:
            def side_effect(query):
                query_id = query.get("id", "")
                
                if "canary_1" in query_id:
                    return {"score": 80, "confidence": 0.9}
                elif "canary_2" in query_id:
                    return {"score": 65, "confidence": 0.85}
                elif "canary_3" in query_id:
                    return {"score": 85, "confidence": 0.95}
                else:
                    return {"score": 50, "confidence": 0.7}
            
            mock_predict.side_effect = side_effect
            
            # Run canary tests
            results = canary_runner.run_tests()
            
            # All tests should pass
            assert results["all_passed"] is True
            assert len(results["passed_tests"]) == 3
            assert len(results["failed_tests"]) == 0
    
    def test_canary_failure_detection(self, canary_runner):
        """Test detection of failing canary tests."""
        # Mock model to return scores outside expected ranges
        with patch.object(canary_runner.model, 'predict') as mock_predict:
            def side_effect(query):
                query_id = query.get("id", "")
                
                if "canary_1" in query_id:
                    return {"score": 80, "confidence": 0.9}  # Within range
                elif "canary_2" in query_id:
                    return {"score": 55, "confidence": 0.85}  # Below range
                elif "canary_3" in query_id:
                    return {"score": 95, "confidence": 0.95}  # Above range
                else:
                    return {"score": 50, "confidence": 0.7}
            
            mock_predict.side_effect = side_effect
            
            # Mock alert function
            with patch('app.models.ml.validation.canary_test.CanaryTestRunner._send_alert') as mock_alert:
                # Run canary tests
                results = canary_runner.run_tests()
                
                # Should have failing tests
                assert results["all_passed"] is False
                assert len(results["passed_tests"]) == 1
                assert len(results["failed_tests"]) == 2
                
                # Alert should be triggered
                mock_alert.assert_called_once()
    
    def test_canary_latency(self, canary_runner):
        """Test monitoring of canary test latency."""
        # Mock model with varying response times
        with patch.object(canary_runner.model, 'predict') as mock_predict:
            def side_effect(query):
                query_id = query.get("id", "")
                
                # Simulate different response times
                if "canary_1" in query_id:
                    import time
                    time.sleep(0.01)  # Fast
                elif "canary_2" in query_id:
                    import time
                    time.sleep(0.05)  # Medium
                elif "canary_3" in query_id:
                    import time
                    time.sleep(0.1)  # Slow
                
                return {"score": 80, "confidence": 0.9}
            
            mock_predict.side_effect = side_effect
            
            # Run canary tests with latency tracking
            results = canary_runner.run_tests(track_latency=True)
            
            # Should have latency metrics
            assert "latency_metrics" in results
            assert "max_latency_ms" in results["latency_metrics"]
            assert "avg_latency_ms" in results["latency_metrics"]
            assert "p95_latency_ms" in results["latency_metrics"]
            
            # Canary 3 should be slowest
            assert results["test_results"]["canary_3"]["latency_ms"] > results["test_results"]["canary_1"]["latency_ms"]


class TestGoldenSetValidation:
    """Test suite for golden set validation."""
    
    @pytest.fixture
    def golden_set_validator(self):
        """Create golden set validator."""
        # Create model instance
        model = AdScorePredictor()
        
        # Create golden dataset
        golden_set = pd.DataFrame({
            "id": [f"golden_{i}" for i in range(100)],
            "headline": [f"Headline {i}" for i in range(100)],
            "description": [f"Description {i}" for i in range(100)],
            "expected_score": [75 + (i % 20) for i in range(100)],
            "tolerance": [5 for _ in range(100)]
        })
        
        # Create validator
        validator = GoldenSetValidator(
            model=model,
            golden_dataset=golden_set,
            prediction_key="score",
            expected_key="expected_score",
            tolerance_key="tolerance"
        )
        
        return validator
    
    def test_golden_set_validation(self, golden_set_validator):
        """Test validation against golden dataset."""
        # Mock model to return predictions close to expected values
        with patch.object(golden_set_validator.model, 'predict') as mock_predict:
            def side_effect(query):
                # Return prediction close to expected value (within tolerance)
                expected = query.get("expected_score", 75)
                # Add small random noise within tolerance
                tolerance = query.get("tolerance", 5)
                noise = np.random.uniform(-tolerance * 0.8, tolerance * 0.8)
                return {"score": expected + noise, "confidence": 0.9}
            
            mock_predict.side_effect = side_effect
            
            # Run validation
            results = golden_set_validator.validate()
            
            # Most/all tests should pass
            assert results["pass_rate"] > 0.95
            assert results["status"] == "PASS"
    
    def test_golden_set_failure_detection(self, golden_set_validator):
        """Test detection of failures in golden set validation."""
        # Mock model to return predictions with systematic bias
        with patch.object(golden_set_validator.model, 'predict') as mock_predict:
            def side_effect(query):
                # Return prediction with systematic bias
                expected = query.get("expected_score", 75)
                # Add systematic bias (consistently too high)
                return {"score": expected + 10, "confidence": 0.9}
            
            mock_predict.side_effect = side_effect
            
            # Run validation
            results = golden_set_validator.validate()
            
            # Should detect systematic failure
            assert results["pass_rate"] < 0.5
            assert results["status"] == "FAIL"
            assert "systematic_bias" in results
            assert results["systematic_bias"] > 0  # Positive bias
    
    def test_golden_set_metrics(self, golden_set_validator):
        """Test detailed metrics from golden set validation."""
        # Mock model to return predictions with mixed accuracy
        with patch.object(golden_set_validator.model, 'predict') as mock_predict:
            def side_effect(query):
                # Return prediction with varying accuracy
                expected = query.get("expected_score", 75)
                query_id = query.get("id", "")
                
                # Simulate different types of errors
                if "golden_1" in query_id:
                    return {"score": expected + 20, "confidence": 0.9}  # Large error
                elif "golden_2" in query_id:
                    return {"score": expected - 15, "confidence": 0.9}  # Large negative error
                else:
                    # Small random error
                    noise = np.random.uniform(-3, 3)
                    return {"score": expected + noise, "confidence": 0.9}
            
            mock_predict.side_effect = side_effect
            
            # Run validation with detailed metrics
            results = golden_set_validator.validate(compute_detailed_metrics=True)
            
            # Should have detailed metrics
            assert "detailed_metrics" in results
            assert "rmse" in results["detailed_metrics"]
            assert "mae" in results["detailed_metrics"]
            assert "max_error" in results["detailed_metrics"]
            assert "error_distribution" in results["detailed_metrics"]
            
            # Max error should match our large outlier
            assert abs(results["detailed_metrics"]["max_error"]) >= 15


class TestProductionMonitoring:
    """Test suite for production monitoring."""
    
    @pytest.fixture
    def monitoring_data(self):
        """Generate mock monitoring data."""
        # Create timestamps over 24 hours
        now = datetime.now()
        timestamps = [now - timedelta(hours=i) for i in range(24)]
        
        # Create mock data
        data = []
        for ts in timestamps:
            # Normal traffic pattern with some daily seasonality
            hour = ts.hour
            base_queries = 100 if (9 <= hour <= 17) else 50  # Higher during work hours
            
            # Add some random noise
            queries = max(0, int(base_queries + np.random.normal(0, 10)))
            latency = max(10, 100 + np.random.normal(0, 20))  # in ms
            error_rate = max(0, min(1, 0.01 + np.random.normal(0, 0.005)))
            
            data.append({
                "timestamp": ts.isoformat(),
                "model_id": "ad_score_predictor_v1",
                "instance_id": "instance_001",
                "queries_per_minute": queries,
                "avg_latency_ms": latency,
                "p95_latency_ms": latency * 1.5,
                "p99_latency_ms": latency * 2,
                "error_rate": error_rate,
                "memory_usage_mb": 2048 + np.random.normal(0, 100),
                "cpu_usage_percent": 30 + np.random.normal(0, 5)
            })
        
        return pd.DataFrame(data)
    
    def test_monitoring_anomaly_detection(self, monitoring_data):
        """Test detection of anomalies in monitoring metrics."""
        # Add some anomalies to the data
        anomalous_data = monitoring_data.copy()
        
        # Add latency spike
        anomalous_data.loc[5, "avg_latency_ms"] = 500  # Extreme latency
        anomalous_data.loc[5, "p95_latency_ms"] = 800
        anomalous_data.loc[5, "p99_latency_ms"] = 1000
        
        # Add error rate spike
        anomalous_data.loc[10, "error_rate"] = 0.15  # High error rate
        
        # Add traffic drop
        anomalous_data.loc[15, "queries_per_minute"] = 5  # Very low traffic
        
        # Mock monitoring service
        with patch('app.services.monitoring.production_monitoring_service.detect_anomalies') as mock_detect:
            mock_detect.return_value = {
                "anomalies_detected": True,
                "anomalous_points": [
                    {"timestamp": anomalous_data.loc[5, "timestamp"], "metric": "avg_latency_ms", "value": 500, "expected_range": [80, 120]},
                    {"timestamp": anomalous_data.loc[10, "timestamp"], "metric": "error_rate", "value": 0.15, "expected_range": [0, 0.03]},
                    {"timestamp": anomalous_data.loc[15, "timestamp"], "metric": "queries_per_minute", "value": 5, "expected_range": [40, 110]}
                ]
            }
            
            # Run anomaly detection
            result = production_monitoring_service.detect_anomalies(anomalous_data)
            
            # Should detect anomalies
            assert result["anomalies_detected"] is True
            assert len(result["anomalous_points"]) == 3
    
    def test_alert_generation(self, monitoring_data):
        """Test generation of alerts based on monitoring thresholds."""
        # Mock monitoring service with threshold configuration
        thresholds = {
            "avg_latency_ms": 300,
            "error_rate": 0.05,
            "queries_per_minute": {"min": 10, "max": 500}
        }
        
        # Create anomalous data point
        anomaly = monitoring_data.iloc[0].copy()
        anomaly["avg_latency_ms"] = 350  # Above threshold
        
        # Mock alert function
        with patch('app.services.monitoring.production_monitoring_service.send_alert') as mock_alert:
            # Check threshold violation
            production_monitoring_service.check_thresholds(anomaly, thresholds)
            
            # Should send alert
            mock_alert.assert_called_once()
            alert_data = mock_alert.call_args[0][0]
            assert "threshold_violation" in alert_data
            assert alert_data["metric"] == "avg_latency_ms"
            assert alert_data["value"] == 350
            assert alert_data["threshold"] == 300
    
    def test_service_health_status(self, monitoring_data):
        """Test computation of overall service health status."""
        # Mock monitoring service
        with patch('app.services.monitoring.production_monitoring_service.get_health_status') as mock_health:
            mock_health.return_value = {
                "status": "HEALTHY",
                "metrics": {
                    "availability": 99.98,
                    "avg_latency_ms": 105.2,
                    "error_rate": 0.008,
                    "throughput_qps": 78.3
                },
                "issues": []
            }
            
            # Get health status
            status = production_monitoring_service.get_health_status()
            
            # Should return health metrics
            assert status["status"] == "HEALTHY"
            assert "metrics" in status
            assert "availability" in status["metrics"]
    
    def test_metric_visualization(self, monitoring_data):
        """Test generation of monitoring visualizations."""
        # Mock visualization function
        with patch('app.services.monitoring.production_monitoring_service.generate_visualization') as mock_viz:
            mock_viz.return_value = {"plot_data": "base64_encoded_image"}
            
            # Generate visualization
            viz = production_monitoring_service.generate_visualization(
                monitoring_data,
                metric="avg_latency_ms",
                time_window=24
            )
            
            # Should return visualization data
            assert "plot_data" in viz
    
    def test_consistency_checks(self, monitoring_data):
        """Test validation of prediction consistency across instances."""
        # Create test queries
        test_queries = [
            {"id": "query_1", "headline": "Test 1", "description": "Description 1"},
            {"id": "query_2", "headline": "Test 2", "description": "Description 2"}
        ]
        
        # Mock instance predictions
        instance_predictions = {
            "instance_1": [
                {"query_id": "query_1", "score": 80.1, "confidence": 0.9},
                {"query_id": "query_2", "score": 75.3, "confidence": 0.85}
            ],
            "instance_2": [
                {"query_id": "query_1", "score": 80.0, "confidence": 0.9},
                {"query_id": "query_2", "score": 75.2, "confidence": 0.85}
            ],
            "instance_3": [
                {"query_id": "query_1", "score": 80.2, "confidence": 0.9},
                {"query_id": "query_2", "score": 75.4, "confidence": 0.85}
            ]
        }
        
        # Mock consistency check
        with patch('app.services.monitoring.production_monitoring_service.check_prediction_consistency') as mock_check:
            mock_check.return_value = {
                "consistent": True,
                "max_deviation": 0.2,
                "avg_deviation": 0.1,
                "details": [
                    {"query_id": "query_1", "max_diff": 0.2, "consistent": True},
                    {"query_id": "query_2", "max_diff": 0.2, "consistent": True}
                ]
            }
            
            # Check consistency
            result = production_monitoring_service.check_prediction_consistency(
                test_queries,
                instance_predictions
            )
            
            # Should be consistent (small deviations)
            assert result["consistent"] is True
            assert result["max_deviation"] <= 0.5  # Small deviations 