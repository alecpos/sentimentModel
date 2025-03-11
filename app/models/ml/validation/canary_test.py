"""
Canary testing framework for ML models.

This module provides tools for safely deploying ML models using canary testing,
where a small portion of traffic is initially directed to the new model to verify
its performance before full deployment.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import logging
import time
import json
import uuid
from enum import Enum

logger = logging.getLogger(__name__)

class CanaryStatus(Enum):
    """Status of a canary test."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    ABORTED = "aborted"


class CanaryTestRunner:
    """
    Runner for canary tests to validate new model deployments.
    
    This class implements methods to run canary tests, which validate
    new model deployments by sending a small percentage of traffic to
    the new model and monitoring its performance compared to the current model.
    """
    
    def __init__(
        self,
        current_model: Optional[Any] = None,
        candidate_model: Optional[Any] = None,
        model: Optional[Any] = None,
        test_name: Optional[str] = None,
        golden_queries: Optional[List[Dict[str, Any]]] = None,
        traffic_percentage: float = 5.0,
        monitoring_metrics: Optional[List[str]] = None,
        success_criteria: Optional[Dict[str, Dict[str, float]]] = None,
        failback_threshold: Optional[Dict[str, float]] = None,
        test_duration: int = 60,  # Duration in minutes
        rollback_on_failure: bool = True,
        alert_on_failure: bool = True
    ):
        """
        Initialize the canary test runner.
        
        Args:
            current_model: Current production model (can be None if model is provided)
            candidate_model: Candidate model to test (can be None if model is provided)
            model: Single model to test with golden queries (can be None if current_model and candidate_model are provided)
            test_name: Name of the canary test
            golden_queries: Optional list of golden queries for validation
            traffic_percentage: Percentage of traffic to route to canary
            monitoring_metrics: Metrics to monitor during test
            success_criteria: Criteria for test success
            failback_threshold: Thresholds for immediate failback
            test_duration: Test duration in minutes
            rollback_on_failure: Whether to roll back if test fails
            alert_on_failure: Whether to alert on test failure
        """
        # Handle models - either provide current/candidate pair or a single model
        if model is not None:
            self.current_model = model  # Use single model for both if doing golden query tests
            self.candidate_model = model
            self.model = model
        else:
            self.current_model = current_model
            self.candidate_model = candidate_model
            if current_model is not None:
                self.model = current_model  # For backward compatibility

        self.test_name = test_name or f"canary_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.golden_queries = golden_queries or []
        self.traffic_percentage = traffic_percentage
        
        # Default monitoring metrics if none provided
        self.monitoring_metrics = monitoring_metrics or [
            "latency_ms",
            "error_rate",
            "prediction_drift",
            "memory_usage_mb"
        ]
        
        # Default success criteria if none provided
        self.success_criteria = success_criteria or {
            "error_rate": {"max": 0.01, "max_increase": 100.0},  # Max 1% errors, max 100% increase
            "latency_ms": {"max": 100, "max_increase": 20.0},     # Max 100ms, max 20% increase
            "prediction_drift": {"max": 0.1},                    # Max 10% prediction drift
            "memory_usage_mb": {"max": 1000}                      # Max 1GB memory usage
        }
        
        # Default failback thresholds if none provided
        self.failback_threshold = failback_threshold or {
            "error_rate": 0.05,        # 5% errors
            "latency_ms": 500,         # 500ms latency
            "prediction_drift": 0.2    # 20% prediction drift
        }
        
        self.test_duration = test_duration
        self.rollback_on_failure = rollback_on_failure
        self.alert_on_failure = alert_on_failure
        
        # Test state
        self.status = CanaryStatus.PENDING
        self.start_time = None
        self.end_time = None
        self.expected_end_time = None
        
        # Metrics storage
        self.current_metrics = {metric: [] for metric in self.monitoring_metrics}
        self.candidate_metrics = {metric: [] for metric in self.monitoring_metrics}
        self.timestamps = []
        
        # Results
        self.results = {}
        self.failure_reasons = []
    
    def start_test(self) -> Dict[str, Any]:
        """
        Start the canary test.
        
        Returns:
            Dictionary with test details
        """
        if self.status != CanaryStatus.PENDING:
            return {
                "success": False,
                "message": f"Cannot start test with status: {self.status.value}",
                "test_id": self.test_id
            }
        
        self.start_time = datetime.now()
        self.expected_end_time = self.start_time + timedelta(minutes=self.test_duration)
        self.status = CanaryStatus.RUNNING
        
        # Record initial state (no metrics yet)
        self._record_initial_state()
        
        return {
            "success": True,
            "message": "Canary test started successfully",
            "test_id": self.test_id,
            "test_name": self.test_name,
            "start_time": self.start_time,
            "expected_end_time": self.expected_end_time,
            "traffic_percentage": self.traffic_percentage,
            "monitoring_metrics": self.monitoring_metrics
        }
    
    def _record_initial_state(self) -> None:
        """Record the initial state of the models."""
        # In a real implementation, this might include:
        # - Taking a snapshot of current model configuration
        # - Recording baseline metrics
        # - Setting up monitoring infrastructure
        self.timestamps.append(datetime.now())
        
        # Mock initial metrics
        for metric in self.monitoring_metrics:
            self.current_metrics[metric].append(0.0)
            self.candidate_metrics[metric].append(0.0)
    
    def record_metrics(
        self,
        current_model_metrics: Dict[str, float],
        candidate_model_metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Record metrics for both models.
        
        Args:
            current_model_metrics: Metrics for the current model
            candidate_model_metrics: Metrics for the candidate model
            timestamp: Timestamp for the metrics (defaults to now)
            
        Returns:
            Dictionary indicating success or failure
        """
        if self.status != CanaryStatus.RUNNING:
            return {
                "success": False,
                "message": f"Cannot record metrics for test with status: {self.status.value}"
            }
        
        timestamp = timestamp or datetime.now()
        self.timestamps.append(timestamp)
        
        # Record metrics for each model
        for metric in self.monitoring_metrics:
            current_value = current_model_metrics.get(metric, 0.0)
            candidate_value = candidate_model_metrics.get(metric, 0.0)
            
            self.current_metrics[metric].append(current_value)
            self.candidate_metrics[metric].append(candidate_value)
        
        # Check for immediate failures
        failure_detected = self._check_for_immediate_failure(candidate_model_metrics)
        
        if failure_detected and self.rollback_on_failure:
            self._abort_test("Failback thresholds exceeded")
        
        return {
            "success": True,
            "message": "Metrics recorded successfully",
            "immediate_failure": failure_detected
        }
    
    def _check_for_immediate_failure(self, candidate_metrics: Dict[str, float]) -> bool:
        """
        Check if any metrics exceed the immediate failure thresholds.
        
        Args:
            candidate_metrics: Current metrics for the candidate model
            
        Returns:
            True if a failure threshold is exceeded, False otherwise
        """
        for metric, threshold in self.failback_threshold.items():
            if metric in candidate_metrics and candidate_metrics[metric] > threshold:
                reason = f"Metric {metric} exceeded failback threshold: {candidate_metrics[metric]} > {threshold}"
                self.failure_reasons.append(reason)
                logger.warning(f"Canary test failure: {reason}")
                return True
                
        return False
    
    def _abort_test(self, reason: str) -> None:
        """
        Abort the canary test.
        
        Args:
            reason: Reason for aborting the test
        """
        self.status = CanaryStatus.ABORTED
        self.end_time = datetime.now()
        self.failure_reasons.append(reason)
        
        if self.alert_on_failure:
            self._send_alert(reason)
    
    def _send_alert(self, message: str) -> None:
        """
        Send an alert about the canary test.
        
        Args:
            message: Alert message
        """
        logger.warning(f"Canary test alert: {message}")
        # In a real implementation, this would send to an alerting system
        pass
    
    def evaluate_results(self) -> Dict[str, Any]:
        """
        Evaluate the results of the canary test.
        
        Returns:
            Dictionary with test results
        """
        if self.status == CanaryStatus.PENDING:
            return {
                "success": False,
                "message": "Cannot evaluate results for a test that hasn't started",
                "test_id": self.test_id
            }
        
        if self.status == CanaryStatus.RUNNING:
            # End the test if it's still running
            self.end_time = datetime.now()
            self.status = CanaryStatus.INCONCLUSIVE
        
        # Calculate summary metrics
        summary = self._calculate_summary_metrics()
        
        # Determine test result
        if self.status == CanaryStatus.ABORTED:
            test_passed = False
        else:
            test_passed = self._evaluate_success_criteria(summary)
            self.status = CanaryStatus.PASSED if test_passed else CanaryStatus.FAILED
        
        # Store results
        self.results = {
            "test_id": self.test_id,
            "test_name": self.test_name,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_minutes": (self.end_time - self.start_time).total_seconds() / 60,
            "traffic_percentage": self.traffic_percentage,
            "metrics_summary": summary,
            "success_criteria": self.success_criteria,
            "passed": test_passed,
            "failure_reasons": self.failure_reasons if not test_passed else []
        }
        
        return self.results
    
    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """
        Calculate summary metrics for the test.
        
        Returns:
            Dictionary with summary metrics
        """
        summary = {}
        
        for metric in self.monitoring_metrics:
            if not self.current_metrics[metric] or not self.candidate_metrics[metric]:
                continue
                
            current_values = self.current_metrics[metric]
            candidate_values = self.candidate_metrics[metric]
            
            # Calculate basic statistics
            summary[metric] = {
                "current": {
                    "mean": np.mean(current_values),
                    "median": np.median(current_values),
                    "min": np.min(current_values),
                    "max": np.max(current_values),
                    "std": np.std(current_values)
                },
                "candidate": {
                    "mean": np.mean(candidate_values),
                    "median": np.median(candidate_values),
                    "min": np.min(candidate_values),
                    "max": np.max(candidate_values),
                    "std": np.std(candidate_values)
                }
            }
            
            # Calculate differences
            curr_mean = summary[metric]["current"]["mean"]
            cand_mean = summary[metric]["candidate"]["mean"]
            
            if curr_mean > 0:
                pct_change = (cand_mean - curr_mean) / curr_mean * 100.0
            else:
                pct_change = 0.0 if cand_mean == 0 else float('inf')
                
            summary[metric]["comparison"] = {
                "absolute_diff": cand_mean - curr_mean,
                "percent_change": pct_change
            }
        
        return summary
    
    def _evaluate_success_criteria(self, summary: Dict[str, Any]) -> bool:
        """
        Evaluate if the test meets the success criteria.
        
        Args:
            summary: Dictionary with summary metrics
            
        Returns:
            True if the test passes all criteria, False otherwise
        """
        all_criteria_passed = True
        
        for metric, criteria in self.success_criteria.items():
            if metric not in summary:
                # Skip metrics that aren't in the summary
                continue
                
            metric_summary = summary[metric]
            candidate_mean = metric_summary["candidate"]["mean"]
            
            # Check maximum absolute value
            if "max" in criteria and candidate_mean > criteria["max"]:
                reason = f"Metric {metric} exceeded maximum value: {candidate_mean} > {criteria['max']}"
                self.failure_reasons.append(reason)
                all_criteria_passed = False
            
            # Check maximum increase relative to current model
            if "max_increase" in criteria:
                percent_change = metric_summary["comparison"]["percent_change"]
                if percent_change > criteria["max_increase"]:
                    reason = f"Metric {metric} increased by {percent_change:.2f}%, exceeding maximum allowed increase of {criteria['max_increase']}%"
                    self.failure_reasons.append(reason)
                    all_criteria_passed = False
        
        return all_criteria_passed
    
    def get_test_results(self) -> Dict[str, Any]:
        """
        Get the results of the canary test.
        
        Returns:
            Dictionary with test results
        """
        if not self.results:
            # If results haven't been computed yet, evaluate them now
            return self.evaluate_results()
            
        return self.results
    
    def stop_test(self) -> Dict[str, Any]:
        """
        Stop the canary test.
        
        Returns:
            Dictionary with test details
        """
        if self.status != CanaryStatus.RUNNING:
            return {
                "success": False,
                "message": f"Cannot stop test with status: {self.status.value}",
                "test_id": self.test_id
            }
            
        self.end_time = datetime.now()
        
        # Evaluate results to determine final status
        results = self.evaluate_results()
        
        return {
            "success": True,
            "message": "Canary test stopped successfully",
            "test_id": self.test_id,
            "test_name": self.test_name,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_minutes": (self.end_time - self.start_time).total_seconds() / 60,
            "results": results
        }
    
    def promote_candidate(self) -> Dict[str, Any]:
        """
        Promote the candidate model to production.
        
        Returns:
            Dictionary indicating success or failure
        """
        if self.status != CanaryStatus.PASSED:
            return {
                "success": False,
                "message": f"Cannot promote candidate with test status: {self.status.value}",
                "test_id": self.test_id
            }
            
        # In a real implementation, this would handle the actual promotion
        # For this stub, we just return success
        return {
            "success": True,
            "message": "Candidate model promoted to production successfully",
            "test_id": self.test_id,
            "test_name": self.test_name,
            "promotion_time": datetime.now()
        }
    
    def rollback(self) -> Dict[str, Any]:
        """
        Rollback to the current model.
        
        Returns:
            Dictionary indicating success or failure
        """
        if self.status == CanaryStatus.PENDING:
            return {
                "success": False,
                "message": "Cannot rollback a test that hasn't started",
                "test_id": self.test_id
            }
            
        # In a real implementation, this would handle the actual rollback
        # For this stub, we just return success
        return {
            "success": True,
            "message": "Rolled back to current model successfully",
            "test_id": self.test_id,
            "test_name": self.test_name,
            "rollback_time": datetime.now(),
            "rollback_reason": ", ".join(self.failure_reasons) if self.failure_reasons else "Manual rollback"
        }

    def run_tests(self, track_latency: bool = False) -> Dict[str, Any]:
        """
        Run canary tests on the golden queries and return the results.
        
        Args:
            track_latency: Whether to track and report latency metrics
            
        Returns:
            Dictionary containing test results with the following structure:
            {
                "test_id": str,
                "status": str,
                "passed": bool,
                "all_passed": bool,  # For backward compatibility
                "total_tests": int,
                "passed_tests": List[str],  # List of passed test IDs
                "failed_tests": List[str],  # List of failed test IDs
                "results": List[Dict],
                "test_results": Dict, # For test compatibility
                "metrics": Dict,
                "latency_metrics": Dict (if track_latency is True)
            }
        """
        if not self.golden_queries:
            logger.warning("No golden queries provided for canary testing")
            return {
                "test_id": self.test_name,
                "status": CanaryStatus.INCONCLUSIVE.value,
                "passed": False,
                "all_passed": False,
                "total_tests": 0,
                "passed_tests": [],  # Empty list for test compatibility
                "failed_tests": [],  # Empty list for test compatibility
                "results": [],
                "test_results": {},  # Empty dict for test compatibility
                "metrics": {}
            }
            
        logger.info(f"Running {len(self.golden_queries)} canary tests")
        
        results = []
        passed_tests = []  # List instead of int for test compatibility
        failed_tests = []  # List instead of int for test compatibility
        latency_metrics = {} if track_latency else None
        test_results = {}
        
        # Initialize latency metrics if tracking
        if track_latency:
            latency_metrics = {
                "total_latency_ms": 0,
                "min_latency_ms": float('inf'),
                "max_latency_ms": 0,
                "latencies_ms": []
            }
        
        # Run each test
        for query in self.golden_queries:
            # Get query details
            query_id = query.get("id", str(uuid.uuid4()))
            
            # Get expected score range
            expected_range = query.get("expected_score_range", [0, 100])
            min_expected = expected_range[0]
            max_expected = expected_range[1]
            
            # Track latency if needed
            start_time = time.time() if track_latency else None
            
            try:
                # Get prediction from model
                prediction = self.model.predict(query)
                
                # Calculate latency if tracking
                if track_latency:
                    latency_ms = (time.time() - start_time) * 1000
                    latency_metrics["total_latency_ms"] += latency_ms
                    latency_metrics["min_latency_ms"] = min(latency_metrics["min_latency_ms"], latency_ms)
                    latency_metrics["max_latency_ms"] = max(latency_metrics["max_latency_ms"], latency_ms)
                    latency_metrics["latencies_ms"].append(latency_ms)
                
                # Extract score from prediction
                score = prediction.get("score", 0)
                
                # Check if prediction is within expected range
                passed = min_expected <= score <= max_expected
                
                # Record result
                result = {
                    "query_id": query_id,
                    "passed": passed,
                    "expected_min": min_expected,
                    "expected_max": max_expected,
                    "actual": score,
                    "latency_ms": latency_ms if track_latency else None
                }
                
                results.append(result)
                
                # Also store in test_results for test compatibility
                test_results[query_id] = result
                
                if passed:
                    passed_tests.append(query_id)  # Append query_id instead of counting
                else:
                    failed_tests.append(query_id)  # Append query_id instead of counting
                    logger.warning(f"Canary test failed for query {query_id}: expected range [{min_expected}, {max_expected}], got {score}")
            except Exception as e:
                logger.error(f"Error running canary test for query {query_id}: {str(e)}")
                result = {
                    "query_id": query_id,
                    "passed": False,
                    "error": str(e),
                    "expected_min": min_expected,
                    "expected_max": max_expected,
                    "actual": None,
                    "latency_ms": None
                }
                results.append(result)
                test_results[query_id] = result
                failed_tests.append(query_id)
        
        # Calculate overall status
        all_passed = len(failed_tests) == 0 and len(passed_tests) > 0
        status = CanaryStatus.PASSED if all_passed else CanaryStatus.FAILED
        
        # Calculate latency stats if tracking
        if track_latency and latency_metrics["latencies_ms"]:
            latency_metrics["avg_latency_ms"] = latency_metrics["total_latency_ms"] / len(latency_metrics["latencies_ms"])
            latency_metrics["latencies_ms"] = sorted(latency_metrics["latencies_ms"])
            
            # Calculate percentiles
            n = len(latency_metrics["latencies_ms"])
            latency_metrics["p50_latency_ms"] = latency_metrics["latencies_ms"][int(n * 0.5)]
            latency_metrics["p90_latency_ms"] = latency_metrics["latencies_ms"][int(n * 0.9)]
            latency_metrics["p95_latency_ms"] = latency_metrics["latencies_ms"][int(n * 0.95)]
            latency_metrics["p99_latency_ms"] = latency_metrics["latencies_ms"][int(n * 0.99)] if n >= 100 else latency_metrics["max_latency_ms"]
        
        # Send alert if configured and tests failed
        if self.alert_on_failure and not all_passed:
            self._send_alert(failed_tests)
        
        # Prepare result dictionary
        result_dict = {
            "test_id": self.test_name,
            "status": status.value,
            "passed": all_passed,
            "all_passed": all_passed,  # Ensure all_passed is set correctly for test compatibility
            "total_tests": len(self.golden_queries),
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "results": results,
            "test_results": test_results,  # For test compatibility
            "metrics": {
                "pass_rate": len(passed_tests) / len(self.golden_queries) if self.golden_queries else 0,
                "total_tests": len(self.golden_queries),
                "passed_tests": len(passed_tests),
                "failed_tests": len(failed_tests)
            }
        }
        
        # Add latency metrics if tracking
        if track_latency:
            result_dict["latency_metrics"] = latency_metrics
        
        return result_dict


class CanaryTestRegistry:
    """
    Registry for tracking and managing canary tests.
    """
    
    def __init__(self):
        """Initialize the canary test registry."""
        self.tests = {}
        
    def register_test(self, test: CanaryTestRunner) -> Dict[str, Any]:
        """
        Register a canary test in the registry.
        
        Args:
            test: CanaryTestRunner instance to register
            
        Returns:
            Dictionary with registration details
        """
        if test.test_id in self.tests:
            return {
                "success": False,
                "message": f"Test with ID {test.test_id} already registered",
                "test_id": test.test_id
            }
            
        self.tests[test.test_id] = test
        
        return {
            "success": True,
            "message": "Test registered successfully",
            "test_id": test.test_id,
            "test_name": test.test_name
        }
    
    def get_test(self, test_id: str) -> Optional[CanaryTestRunner]:
        """
        Get a canary test by ID.
        
        Args:
            test_id: ID of the test to retrieve
            
        Returns:
            CanaryTestRunner instance if found, None otherwise
        """
        return self.tests.get(test_id)
    
    def list_tests(self, status: Optional[CanaryStatus] = None) -> List[Dict[str, Any]]:
        """
        List all registered tests.
        
        Args:
            status: Optional status filter
            
        Returns:
            List of test summaries
        """
        result = []
        
        for test_id, test in self.tests.items():
            if status is None or test.status == status:
                result.append({
                    "test_id": test_id,
                    "test_name": test.test_name,
                    "status": test.status.value,
                    "start_time": test.start_time,
                    "end_time": test.end_time
                })
                
        return result 