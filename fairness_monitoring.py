#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fairness Monitoring System

This module provides tools for monitoring fairness metrics in production environments,
detecting drift in fairness, and generating alerts when fairness degrades.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FairnessMonitor:
    """
    Monitor fairness metrics over time and detect fairness drift.
    
    This class provides tools for tracking fairness metrics in production,
    visualizing trends, and generating alerts when fairness degrades.
    """
    
    def __init__(
        self,
        protected_attributes: List[str],
        fairness_metrics: List[str] = None,
        alert_threshold: float = 0.05,
        monitoring_dir: str = 'fairness_monitoring',
        rolling_window: int = 3
    ):
        """
        Initialize the FairnessMonitor.
        
        Args:
            protected_attributes: List of protected attribute names to monitor
            fairness_metrics: List of fairness metrics to track
            alert_threshold: Threshold for alerting on fairness degradation
            monitoring_dir: Directory to store monitoring data and visualizations
            rolling_window: Number of batches to include in rolling average calculation
        """
        self.protected_attributes = protected_attributes
        self.fairness_metrics = fairness_metrics or ["demographic_parity", "equal_opportunity"]
        self.alert_threshold = alert_threshold
        self.monitoring_dir = monitoring_dir
        self.rolling_window = rolling_window
        
        self.baseline_metrics = None
        self.history = []
        self.history_df = None
        
        # Create monitoring directory structure
        os.makedirs(monitoring_dir, exist_ok=True)
        os.makedirs(os.path.join(monitoring_dir, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(monitoring_dir, 'alerts'), exist_ok=True)
        
        logger.info(f"Fairness monitor initialized for attributes: {protected_attributes}")
        logger.info(f"Monitoring metrics: {self.fairness_metrics}")
        logger.info(f"Alert threshold: {alert_threshold}")
    
    def set_baseline(self, baseline_metrics, save_path=None):
        """
        Set baseline fairness metrics for comparison.
        
        Args:
            baseline_metrics: Dictionary containing fairness evaluation results
            save_path: Path to save baseline metrics (optional)
        """
        self.baseline_metrics = baseline_metrics
        
        # Save baseline metrics to file
        if save_path is None:
            save_path = os.path.join(self.monitoring_dir, 'baseline_metrics.json')
        
        with open(save_path, 'w') as f:
            json.dump(baseline_metrics, f, indent=2)
        
        logger.info(f"Baseline metrics set and saved to {save_path}")
    
    def update(
        self,
        predictions: np.ndarray,
        labels: Optional[np.ndarray] = None,
        protected_attributes: Dict[str, np.ndarray] = None,
        batch_id: str = None,
        timestamp: str = None
    ):
        """
        Update monitoring with new batch of predictions.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels (optional)
            protected_attributes: Dictionary mapping attribute names to values
            batch_id: Identifier for this batch
            timestamp: Timestamp for this batch (optional)
        
        Returns:
            Dictionary containing current metrics and alert status
        """
        # Generate batch ID and timestamp if not provided
        if batch_id is None:
            batch_id = f"batch_{len(self.history) + 1:03d}"
        
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        # Create batch directory
        batch_dir = os.path.join(self.monitoring_dir, f"batch_{batch_id}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Calculate fairness metrics
        if labels is not None:
            # If labels are available, calculate standard fairness metrics
            from app.models.ml.fairness.evaluator import FairnessEvaluator
            
            evaluator = FairnessEvaluator(
                protected_attributes=self.protected_attributes,
                fairness_threshold=self.alert_threshold,
                metrics=self.fairness_metrics
            )
            
            current_metrics = evaluator.evaluate(predictions, labels, protected_attributes)
        else:
            # If no labels, evaluate distributions
            current_metrics = self._evaluate_distributions(predictions, protected_attributes)
        
        # Save current metrics
        with open(os.path.join(batch_dir, 'metrics.json'), 'w') as f:
            json.dump(current_metrics, f, indent=2)
        
        # Check for alerts
        alerts = self._check_alerts(current_metrics)
        
        # Save alerts if any
        if alerts:
            with open(os.path.join(batch_dir, 'alerts.json'), 'w') as f:
                json.dump(alerts, f, indent=2)
            
            with open(os.path.join(self.monitoring_dir, 'alerts', f"{batch_id}_alerts.json"), 'w') as f:
                json.dump(alerts, f, indent=2)
        
        # Add to history
        batch_summary = {
            'batch_id': batch_id,
            'timestamp': timestamp,
            'metrics': current_metrics,
            'alerts': alerts
        }
        
        self.history.append(batch_summary)
        
        # Update history dataframe
        self._update_history_df()
        
        # Create visualizations
        self._create_visualizations(batch_id)
        
        # Save monitoring history
        self._save_history()
        
        return {
            'metrics': current_metrics,
            'alerts': alerts
        }
    
    def _evaluate_distributions(self, predictions, protected_attributes):
        """
        Evaluate fairness metrics based on prediction distributions when labels are not available.
        
        Args:
            predictions: Model predictions
            protected_attributes: Dictionary mapping attribute names to values
            
        Returns:
            Dictionary containing distribution-based fairness metrics
        """
        results = {
            "fairness_metrics": {},
            "group_metrics": {}
        }
        
        # Process each protected attribute
        for attr_name, attr_values in protected_attributes.items():
            if attr_name not in self.protected_attributes:
                continue
            
            # Convert to numpy array if needed
            if hasattr(attr_values, 'values'):
                attr_values = attr_values.values
            
            # Calculate metrics for each group
            unique_values = np.unique(attr_values)
            group_metrics = {}
            
            for value in unique_values:
                mask = (attr_values == value)
                
                if not np.any(mask):
                    continue
                
                group_preds = predictions[mask]
                
                group_metrics[str(value)] = {
                    "count": int(np.sum(mask)),
                    "mean_score": float(np.mean(group_preds)),
                    "median_score": float(np.median(group_preds)),
                    "std_score": float(np.std(group_preds)),
                    "min_score": float(np.min(group_preds)),
                    "max_score": float(np.max(group_preds)),
                    "positive_rate": float(np.mean(group_preds >= 0.5))
                }
            
            results["group_metrics"][attr_name] = group_metrics
            
            # Calculate distribution differences (as a proxy for fairness metrics)
            if len(group_metrics) > 1:
                # For demographic parity: difference in mean scores
                mean_scores = [metrics["mean_score"] for metrics in group_metrics.values()]
                diff = max(mean_scores) - min(mean_scores)
                
                results["fairness_metrics"][f"{attr_name}_demographic_parity"] = {
                    "difference": float(diff),
                    "passes_threshold": diff <= self.alert_threshold
                }
                
                # For positive rate parity: difference in positive rates
                pos_rates = [metrics["positive_rate"] for metrics in group_metrics.values()]
                pos_rate_diff = max(pos_rates) - min(pos_rates)
                
                results["fairness_metrics"][f"{attr_name}_positive_rate_parity"] = {
                    "difference": float(pos_rate_diff),
                    "passes_threshold": pos_rate_diff <= self.alert_threshold
                }
        
        return results
    
    def _check_alerts(self, current_metrics):
        """
        Check for fairness alerts by comparing current metrics to baseline.
        
        Args:
            current_metrics: Dictionary containing current fairness metrics
            
        Returns:
            Dictionary containing alert information
        """
        if self.baseline_metrics is None:
            logger.warning("No baseline metrics set, cannot check for alerts")
            return {}
        
        alerts = {}
        
        # Compare fairness metrics
        baseline_fairness = self.baseline_metrics.get("fairness_metrics", {})
        current_fairness = current_metrics.get("fairness_metrics", {})
        
        for metric_key, baseline_value in baseline_fairness.items():
            if metric_key in current_fairness:
                # Check if the difference has increased beyond the threshold
                if "difference" in baseline_value and "difference" in current_fairness[metric_key]:
                    baseline_diff = baseline_value["difference"]
                    current_diff = current_fairness[metric_key]["difference"]
                    
                    change = current_diff - baseline_diff
                    
                    if change > self.alert_threshold:
                        alerts[metric_key] = {
                            "baseline_difference": baseline_diff,
                            "current_difference": current_diff,
                            "change": change,
                            "threshold": self.alert_threshold,
                            "severity": "high" if change > self.alert_threshold * 2 else "medium"
                        }
        
        return alerts
    
    def _update_history_df(self):
        """
        Update the history dataframe with the latest metrics.
        """
        # Extract metrics from history
        rows = []
        
        for batch in self.history:
            batch_id = batch['batch_id']
            timestamp = batch['timestamp']
            metrics = batch['metrics']
            
            fairness_metrics = metrics.get('fairness_metrics', {})
            
            for metric_key, metric_value in fairness_metrics.items():
                if 'difference' in metric_value:
                    row = {
                        'batch_id': batch_id,
                        'timestamp': timestamp,
                        'metric': metric_key,
                        'difference': metric_value['difference'],
                        'passes_threshold': metric_value.get('passes_threshold', False)
                    }
                    rows.append(row)
        
        self.history_df = pd.DataFrame(rows)
        
        # Convert timestamp to datetime
        if not self.history_df.empty and 'timestamp' in self.history_df.columns:
            self.history_df['timestamp'] = pd.to_datetime(self.history_df['timestamp'])
    
    def _create_visualizations(self, batch_id):
        """
        Create visualizations for the current monitoring state.
        
        Args:
            batch_id: ID of the current batch
        """
        if self.history_df is None or self.history_df.empty:
            logger.warning("No history data available for visualization")
            return
        
        vis_dir = os.path.join(self.monitoring_dir, 'visualizations')
        
        # Create trend plots for each metric
        self._create_trend_plots(vis_dir)
        
        # Create comparison visualizations
        if len(self.history) > 1:
            self._create_comparison_visualizations(vis_dir, batch_id)
        
        # Create alert visualizations if there are alerts
        latest_batch = self.history[-1]
        if latest_batch.get('alerts'):
            self._create_alert_visualizations(vis_dir, batch_id, latest_batch['alerts'])
    
    def _create_trend_plots(self, vis_dir):
        """
        Create trend plots for each fairness metric.
        
        Args:
            vis_dir: Directory to save visualizations
        """
        # Get unique metrics
        metrics = self.history_df['metric'].unique()
        
        # Set up a good color palette
        colors = plt.cm.tab10.colors
        
        for i, metric in enumerate(metrics):
            metric_data = self.history_df[self.history_df['metric'] == metric].sort_values('timestamp')
            
            if len(metric_data) <= 1:
                continue
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot metric trend
            plt.plot(
                metric_data['timestamp'], 
                metric_data['difference'], 
                marker='o', 
                color=colors[i % len(colors)],
                label=metric
            )
            
            # Add rolling average if we have enough data
            if len(metric_data) >= self.rolling_window:
                rolling_avg = metric_data['difference'].rolling(window=self.rolling_window).mean()
                plt.plot(
                    metric_data['timestamp'],
                    rolling_avg,
                    linestyle='--',
                    color=colors[(i+1) % len(colors)],
                    label=f"{self.rolling_window}-batch Rolling Average"
                )
            
            # Add threshold line
            plt.axhline(
                y=self.alert_threshold,
                color='red',
                linestyle='--',
                label=f"Alert Threshold ({self.alert_threshold})"
            )
            
            # Add baseline level if available
            if self.baseline_metrics and "fairness_metrics" in self.baseline_metrics:
                baseline_metrics = self.baseline_metrics["fairness_metrics"]
                if metric in baseline_metrics and "difference" in baseline_metrics[metric]:
                    baseline_diff = baseline_metrics[metric]["difference"]
                    plt.axhline(
                        y=baseline_diff,
                        color='green',
                        linestyle=':',
                        label=f"Baseline ({baseline_diff:.4f})"
                    )
            
            # Customize plot
            plt.title(f"Trend for {metric}", fontsize=14)
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Fairness Disparity", fontsize=12)
            plt.grid(alpha=0.3)
            plt.legend()
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Highlight points above threshold
            above_threshold = metric_data[metric_data['difference'] > self.alert_threshold]
            if not above_threshold.empty:
                plt.scatter(
                    above_threshold['timestamp'],
                    above_threshold['difference'],
                    color='red',
                    s=100,
                    zorder=5,
                    label="Alert"
                )
            
            plt.tight_layout()
            
            # Save plot
            clean_metric_name = metric.replace('/', '_').replace(' ', '_')
            plt.savefig(os.path.join(vis_dir, f"{clean_metric_name}_trend.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_comparison_visualizations(self, vis_dir, batch_id):
        """
        Create visualizations comparing the current batch to previous batches.
        
        Args:
            vis_dir: Directory to save visualizations
            batch_id: ID of the current batch
        """
        # Get the latest two batches
        if len(self.history) < 2:
            return
        
        current_batch = self.history[-1]
        previous_batch = self.history[-2]
        
        current_metrics = current_batch['metrics']
        previous_metrics = previous_batch['metrics']
        
        # Compare group metrics
        for attr_name, current_group_metrics in current_metrics.get('group_metrics', {}).items():
            if attr_name not in previous_metrics.get('group_metrics', {}):
                continue
            
            previous_group_metrics = previous_metrics['group_metrics'][attr_name]
            
            # Get all groups
            all_groups = set(current_group_metrics.keys()) | set(previous_group_metrics.keys())
            
            # Prepare data for comparison
            groups = []
            current_means = []
            previous_means = []
            
            for group in all_groups:
                groups.append(group)
                
                if group in current_group_metrics:
                    current_means.append(current_group_metrics[group].get('mean_score', 0))
                else:
                    current_means.append(0)
                    
                if group in previous_group_metrics:
                    previous_means.append(previous_group_metrics[group].get('mean_score', 0))
                else:
                    previous_means.append(0)
            
            # Create comparison plot
            plt.figure(figsize=(10, 6))
            
            x = np.arange(len(groups))
            width = 0.35
            
            plt.bar(x - width/2, previous_means, width, label=f"Previous ({previous_batch['batch_id']})")
            plt.bar(x + width/2, current_means, width, label=f"Current ({current_batch['batch_id']})")
            
            plt.xlabel('Groups')
            plt.ylabel('Mean Score')
            plt.title(f'Comparison of {attr_name} Groups Between Batches')
            plt.xticks(x, groups)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, f"{batch_id}_{attr_name}_comparison.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_alert_visualizations(self, vis_dir, batch_id, alerts):
        """
        Create visualizations for alerts.
        
        Args:
            vis_dir: Directory to save visualizations
            batch_id: ID of the current batch
            alerts: Dictionary containing alert information
        """
        if not alerts:
            return
        
        # Create alert directory
        alert_dir = os.path.join(self.monitoring_dir, 'alerts', batch_id)
        os.makedirs(alert_dir, exist_ok=True)
        
        # Create alert summary visualization
        plt.figure(figsize=(12, 6))
        
        metrics = []
        changes = []
        severities = []
        
        for metric, alert in alerts.items():
            metrics.append(metric)
            changes.append(alert['change'])
            severities.append(alert['severity'])
        
        # Sort by change amount
        sorted_indices = np.argsort(changes)
        metrics = [metrics[i] for i in sorted_indices]
        changes = [changes[i] for i in sorted_indices]
        severities = [severities[i] for i in sorted_indices]
        
        # Create bar colors based on severity
        colors = ['orange' if s == 'medium' else 'red' for s in severities]
        
        # Create horizontal bar chart
        plt.barh(metrics, changes, color=colors)
        
        # Add threshold line
        plt.axvline(x=self.alert_threshold, color='red', linestyle='--', label=f"Alert Threshold ({self.alert_threshold})")
        
        plt.xlabel('Change in Fairness Disparity')
        plt.title(f'Fairness Alerts for Batch {batch_id}')
        plt.grid(axis='x', alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(alert_dir, f"{batch_id}_alerts.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(vis_dir, f"{batch_id}_alerts.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create individual alert visualizations
        for metric, alert in alerts.items():
            # Get metric history
            if self.history_df is not None and not self.history_df.empty:
                metric_data = self.history_df[self.history_df['metric'] == metric].sort_values('timestamp')
                
                if not metric_data.empty:
                    plt.figure(figsize=(12, 6))
                    
                    # Plot metric trend
                    plt.plot(
                        metric_data['timestamp'], 
                        metric_data['difference'], 
                        marker='o', 
                        color='blue',
                        label=metric
                    )
                    
                    # Mark the alert point
                    latest_point = metric_data.iloc[-1]
                    plt.scatter(
                        [latest_point['timestamp']],
                        [latest_point['difference']],
                        color='red',
                        s=100,
                        zorder=5,
                        label="Alert"
                    )
                    
                    # Add threshold and baseline lines
                    plt.axhline(
                        y=self.alert_threshold,
                        color='red',
                        linestyle='--',
                        label=f"Alert Threshold ({self.alert_threshold})"
                    )
                    
                    baseline_diff = alert['baseline_difference']
                    plt.axhline(
                        y=baseline_diff,
                        color='green',
                        linestyle=':',
                        label=f"Baseline ({baseline_diff:.4f})"
                    )
                    
                    # Add alert annotation
                    plt.annotate(
                        f"Alert: {alert['change']:.4f} increase",
                        xy=(latest_point['timestamp'], latest_point['difference']),
                        xytext=(10, 30),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='red'),
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7)
                    )
                    
                    # Customize plot
                    plt.title(f"Alert for {metric}", fontsize=14)
                    plt.xlabel("Time", fontsize=12)
                    plt.ylabel("Fairness Disparity", fontsize=12)
                    plt.grid(alpha=0.3)
                    plt.legend()
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    
                    # Save plot
                    clean_metric_name = metric.replace('/', '_').replace(' ', '_')
                    plt.savefig(os.path.join(alert_dir, f"{clean_metric_name}_alert.png"), dpi=300, bbox_inches='tight')
                    plt.close()
    
    def _save_history(self):
        """
        Save monitoring history to disk.
        """
        history_path = os.path.join(self.monitoring_dir, 'monitoring_history.json')
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Also save as CSV for easier analysis
        if self.history_df is not None and not self.history_df.empty:
            csv_path = os.path.join(self.monitoring_dir, 'monitoring_history.csv')
            self.history_df.to_csv(csv_path, index=False)
    
    def get_trend_analysis(self, metric_key):
        """
        Analyze trends for a specific metric.
        
        Args:
            metric_key: Metric key to analyze
            
        Returns:
            Dictionary containing trend analysis results
        """
        if self.history_df is None or self.history_df.empty:
            return {"error": "No history data available"}
        
        # Filter data for the specific metric
        metric_data = self.history_df[self.history_df['metric'] == metric_key].sort_values('timestamp')
        
        if metric_data.empty:
            return {"error": f"No data available for metric {metric_key}"}
        
        # Calculate trends
        differences = metric_data['difference'].values
        n = len(differences)
        
        if n < 2:
            return {
                "metric": metric_key,
                "current_value": float(differences[0]),
                "trend": "stable",
                "num_batches": 1
            }
        
        # Simple linear regression to get trend
        x = np.arange(n)
        A = np.vstack([x, np.ones(n)]).T
        slope, _ = np.linalg.lstsq(A, differences, rcond=None)[0]
        
        # Calculate recent change
        recent_change = differences[-1] - differences[-2]
        
        # Determine trend
        if abs(slope) < 0.01:
            trend = "stable"
        elif slope > 0:
            trend = "increasing" if slope > 0.05 else "slightly increasing"
        else:
            trend = "decreasing" if slope < -0.05 else "slightly decreasing"
        
        # Check for alerts
        alerts = []
        for i, batch in enumerate(self.history[-n:]):
            if metric_key in batch.get('alerts', {}):
                alerts.append({
                    "batch_id": batch['batch_id'],
                    "timestamp": batch['timestamp'],
                    "alert": batch['alerts'][metric_key]
                })
        
        return {
            "metric": metric_key,
            "current_value": float(differences[-1]),
            "previous_value": float(differences[-2]),
            "change": float(recent_change),
            "trend": trend,
            "slope": float(slope),
            "num_batches": n,
            "alerts": alerts,
            "visualization": os.path.join(self.monitoring_dir, 'visualizations', f"{metric_key.replace('/', '_').replace(' ', '_')}_trend.png")
        }

# Example usage function
def run_monitoring_example():
    """
    Run a demonstration of the fairness monitoring system.
    """
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from app.models.ml.prediction.ad_score_predictor import AdScorePredictor
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Features
    age = np.random.normal(35, 10, n_samples)
    income = np.random.normal(50000, 15000, n_samples)
    gender = np.random.choice(['male', 'female'], size=n_samples, p=[0.6, 0.4])
    
    # Create data with bias
    gender_bias = np.zeros(n_samples)
    gender_bias[gender == 'male'] = 0.2
    
    # Create target with bias
    ad_score = 0.5 * np.random.random(n_samples) + 0.3 * (age / 50) + 0.2 * (income / 100000) + gender_bias
    ad_score = np.clip(ad_score, 0, 1)
    
    # Binary outcome
    high_performing = (ad_score >= 0.7).astype(int)
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'income': income,
        'gender': gender,
        'ad_score': ad_score,
        'high_performing': high_performing
    })
    
    # Split data
    X = data.drop(['ad_score', 'high_performing'], axis=1)
    y = data['high_performing']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Extract protected attributes
    gender_train = X_train['gender'].values
    gender_test = X_test['gender'].values
    
    # Train model
    model = AdScorePredictor()
    model.fit(X_train, y_train)
    
    # Generate baseline predictions
    baseline_preds = model.predict(X_test)
    
    # Initialize fairness monitor
    monitor = FairnessMonitor(
        protected_attributes=['gender'],
        fairness_metrics=['demographic_parity', 'equal_opportunity'],
        alert_threshold=0.05
    )
    
    # Evaluate baseline fairness
    from app.models.ml.fairness.evaluator import FairnessEvaluator
    
    evaluator = FairnessEvaluator(
        protected_attributes=['gender'],
        fairness_threshold=0.05,
        metrics=['demographic_parity', 'equal_opportunity']
    )
    
    baseline_metrics = evaluator.evaluate(baseline_preds, y_test, {'gender': gender_test})
    
    # Set baseline
    monitor.set_baseline(baseline_metrics)
    
    # Simulate multiple production batches
    
    # Batch 1: No bias change
    print("\nBatch 1: No drift")
    batch1_preds = baseline_preds * np.random.normal(1.0, 0.02, len(baseline_preds))
    batch1_preds = np.clip(batch1_preds, 0, 1)
    
    # Update monitor
    result1 = monitor.update(
        batch1_preds, 
        y_test, 
        {'gender': gender_test}, 
        batch_id="batch_001",
        timestamp=datetime.now().isoformat()
    )
    
    # Wait a bit
    import time
    time.sleep(1)
    
    # Batch 2: Small bias change
    print("\nBatch 2: Small drift")
    batch2_preds = baseline_preds * np.random.normal(1.0, 0.05, len(baseline_preds))
    # Introduce slight gender bias
    gender_mask = gender_test == 'male'
    batch2_preds[gender_mask] *= 1.05
    batch2_preds = np.clip(batch2_preds, 0, 1)
    
    # Update monitor
    result2 = monitor.update(
        batch2_preds, 
        y_test, 
        {'gender': gender_test}, 
        batch_id="batch_002",
        timestamp=datetime.now().isoformat()
    )
    
    # Wait a bit
    time.sleep(1)
    
    # Batch 3: Major bias change
    print("\nBatch 3: Significant fairness degradation")
    batch3_preds = baseline_preds * np.random.normal(1.0, 0.05, len(baseline_preds))
    # Introduce major gender bias
    batch3_preds[gender_mask] *= 1.3
    batch3_preds = np.clip(batch3_preds, 0, 1)
    
    # Update monitor
    result3 = monitor.update(
        batch3_preds, 
        y_test, 
        {'gender': gender_test}, 
        batch_id="batch_003",
        timestamp=datetime.now().isoformat()
    )
    
    # Get trend analysis
    trend = monitor.get_trend_analysis('gender_demographic_parity')
    
    print("\nMonitoring Results:")
    print(f"Trend Analysis: {trend['trend']}")
    print(f"Current Value: {trend['current_value']:.4f}")
    print(f"Slope: {trend['slope']:.4f}")
    
    if trend['alerts']:
        print("\nAlerts:")
        for alert in trend['alerts']:
            print(f"  Batch {alert['batch_id']}: {alert['alert']['change']:.4f} increase (Severity: {alert['alert']['severity']})")
    
    print("\nKey files generated:")
    print(f"- {monitor.monitoring_dir}/monitoring_history.json: Complete monitoring history")
    print(f"- {monitor.monitoring_dir}/visualizations/gender_demographic_parity_trend.png: Trend visualization")
    print(f"- {monitor.monitoring_dir}/alerts/batch_003/gender_demographic_parity_alert.png: Alert visualization")

if __name__ == "__main__":
    run_monitoring_example() 