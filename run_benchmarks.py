#!/usr/bin/env python
"""
ML Model Benchmark Runner

This script runs comprehensive benchmark tests on ML models to evaluate 
performance against documented benchmarks and validate fairness properties.

Usage:
    python run_benchmarks.py [--output-dir DIR] [--fairness] [--performance]

Options:
    --output-dir DIR   Directory to save benchmark results [default: benchmark_results]
    --fairness         Run fairness benchmark tests
    --performance      Run performance benchmark tests
    --all              Run all benchmark tests (default)
"""

import os
import sys
import argparse
import subprocess
import json
import logging
import datetime
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark_run.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("benchmark_runner")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ML model benchmarks")
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="benchmark_results",
        help="Directory to save benchmark results"
    )
    parser.add_argument(
        "--fairness", 
        action="store_true",
        help="Run fairness benchmark tests"
    )
    parser.add_argument(
        "--performance", 
        action="store_true",
        help="Run performance benchmark tests"
    )
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Run all benchmark tests"
    )
    parser.add_argument(
        "--compare", 
        action="store_true",
        help="Compare results with previous benchmarks"
    )
    parser.add_argument(
        "--report", 
        action="store_true",
        help="Generate detailed report"
    )
    
    args = parser.parse_args()
    
    # If no specific flags are set, run everything
    if not (args.fairness or args.performance or args.all):
        args.all = True
    
    # If --all is set, enable all benchmarks
    if args.all:
        args.fairness = True
        args.performance = True
    
    return args

def run_performance_benchmarks(output_dir: str) -> Dict[str, Any]:
    """Run performance benchmark tests.
    
    Args:
        output_dir: Directory to save benchmark results
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info("Running performance benchmark tests...")
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run benchmark tests using pytest
    cmd = [
        "python", "-m", "pytest", 
        "tests/benchmark/test_model_benchmarks.py", 
        "-v"
    ]
    
    # Pass the output directory via environment variable instead
    env = os.environ.copy()
    env["BENCHMARK_OUTPUT_DIR"] = output_dir
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            check=True,
            env=env  # Use the modified environment
        )
        logger.info(f"Performance benchmarks completed with: {result.returncode}")
        
        # Log output
        if result.stdout:
            logger.info(f"Performance benchmark output: {result.stdout}")
        
        # Process results (look for JSON files in output directory)
        json_files = list(Path(output_dir).glob("*benchmark*.json"))
        results = {}
        
        for jf in json_files:
            if jf.is_file() and jf.suffix == '.json':
                try:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                        model_name = data.get('model_name', jf.stem)
                        results[model_name] = data
                except Exception as e:
                    logger.error(f"Error loading benchmark results from {jf}: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Performance benchmarks completed in {elapsed_time:.2f} seconds")
        
        return {
            "success": True,
            "results": results,
            "elapsed_time": elapsed_time
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Performance benchmarks failed with code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        
        return {
            "success": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }

def run_fairness_benchmarks(output_dir: str) -> Dict[str, Any]:
    """Run fairness benchmark tests.
    
    Args:
        output_dir: Directory to save benchmark results
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info("Running fairness benchmark tests...")
    start_time = time.time()
    
    # Create output directory
    fairness_dir = os.path.join(output_dir, "fairness")
    os.makedirs(fairness_dir, exist_ok=True)
    
    # Run fairness tests using pytest
    cmd = [
        "python", "-m", "pytest", 
        "tests/property-based/test_fairness_properties.py", 
        "-v"
    ]
    
    # Pass the output directory via environment variable instead
    env = os.environ.copy()
    env["FAIRNESS_OUTPUT_DIR"] = fairness_dir
    
    try:
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            check=True,
            env=env  # Use the modified environment
        )
        logger.info(f"Fairness benchmarks completed with: {result.returncode}")
        
        # Log output
        if result.stdout:
            logger.info(f"Fairness benchmark output: {result.stdout}")
        
        # Process results (look for JSON files in fairness directory)
        json_files = list(Path(fairness_dir).glob("*fairness*.json"))
        results = {}
        
        for jf in json_files:
            if jf.is_file() and jf.suffix == '.json':
                try:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                        model_name = data.get('model_name', jf.stem)
                        attribute = data.get('protected_attribute', 'unknown')
                        key = f"{model_name}_{attribute}"
                        results[key] = data
                except Exception as e:
                    logger.error(f"Error loading fairness results from {jf}: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Fairness benchmarks completed in {elapsed_time:.2f} seconds")
        
        return {
            "success": True,
            "results": results,
            "elapsed_time": elapsed_time
        }
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Fairness benchmarks failed with code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        
        return {
            "success": False,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }

def compare_with_previous(current_results: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Compare current benchmark results with previous runs.
    
    Args:
        current_results: Current benchmark results
        output_dir: Directory with benchmark results
        
    Returns:
        Dictionary with comparison results
    """
    logger.info("Comparing with previous benchmark results...")
    
    # Find previous benchmark results
    prev_dirs = [
        d for d in os.listdir(output_dir) 
        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith('20')
    ]
    
    if not prev_dirs:
        logger.info("No previous benchmark results found for comparison")
        return {
            "success": True,
            "message": "No previous benchmarks found"
        }
    
    # Sort directories by date (newest first, excluding current)
    prev_dirs.sort(reverse=True)
    
    if len(prev_dirs) > 0:
        prev_dir = os.path.join(output_dir, prev_dirs[0])
        logger.info(f"Comparing with previous benchmarks from: {prev_dir}")
        
        # Load previous results
        prev_results = {}
        json_files = list(Path(prev_dir).glob("*.json"))
        
        for jf in json_files:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    if 'model_name' in data:
                        model_name = data['model_name']
                        prev_results[model_name] = data
            except Exception as e:
                logger.error(f"Error loading previous results from {jf}: {e}")
        
        # Compare metrics
        comparison = {}
        
        for model_name, current in current_results.get('performance', {}).get('results', {}).items():
            if model_name in prev_results:
                prev = prev_results[model_name]
                
                # Compare metrics
                metric_diff = {}
                
                # Process current metrics for comparison
                current_metrics = current.get('metrics', {})
                prev_metrics = prev.get('metrics', {})
                
                for metric in set(current_metrics.keys()).intersection(prev_metrics.keys()):
                    current_val = current_metrics[metric]
                    prev_val = prev_metrics[metric]
                    diff = current_val - prev_val
                    
                    # Calculate percent change
                    if prev_val != 0:
                        percent_change = (diff / prev_val) * 100
                    else:
                        percent_change = float('inf') if diff > 0 else 0
                    
                    # Determine if change is improvement
                    is_improvement = False
                    
                    # For RMSE, lower is better
                    if metric.startswith('RMSE'):
                        is_improvement = diff < 0
                    # For most other metrics, higher is better
                    else:
                        is_improvement = diff > 0
                    
                    metric_diff[metric] = {
                        'current': current_val,
                        'previous': prev_val,
                        'absolute_diff': diff,
                        'percent_change': percent_change,
                        'is_improvement': is_improvement
                    }
                
                comparison[model_name] = {
                    'metrics': metric_diff,
                    'overall_success': current.get('success', False),
                    'previous_success': prev.get('success', False)
                }
        
        # Save comparison results
        comparison_file = os.path.join(output_dir, "benchmark_comparison.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        logger.info(f"Saved benchmark comparison to: {comparison_file}")
        
        return {
            "success": True,
            "comparison": comparison,
            "previous_date": prev_dirs[0]
        }
    
    return {
        "success": True,
        "message": "No previous benchmarks found for comparison"
    }

def generate_report(benchmark_results: Dict[str, Any], output_dir: str) -> str:
    """Generate a comprehensive benchmark report.
    
    Args:
        benchmark_results: Results from benchmark runs
        output_dir: Directory to save the report
        
    Returns:
        Path to generated report
    """
    logger.info("Generating benchmark report...")
    
    report = []
    report.append("# ML Model Benchmark Report")
    report.append(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Add performance benchmarks section
    report.append("## Performance Benchmarks")
    
    performance_results = benchmark_results.get('performance', {}).get('results', {})
    if performance_results:
        report.append("### Overall Performance Metrics\n")
        
        # Create performance summary table
        report.append("| Model | RMSE | R² | Spearman | Precision@10 | Recall@10 | Status |")
        report.append("|-------|------|----|----|--------------|-----------|--------|")
        
        for model_name, result in performance_results.items():
            metrics = result.get('metrics', {})
            success = result.get('success', False)
            
            rmse = metrics.get('RMSE', 'N/A')
            r2 = metrics.get('R²', 'N/A')
            spearman = metrics.get('Spearman', 'N/A')
            precision = metrics.get('Precision@10', 'N/A')
            recall = metrics.get('Recall@10', 'N/A')
            
            # Format values
            if isinstance(rmse, (int, float)):
                rmse = f"{rmse:.4f}"
            if isinstance(r2, (int, float)):
                r2 = f"{r2:.4f}"
            if isinstance(spearman, (int, float)):
                spearman = f"{spearman:.4f}"
            if isinstance(precision, (int, float)):
                precision = f"{precision:.4f}"
            if isinstance(recall, (int, float)):
                recall = f"{recall:.4f}"
            
            status = "✅ PASS" if success else "❌ FAIL"
            
            report.append(f"| {model_name} | {rmse} | {r2} | {spearman} | {precision} | {recall} | {status} |")
        
        report.append("\n")
    else:
        report.append("No performance benchmark results available.\n")
    
    # Add fairness benchmarks section
    report.append("## Fairness Benchmarks")
    
    fairness_results = benchmark_results.get('fairness', {}).get('results', {})
    if fairness_results:
        report.append("### Fairness Metrics\n")
        
        # Create fairness summary table
        report.append("| Model | Protected Attribute | Demographic Parity | Equal Opportunity | Status |")
        report.append("|-------|---------------------|-------------------|------------------|--------|")
        
        for key, result in fairness_results.items():
            model_name = result.get('model_name', 'Unknown')
            attr = result.get('protected_attribute', 'Unknown')
            metrics = result.get('metrics', {})
            passed = result.get('passed', False)
            
            dp_diff = metrics.get('demographic_parity_difference', 'N/A')
            eo_diff = metrics.get('equal_opportunity_difference', 'N/A')
            
            # Format values
            if isinstance(dp_diff, (int, float)):
                dp_diff = f"{dp_diff:.4f}"
            if isinstance(eo_diff, (int, float)):
                eo_diff = f"{eo_diff:.4f}"
            
            status = "✅ PASS" if passed else "❌ FAIL"
            
            report.append(f"| {model_name} | {attr} | {dp_diff} | {eo_diff} | {status} |")
        
        report.append("\n")
    else:
        report.append("No fairness benchmark results available.\n")
    
    # Add comparison section
    if 'comparison' in benchmark_results:
        report.append("## Comparison with Previous Benchmarks")
        
        comparison = benchmark_results.get('comparison', {}).get('comparison', {})
        prev_date = benchmark_results.get('comparison', {}).get('previous_date', 'Unknown')
        
        report.append(f"Comparing with benchmarks from: {prev_date}\n")
        
        if comparison:
            # Create comparison table
            report.append("| Model | Metric | Current | Previous | Change | Status |")
            report.append("|-------|--------|---------|----------|--------|--------|")
            
            for model_name, model_comparison in comparison.items():
                metrics = model_comparison.get('metrics', {})
                
                for metric, values in metrics.items():
                    current = values.get('current', 'N/A')
                    previous = values.get('previous', 'N/A')
                    percent = values.get('percent_change', 'N/A')
                    is_improvement = values.get('is_improvement', False)
                    
                    # Format values
                    if isinstance(current, (int, float)):
                        current = f"{current:.4f}"
                    if isinstance(previous, (int, float)):
                        previous = f"{previous:.4f}"
                    if isinstance(percent, (int, float)):
                        percent = f"{percent:+.2f}%"
                    
                    status = "✅ Better" if is_improvement else "❌ Worse"
                    
                    report.append(f"| {model_name} | {metric} | {current} | {previous} | {percent} | {status} |")
            
            report.append("\n")
        else:
            report.append("No comparison data available.\n")
    
    # Write report to file
    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, "benchmark_report.md")
    
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Benchmark report saved to: {report_path}")
    
    return report_path

def main() -> None:
    """Run benchmark tests and generate reports."""
    args = parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create run-specific directory
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    
    logger.info(f"Starting benchmark run {timestamp}")
    logger.info(f"Output directory: {run_dir}")
    
    all_results = {}
    
    # Run performance benchmarks
    if args.performance:
        perf_results = run_performance_benchmarks(run_dir)
        all_results['performance'] = perf_results
    
    # Run fairness benchmarks
    if args.fairness:
        fairness_results = run_fairness_benchmarks(run_dir)
        all_results['fairness'] = fairness_results
    
    # Compare with previous runs
    if args.compare or args.report:
        comparison = compare_with_previous(all_results, output_dir)
        all_results['comparison'] = comparison
    
    # Generate report
    if args.report:
        report_path = generate_report(all_results, run_dir)
        logger.info(f"Benchmark report available at: {report_path}")
    
    # Save run metadata
    metadata = {
        "timestamp": timestamp,
        "runtime": {
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": sys.version,
            "command_args": vars(args)
        },
        "summary": {
            "performance_success": all_results.get('performance', {}).get('success', False),
            "fairness_success": all_results.get('fairness', {}).get('success', False)
        }
    }
    
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create symlink to latest run
    latest_link = os.path.join(output_dir, "latest")
    if os.path.exists(latest_link):
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        else:
            os.rename(latest_link, f"{latest_link}_backup_{timestamp}")
    
    os.symlink(run_dir, latest_link, target_is_directory=True)
    
    logger.info(f"Benchmark run completed. Results saved to: {run_dir}")
    logger.info(f"Latest results link: {latest_link}")

if __name__ == "__main__":
    main() 