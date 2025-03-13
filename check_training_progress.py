#!/usr/bin/env python
"""
Training Progress Checker for Sentiment140 Benchmarks

This script checks the progress of sentiment analysis training jobs by reading
status files and log files to provide a real-time update on training progress.

Usage:
    python check_training_progress.py [--dir DIRECTORY]
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Check the progress of sentiment analysis training jobs")
    parser.add_argument('--dir', type=str, default='benchmark_results/traditional',
                       help='The directory containing the training job results')
    return parser.parse_args()

def format_time_elapsed(start_time_str, end_time_str=None):
    """Calculate and format the time elapsed between start and end times."""
    try:
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M:%S')
        if end_time_str:
            end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M:%S')
        else:
            end_time = datetime.now()
        
        elapsed = end_time - start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except:
        return "unknown"

def check_training_logs(output_dir):
    """Check the logs directory for training progress."""
    output_dir = Path(output_dir)
    
    # First check the main status file
    status_file = output_dir / "status.json"
    if status_file.exists():
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            print("=" * 50)
            print(f"Training Job Status: {status.get('overall_status', 'in_progress')}")
            print(f"Started at: {status.get('started_at', 'unknown')}")
            
            if 'completed_at' in status:
                print(f"Completed at: {status['completed_at']}")
                print(f"Total time: {format_time_elapsed(status['started_at'], status['completed_at'])}")
            else:
                print(f"Running for: {format_time_elapsed(status['started_at'])}")
            
            print("\nPhases:")
            for phase, data in status.get('steps', {}).items():
                phase_status = data.get('status', 'unknown')
                if phase_status == 'completed':
                    start = data.get('started_at', '')
                    end = data.get('completed_at', '')
                    print(f"  ✅ {phase.replace('_', ' ').title()}: Completed in {end}")
                elif phase_status == 'in_progress':
                    start = data.get('started_at', '')
                    print(f"  🔄 {phase.replace('_', ' ').title()}: In progress since {start}")
                else:
                    print(f"  ❓ {phase.replace('_', ' ').title()}: {phase_status}")
        except Exception as e:
            print(f"Error reading status file: {e}")
    
    # Check for progress log
    progress_file = output_dir / "training_progress.log"
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                lines = f.readlines()
            
            print("\nRecent Progress Updates:")
            # Show the last 10 lines
            for line in lines[-10:]:
                print(f"  {line.strip()}")
        except Exception as e:
            print(f"Error reading progress log: {e}")
    
    # Check for traditional training logs
    enhanced_dir = output_dir / "enhanced"
    baseline_dir = output_dir / "baseline"
    
    for subdir in [enhanced_dir, baseline_dir]:
        logs_dir = subdir / "logs"
        if logs_dir.exists() and logs_dir.is_dir():
            print(f"\nDetailed Training Logs in {subdir.name}:")
            
            # Check training status file
            training_status_file = logs_dir / "training_status.json"
            if training_status_file.exists():
                try:
                    with open(training_status_file, 'r') as f:
                        training_status = json.load(f)
                    
                    print(f"  Model: {training_status.get('model_type', 'unknown')}")
                    print(f"  Status: {training_status.get('overall_status', 'in_progress')}")
                    
                    if 'metrics' in training_status:
                        metrics = training_status['metrics']
                        print(f"  Accuracy: {metrics.get('accuracy', 'unknown'):.4f}")
                        print(f"  F1 Score: {metrics.get('f1_score', 'unknown'):.4f}")
                    
                    print("\n  Phases:")
                    for phase, data in training_status.get('phases', {}).items():
                        status = data.get('status', 'unknown')
                        if status == 'completed':
                            message = data.get('message', '')
                            print(f"    ✅ {phase}: {message}")
                        elif status == 'in_progress':
                            message = data.get('message', '')
                            started = data.get('started_at', '')
                            print(f"    🔄 {phase}: {message} (since {started})")
                        elif status == 'failed':
                            message = data.get('message', '')
                            print(f"    ❌ {phase}: {message}")
                        else:
                            print(f"    ❓ {phase}: {status}")
                except Exception as e:
                    print(f"  Error reading training status: {e}")
            
            # Check for model training progress
            model_progress_file = logs_dir / "model_training_progress.txt"
            if model_progress_file.exists():
                try:
                    with open(model_progress_file, 'r') as f:
                        lines = f.readlines()
                    
                    print("\n  Model Training Progress:")
                    for line in lines:
                        print(f"    {line.strip()}")
                except Exception as e:
                    print(f"  Error reading model training progress: {e}")
            
            # Check for fairness evaluation progress
            fairness_file = logs_dir / "fairness_evaluation_progress.txt"
            if fairness_file.exists():
                try:
                    with open(fairness_file, 'r') as f:
                        lines = f.readlines()
                    
                    print("\n  Fairness Evaluation Progress:")
                    for line in lines:
                        print(f"    {line.strip()}")
                except Exception as e:
                    print(f"  Error reading fairness evaluation progress: {e}")

def main():
    args = parse_args()
    output_dir = args.dir
    
    if not os.path.exists(output_dir):
        print(f"Error: Directory {output_dir} does not exist")
        return 1
    
    print(f"Checking training progress in {output_dir}")
    check_training_logs(output_dir)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 