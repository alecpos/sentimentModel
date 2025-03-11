#!/usr/bin/env python3
"""
Documentation Progress Tracker

This script tracks documentation progress over time, generates reports,
and helps visualize the state of documentation.

It can also identify:
- Documentation coverage gaps
- Files that need attention
- Progress trends over time
"""

import os
import sys
import json
import datetime
import argparse
import subprocess
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import csv

# Define status priorities (for sorting and coloring)
STATUS_PRIORITY = {
    "NOT_IMPLEMENTED": 1,
    "PARTIALLY_IMPLEMENTED": 2,
    "IMPLEMENTED": 3,
    "UNKNOWN": 0
}

class DocProgressTracker:
    def __init__(self, docs_dir: str, history_file: str):
        self.docs_dir = os.path.abspath(docs_dir)
        self.history_file = history_file
        self.history = self._load_history()
        
    def _load_history(self) -> List[Dict[str, Any]]:
        """Load history from file if it exists."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Could not parse history file {self.history_file}")
                return []
        return []
    
    def _save_history(self) -> None:
        """Save history to file."""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _get_current_stats(self) -> Dict[str, int]:
        """Get current documentation statistics."""
        result = subprocess.run(
            ['python', 'filter_docs.py', '--summary'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error running filter_docs.py: {result.stderr}")
            return {}
        
        output = result.stdout.strip().split('\n')
        stats = {}
        
        # Skip header and separator lines
        for line in output[2:-2]:
            parts = line.strip().split()
            if len(parts) >= 2:
                status = parts[0]
                count = int(parts[-1])
                stats[status] = count
                
        return stats
    
    def record_snapshot(self) -> Dict[str, Any]:
        """Record a snapshot of current documentation status."""
        stats = self._get_current_stats()
        snapshot = {
            "date": datetime.datetime.now().isoformat(),
            "stats": stats,
            "total": sum(stats.values())
        }
        
        self.history.append(snapshot)
        self._save_history()
        return snapshot
    
    def generate_progress_report(self) -> str:
        """Generate a progress report showing changes over time."""
        if not self.history:
            return "No history available. Record a snapshot first."
        
        report_lines = ["# Documentation Progress Report", ""]
        
        # Current stats
        current = self.history[-1]
        report_lines.append(f"## Current Status ({current['date'][:10]})")
        report_lines.append("")
        report_lines.append("| Status | Count | Percentage |")
        report_lines.append("|--------|-------|------------|")
        
        for status, count in sorted(current['stats'].items(), 
                                   key=lambda x: STATUS_PRIORITY.get(x[0], 0)):
            percentage = count / current['total'] * 100 if current['total'] > 0 else 0
            report_lines.append(f"| {status} | {count} | {percentage:.1f}% |")
        
        report_lines.append(f"| **TOTAL** | **{current['total']}** | **100%** |")
        report_lines.append("")
        
        # Progress over time
        if len(self.history) > 1:
            report_lines.append("## Progress Over Time")
            report_lines.append("")
            report_lines.append("| Date | Total | NOT_IMPLEMENTED | PARTIALLY_IMPLEMENTED | IMPLEMENTED |")
            report_lines.append("|------|-------|-----------------|------------------------|------------|")
            
            for snapshot in self.history:
                date = snapshot['date'][:10]  # Just the date part
                total = snapshot['total']
                not_impl = snapshot['stats'].get('NOT_IMPLEMENTED', 0)
                partial = snapshot['stats'].get('PARTIALLY_IMPLEMENTED', 0)
                impl = snapshot['stats'].get('IMPLEMENTED', 0)
                
                report_lines.append(f"| {date} | {total} | {not_impl} | {partial} | {impl} |")
            
            report_lines.append("")
            
            # Calculate changes since first snapshot
            first = self.history[0]
            first_date = first['date'][:10]
            
            report_lines.append(f"## Changes Since {first_date}")
            report_lines.append("")
            report_lines.append("| Status | First Count | Current Count | Change | Change % |")
            report_lines.append("|--------|-------------|---------------|--------|----------|")
            
            for status in sorted(set(list(first['stats'].keys()) + list(current['stats'].keys())),
                                key=lambda x: STATUS_PRIORITY.get(x, 0)):
                first_count = first['stats'].get(status, 0)
                current_count = current['stats'].get(status, 0)
                change = current_count - first_count
                change_pct = (change / first_count * 100) if first_count > 0 else float('inf')
                
                change_str = f"+{change}" if change > 0 else str(change)
                change_pct_str = f"+{change_pct:.1f}%" if change_pct > 0 else f"{change_pct:.1f}%"
                
                report_lines.append(f"| {status} | {first_count} | {current_count} | {change_str} | {change_pct_str} |")
            
            total_change = current['total'] - first['total']
            total_change_pct = (total_change / first['total'] * 100) if first['total'] > 0 else float('inf')
            
            total_change_str = f"+{total_change}" if total_change > 0 else str(total_change)
            total_change_pct_str = f"+{total_change_pct:.1f}%" if total_change_pct > 0 else f"{total_change_pct:.1f}%"
            
            report_lines.append(f"| **TOTAL** | **{first['total']}** | **{current['total']}** | **{total_change_str}** | **{total_change_pct_str}** |")
        
        return "\n".join(report_lines)
    
    def export_to_csv(self, output_file: str) -> None:
        """Export history to CSV file."""
        if not self.history:
            print("No history available. Record a snapshot first.")
            return
        
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['date', 'total', 'NOT_IMPLEMENTED', 'PARTIALLY_IMPLEMENTED', 'IMPLEMENTED', 'UNKNOWN']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for snapshot in self.history:
                row = {
                    'date': snapshot['date'][:10],
                    'total': snapshot['total']
                }
                
                for status in ['NOT_IMPLEMENTED', 'PARTIALLY_IMPLEMENTED', 'IMPLEMENTED', 'UNKNOWN']:
                    row[status] = snapshot['stats'].get(status, 0)
                
                writer.writerow(row)
                
        print(f"Exported history to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Track documentation progress over time')
    parser.add_argument('--docs-dir', default='/Users/alecposner/WITHIN/docs',
                       help='Base directory containing documentation files')
    parser.add_argument('--history-file', default='doc_progress_history.json',
                       help='File to store progress history')
    parser.add_argument('--record', action='store_true',
                       help='Record a new snapshot of current status')
    parser.add_argument('--report', action='store_true',
                       help='Generate a progress report')
    parser.add_argument('--export-csv', metavar='FILE',
                       help='Export history to CSV file')
    
    args = parser.parse_args()
    
    tracker = DocProgressTracker(args.docs_dir, args.history_file)
    
    if args.record:
        snapshot = tracker.record_snapshot()
        print(f"Recorded snapshot with {snapshot['total']} total files.")
    
    if args.report:
        report = tracker.generate_progress_report()
        print(report)
    
    if args.export_csv:
        tracker.export_to_csv(args.export_csv)
    
    # If no action specified, show report
    if not (args.record or args.report or args.export_csv):
        report = tracker.generate_progress_report()
        print(report)

if __name__ == "__main__":
    main() 