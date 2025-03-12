#!/usr/bin/env python
"""
User Feedback Integration Module

This module implements mechanisms to collect, store, and analyze user feedback
for fairness improvement of sentiment analysis predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeedbackCollector:
    """
    Collects and manages user feedback for model predictions.
    
    This class handles:
    - Collecting and storing feedback data
    - Analyzing patterns in feedback
    - Generating reports on identified issues
    - Triggering model retraining when necessary
    """
    
    def __init__(self, storage_path="./feedback_data", 
                feedback_db_file="feedback.json",
                trigger_threshold=50):
        """
        Initialize the feedback collector.
        
        Args:
            storage_path: Path to store feedback data
            feedback_db_file: Filename for feedback database
            trigger_threshold: Number of feedback items before triggering analysis
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.feedback_db_path = self.storage_path / feedback_db_file
        self.trigger_threshold = trigger_threshold
        self.feedback_count = 0
        
        # Initialize or load existing feedback database
        if self.feedback_db_path.exists():
            with open(self.feedback_db_path, 'r') as f:
                self.feedback_db = json.load(f)
                self.feedback_count = len(self.feedback_db["items"])
        else:
            self.feedback_db = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "version": "1.0"
                },
                "items": []
            }
            self._save_db()
    
    def collect_feedback(self, text: str, prediction: int, 
                        feedback_type: str, user_correction: Optional[int] = None,
                        demographic_info: Optional[Dict[str, str]] = None,
                        additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Collect user feedback on a prediction.
        
        Args:
            text: The text that was analyzed
            prediction: Model prediction (0 for negative, 1 for positive)
            feedback_type: Type of feedback ('incorrect', 'biased', 'other')
            user_correction: User's corrected label if applicable
            demographic_info: Demographic information if available
            additional_context: Any additional context information
            
        Returns:
            Feedback ID for the stored feedback
        """
        # Generate feedback ID
        feedback_id = f"fb_{len(self.feedback_db['items']) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create feedback item
        feedback_item = {
            "id": feedback_id,
            "text": text,
            "prediction": prediction,
            "feedback_type": feedback_type,
            "timestamp": datetime.now().isoformat(),
            "user_correction": user_correction,
            "demographic_info": demographic_info or {},
            "additional_context": additional_context or {},
            "status": "pending",
            "resolution": None
        }
        
        # Add to database
        self.feedback_db["items"].append(feedback_item)
        self.feedback_db["metadata"]["last_updated"] = datetime.now().isoformat()
        self._save_db()
        
        # Increment counter and check trigger
        self.feedback_count += 1
        if self.feedback_count % self.trigger_threshold == 0:
            self.analyze_feedback_patterns()
        
        return feedback_id
    
    def _save_db(self):
        """Save the feedback database to disk."""
        with open(self.feedback_db_path, 'w') as f:
            json.dump(self.feedback_db, f, indent=2)
        logger.info(f"Feedback database saved with {len(self.feedback_db['items'])} items")
    
    def get_feedback(self, feedback_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific feedback item.
        
        Args:
            feedback_id: ID of the feedback to retrieve
            
        Returns:
            Feedback item as dictionary
        """
        for item in self.feedback_db["items"]:
            if item["id"] == feedback_id:
                return item
        
        raise ValueError(f"Feedback with ID {feedback_id} not found")
    
    def update_feedback_status(self, feedback_id: str, status: str, 
                             resolution: Optional[str] = None) -> Dict[str, Any]:
        """
        Update the status of a feedback item.
        
        Args:
            feedback_id: ID of the feedback to update
            status: New status ('pending', 'reviewed', 'resolved')
            resolution: Resolution description if status is 'resolved'
            
        Returns:
            Updated feedback item
        """
        for item in self.feedback_db["items"]:
            if item["id"] == feedback_id:
                item["status"] = status
                if resolution:
                    item["resolution"] = resolution
                self._save_db()
                return item
        
        raise ValueError(f"Feedback with ID {feedback_id} not found")
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in collected feedback.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.feedback_db["items"]:
            logger.warning("No feedback data available for analysis")
            return {"status": "no_data"}
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.feedback_db["items"])
        
        # Calculate basic statistics
        total_count = len(df)
        feedback_types = df['feedback_type'].value_counts().to_dict()
        feedback_rate = feedback_types.get('incorrect', 0) / total_count if total_count > 0 else 0
        
        # Check for demographic patterns
        demographic_patterns = {}
        
        # Extract demographics from nested JSON
        try:
            # Different approach depending on whether demographic_info is already extracted
            if 'demographic_info' in df.columns and isinstance(df['demographic_info'].iloc[0], dict):
                for item in df['demographic_info']:
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if key not in df.columns:
                                df[key] = None
                            df.loc[df.index[-1], key] = value
            
            # Analyze feedback by demographic group
            demographic_cols = ['gender', 'age_group', 'location', 'ethnicity']
            for col in demographic_cols:
                if col in df.columns:
                    demographic_patterns[col] = {}
                    group_counts = df.groupby([col, 'feedback_type']).size().unstack(fill_value=0)
                    
                    for group in group_counts.index:
                        if 'incorrect' in group_counts.columns:
                            incorrect_rate = group_counts.loc[group, 'incorrect'] / group_counts.loc[group].sum()
                            demographic_patterns[col][group] = incorrect_rate
        except Exception as e:
            logger.error(f"Error analyzing demographic patterns: {str(e)}")
        
        # Check for text content patterns
        text_patterns = {}
        try:
            # Look for common words/phrases in incorrect predictions
            incorrect_texts = df[df['feedback_type'] == 'incorrect']['text'].tolist()
            if incorrect_texts:
                from collections import Counter
                import re
                
                # Tokenize and count
                words = []
                for text in incorrect_texts:
                    words.extend(re.findall(r'\b\w+\b', text.lower()))
                
                word_counts = Counter(words)
                common_words = {word: count for word, count in word_counts.most_common(20)}
                text_patterns['common_words'] = common_words
        except Exception as e:
            logger.error(f"Error analyzing text patterns: {str(e)}")
        
        # Generate analysis report
        analysis_results = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "total_feedback": total_count,
            "feedback_types": feedback_types,
            "incorrect_prediction_rate": feedback_rate,
            "demographic_patterns": demographic_patterns,
            "text_patterns": text_patterns,
            "retraining_recommended": feedback_rate > 0.2
        }
        
        # Save analysis results
        analysis_path = self.storage_path / f"analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"Feedback analysis completed and saved to {analysis_path}")
        
        return analysis_results
    
    def generate_feedback_report(self, output_format: str = 'markdown') -> str:
        """
        Generate a human-readable report of feedback patterns.
        
        Args:
            output_format: Format of the report ('markdown', 'html', 'json')
            
        Returns:
            Report as string in the specified format
        """
        analysis = self.analyze_feedback_patterns()
        
        if analysis["status"] == "no_data":
            return "No feedback data available for reporting"
        
        if output_format == 'json':
            return json.dumps(analysis, indent=2)
        
        # Generate Markdown report (can be converted to HTML later)
        report_lines = [
            "# Sentiment Analysis Feedback Report",
            "",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            "",
            f"Total feedback received: {analysis['total_feedback']}",
            f"Incorrect prediction rate: {analysis['incorrect_prediction_rate']:.2%}",
            "",
            "## Feedback by Type",
            ""
        ]
        
        # Add feedback types table
        report_lines.extend([
            "| Feedback Type | Count | Percentage |",
            "|--------------|-------|------------|"
        ])
        
        for feedback_type, count in analysis['feedback_types'].items():
            percentage = count / analysis['total_feedback'] * 100
            report_lines.append(f"| {feedback_type} | {count} | {percentage:.1f}% |")
        
        # Add demographic patterns
        if analysis['demographic_patterns']:
            report_lines.extend([
                "",
                "## Demographic Patterns",
                ""
            ])
            
            for demo_category, patterns in analysis['demographic_patterns'].items():
                report_lines.extend([
                    f"### {demo_category.title()}",
                    "",
                    "| Group | Incorrect Rate |",
                    "|-------|---------------|"
                ])
                
                for group, rate in patterns.items():
                    report_lines.append(f"| {group} | {rate:.2%} |")
                
                report_lines.append("")
        
        # Add text patterns
        if analysis['text_patterns'] and 'common_words' in analysis['text_patterns']:
            report_lines.extend([
                "## Common Words in Incorrect Predictions",
                "",
                "| Word | Frequency |",
                "|------|-----------|"
            ])
            
            for word, count in analysis['text_patterns']['common_words'].items():
                report_lines.append(f"| {word} | {count} |")
        
        # Add recommendations
        report_lines.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        if analysis['retraining_recommended']:
            report_lines.append("- **Model retraining is recommended** due to high incorrect prediction rate.")
        else:
            report_lines.append("- Model performance appears acceptable based on feedback.")
        
        # Identify demographic groups with high error rates
        high_error_groups = []
        for demo_category, patterns in analysis['demographic_patterns'].items():
            for group, rate in patterns.items():
                if rate > 0.25:  # 25% threshold for high error rate
                    high_error_groups.append(f"{group} ({demo_category})")
        
        if high_error_groups:
            report_lines.append("- Consider bias mitigation techniques for the following demographic groups:")
            for group in high_error_groups:
                report_lines.append(f"  - {group}")
        
        # Save report
        report_text = "\n".join(report_lines)
        report_path = self.storage_path / f"feedback_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.md"
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Feedback report generated and saved to {report_path}")
        
        return report_text
    
    def create_visualizations(self, output_dir: Optional[str] = None) -> List[str]:
        """
        Create visualizations of feedback patterns.
        
        Args:
            output_dir: Directory to save visualizations (defaults to storage_path)
            
        Returns:
            List of paths to generated visualization files
        """
        if not self.feedback_db["items"]:
            logger.warning("No feedback data available for visualization")
            return []
        
        # Set output directory
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = self.storage_path / "visualizations"
        
        output_path.mkdir(exist_ok=True)
        
        # Convert to DataFrame for visualization
        df = pd.DataFrame(self.feedback_db["items"])
        
        visualization_paths = []
        
        # 1. Feedback Type Distribution
        plt.figure(figsize=(10, 6))
        feedback_counts = df['feedback_type'].value_counts()
        ax = feedback_counts.plot.bar(color='skyblue')
        plt.title('Distribution of Feedback Types')
        plt.xlabel('Feedback Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add count labels
        for i, count in enumerate(feedback_counts):
            ax.text(i, count + 0.1, str(count), ha='center')
        
        # Save visualization
        feedback_type_path = output_path / "feedback_type_distribution.png"
        plt.tight_layout()
        plt.savefig(feedback_type_path, dpi=300)
        plt.close()
        
        visualization_paths.append(str(feedback_type_path))
        
        # 2. Feedback Over Time
        try:
            plt.figure(figsize=(12, 6))
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Group by date and feedback type
            df['date'] = df['timestamp'].dt.date
            time_series = df.groupby(['date', 'feedback_type']).size().unstack(fill_value=0)
            
            # Plot
            time_series.plot(kind='line', marker='o')
            plt.title('Feedback Over Time')
            plt.xlabel('Date')
            plt.ylabel('Count')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save visualization
            time_series_path = output_path / "feedback_over_time.png"
            plt.tight_layout()
            plt.savefig(time_series_path, dpi=300)
            plt.close()
            
            visualization_paths.append(str(time_series_path))
        except Exception as e:
            logger.error(f"Error creating time series visualization: {str(e)}")
        
        # 3. Demographic Breakdown
        try:
            # Extract demographic information if available
            if 'demographic_info' in df.columns:
                demo_df = pd.json_normalize(df['demographic_info'].dropna())
                
                # For each demographic dimension, create breakdown
                for col in demo_df.columns:
                    if len(demo_df[col].dropna()) > 0:
                        plt.figure(figsize=(10, 6))
                        
                        # Count of feedback by demographic group
                        demo_counts = demo_df[col].value_counts()
                        
                        # Plot
                        ax = demo_counts.plot.pie(autopct='%1.1f%%', startangle=90, 
                                                 explode=[0.05] * len(demo_counts))
                        plt.title(f'Feedback by {col.title()}')
                        plt.axis('equal')
                        
                        # Save visualization
                        demo_path = output_path / f"feedback_by_{col}.png"
                        plt.tight_layout()
                        plt.savefig(demo_path, dpi=300)
                        plt.close()
                        
                        visualization_paths.append(str(demo_path))
        except Exception as e:
            logger.error(f"Error creating demographic visualizations: {str(e)}")
        
        logger.info(f"Created {len(visualization_paths)} feedback visualizations")
        
        return visualization_paths
    
    def check_retraining_trigger(self) -> bool:
        """
        Check if retraining should be triggered based on feedback.
        
        Returns:
            True if retraining is recommended, False otherwise
        """
        analysis = self.analyze_feedback_patterns()
        
        return analysis.get("retraining_recommended", False)

# Example usage if run as script
if __name__ == "__main__":
    # Simple demo
    collector = FeedbackCollector()
    
    # Collect some feedback
    collector.collect_feedback(
        text="This product is absolutely terrible!",
        prediction=1,  # Incorrectly predicted as positive
        feedback_type="incorrect",
        user_correction=0,
        demographic_info={"gender": "female", "age_group": "25-34"}
    )
    
    collector.collect_feedback(
        text="I loved every minute of the service.",
        prediction=1,  # Correctly predicted as positive
        feedback_type="correct",
        demographic_info={"gender": "male", "age_group": "35-44"}
    )
    
    # Generate report
    report = collector.generate_feedback_report()
    print(report)
    
    # Create visualizations
    viz_paths = collector.create_visualizations()
    print(f"Visualizations created: {viz_paths}") 