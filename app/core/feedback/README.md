# Feedback Management Components

This directory contains components for managing user feedback on ML model predictions in the WITHIN ML Prediction System.

## Purpose

The feedback system provides a structured way to:
- Collect and store user feedback on model predictions
- Analyze patterns in feedback data
- Incorporate feedback into model retraining processes
- Track feedback metrics over time to measure model improvement

## Key Components

### Feedback Collection

Components that collect and validate user feedback:
- Feedback forms and API endpoints
- Validation of feedback data
- Association of feedback with specific predictions

### Feedback Storage

Components for persisting feedback data:
- Database models for feedback storage
- Query interfaces for retrieving feedback
- Aggregation utilities for summarizing feedback

### Feedback Analysis

Components for analyzing patterns in feedback:
- Detection of systematic errors in predictions
- Identification of model weaknesses
- Prioritization of issues based on feedback frequency

### Feedback Loop Integration

Components that connect feedback to the model improvement cycle:
- Integration with model retraining pipelines
- Selection of high-value feedback for manual review
- Generation of reports on feedback impact

## Usage Example

```python
from app.core.feedback import FeedbackManager

# Initialize the feedback manager
feedback_manager = FeedbackManager()

# Record user feedback for a prediction
feedback_manager.record_feedback(
    prediction_id="pred_12345",
    feedback_type="correction",
    user_id="user_789",
    corrected_value=0.85,
    original_value=0.72,
    comments="Ad performed better than predicted"
)

# Retrieve feedback for analysis
recent_feedback = feedback_manager.get_recent_feedback(days=30)
feedback_by_model = feedback_manager.group_by_model(recent_feedback)

# Generate feedback summary for a model
model_feedback_summary = feedback_manager.summarize_feedback("ad_score_v2")
```

## Integration Points

- **ML Models**: Feedback data is used to improve model accuracy
- **API Layer**: Feedback is collected through API endpoints
- **Reporting**: Feedback metrics are included in system reports
- **User Interface**: Feedback forms are presented to users in the UI

## Dependencies

- Database models for feedback storage
- Authentication system for user identification
- Model registry for associating feedback with specific model versions 