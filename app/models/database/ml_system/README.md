# ML System Database Models

This directory contains SQLAlchemy database models for ML system operations in the WITHIN ML Prediction System.

DOCUMENTATION STATUS: COMPLETE

## Purpose

The ML system models provide database representations for:
- Tracking ML model metadata and versioning
- Recording training jobs and parameters
- Logging prediction requests and responses
- Storing feature importance and performance metrics
- Supporting model lifecycle management

## Future Components

This directory is prepared for implementing the following database models:

### Model Metadata

ML model tracking and versioning:
- Model identification and versions
- Model architecture and parameters
- Training dataset references
- Model file locations
- Deployment status tracking

### Training Jobs

ML training process tracking:
- Training job records
- Hyperparameters used
- Training dataset versions
- Compute resources utilized
- Training metrics and results

### Prediction Logs

Prediction request and response tracking:
- Input feature values
- Prediction outputs
- Confidence scores
- Response times
- Request metadata (user, timestamp, etc.)

### Feature Importance

Feature relevance tracking:
- Feature importance scores
- Feature contributions to predictions
- Feature drift metrics
- Global vs. local importance
- Feature correlation data

### Performance Metrics

Model performance tracking:
- Accuracy metrics (precision, recall, F1)
- Error rates and distribution
- Latency and throughput metrics
- Drift detection metrics
- A/B testing results

## Usage Example

Once implemented, ML system models would be used like this:

```python
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.database.ml_system import (
    ModelMetadata, 
    PredictionLog, 
    ModelPerformanceMetrics
)
from app.core.db import get_db

def get_model_performance_history(model_id: str, days: int = 30):
    db = next(get_db())
    
    # Get the model metadata
    model = db.query(ModelMetadata).filter(ModelMetadata.id == model_id).first()
    
    if not model:
        return None
    
    # Get performance metrics for the specified time period
    start_date = datetime.utcnow() - timedelta(days=days)
    
    metrics = db.query(ModelPerformanceMetrics)\
               .filter(ModelPerformanceMetrics.model_id == model_id)\
               .filter(ModelPerformanceMetrics.recorded_at >= start_date)\
               .order_by(ModelPerformanceMetrics.recorded_at)\
               .all()
    
    # Get recent prediction logs
    recent_logs = db.query(PredictionLog)\
                    .filter(PredictionLog.model_id == model_id)\
                    .filter(PredictionLog.timestamp >= start_date)\
                    .order_by(PredictionLog.timestamp.desc())\
                    .limit(100)\
                    .all()
    
    return {
        "model": model.to_dict(),
        "performance_history": [m.to_dict() for m in metrics],
        "recent_predictions": [log.to_dict() for log in recent_logs],
        "summary": {
            "avg_precision": sum(m.precision for m in metrics) / len(metrics) if metrics else 0,
            "avg_recall": sum(m.recall for m in metrics) / len(metrics) if metrics else 0,
            "avg_latency_ms": sum(m.avg_latency_ms for m in metrics) / len(metrics) if metrics else 0,
            "total_predictions": sum(m.prediction_count for m in metrics),
            "trend": "stable" if not metrics else "improving" if metrics[-1].f1_score > metrics[0].f1_score else "declining"
        }
    }
```

## Integration Points

- **ML Models**: Model training and evaluation records using these models
- **API Endpoints**: Prediction endpoints log requests using these models
- **Monitoring System**: Performance tracking relies on these models
- **Model Registry**: Version tracking leverages these models
- **Admin Interface**: Model management UI uses these models

## Dependencies

- SQLAlchemy ORM for database operations
- Core DB components (TimestampedModel for temporal tracking)
- JSON serialization for complex model parameters
- ML framework utilities for model metadata extraction
- Monitoring infrastructure for metric collection 