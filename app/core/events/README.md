# Event Handling Components

This directory contains event system components for the WITHIN ML Prediction System.

## Purpose

The event system provides a robust mechanism for:
- Enabling loose coupling between system components
- Supporting asynchronous communication patterns
- Facilitating event-driven architecture
- Tracking system activities and state changes
- Integrating with external event sources and sinks

## Key Components

### Event Bus

Core event routing and distribution components:
- Central event bus for message passing
- Topic-based routing and distribution
- Event fan-out to multiple subscribers
- Priority-based event handling

### Event Publishing

Components for generating and sending events:
- Standard event publishers
- Event schema validation
- Event metadata enrichment
- Dead letter handling for failed events

### Event Subscription

Components for receiving and processing events:
- Event subscribers and handlers
- Subscription management
- Filter-based subscription
- Batch event processing

### Event Persistence

Components for storing and retrieving events:
- Event storage and archiving
- Event replay capabilities
- Event sourcing patterns
- Event history and auditing

## Usage Example

```python
from app.core.events import EventBus, EventSubscriber, EventPublisher, Event

# Create event bus
event_bus = EventBus()

# Define subscriber
class ModelTrainingCompleteSubscriber(EventSubscriber):
    def handle_event(self, event):
        model_id = event.payload.get("model_id")
        metrics = event.payload.get("metrics")
        print(f"Training completed for model {model_id} with metrics: {metrics}")

# Register subscriber
event_bus.subscribe(
    topic="model.training.complete",
    subscriber=ModelTrainingCompleteSubscriber()
)

# Create publisher
publisher = EventPublisher(event_bus)

# Publish event
publisher.publish(
    Event(
        topic="model.training.complete",
        payload={
            "model_id": "ad_score_v2",
            "metrics": {
                "accuracy": 0.92,
                "f1_score": 0.89,
                "training_time": 1240
            },
            "timestamp": "2023-05-15T14:32:10Z"
        }
    )
)
```

## Event Topics

Common event topics in the system include:

| Topic Pattern | Description | Example |
|---------------|-------------|---------|
| `model.training.*` | Model training lifecycle events | `model.training.complete` |
| `model.prediction.*` | Prediction-related events | `model.prediction.outlier` |
| `data.ingestion.*` | Data ingestion events | `data.ingestion.failed` |
| `system.health.*` | System health events | `system.health.degraded` |
| `user.feedback.*` | User feedback events | `user.feedback.correction` |

## Integration Points

- **ML Pipeline**: Events track model training, validation, and deployment
- **Monitoring**: Events feed into monitoring and alerting systems
- **Analytics**: Events are analyzed for system usage patterns
- **External Systems**: Events integrate with external message queues and webhooks

## Dependencies

- Message serialization libraries
- Concurrency and thread management
- Storage interfaces for event persistence
- Network communication for distributed event processing 