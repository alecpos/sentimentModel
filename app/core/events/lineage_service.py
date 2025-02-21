from confluent_kafka import Producer
import json
from datetime import datetime

class LineageTracker:
    def __init__(self):
        self.producer = Producer({'bootstrap.servers': 'kafka:9092'})
    
    def emit_lineage_event(self, catalog_id: str, operation: str):
        self.producer.produce(
            topic='data-lineage',
            key=catalog_id,
            value=json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "operation": operation
            })
        )