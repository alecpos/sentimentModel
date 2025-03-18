# tests/conftest.py
import pytest
import numpy as np
import pandas as pd
import torch
import json
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from cryptography.fernet import Fernet
from app.core.database import Base
from app.config import settings
from unittest.mock import MagicMock, patch, Mock, ANY
from app.core.data_lake.security_manager import SecurityManager, PolicyEngine, EncryptionService, AuditLogger
from app.models.domain.data_catalog_model import DataCatalogModel
from app.models.domain.data_lake_model import DataLakeModel
from app.core.search.search_service import DataCatalogSearch
from app.core.feedback.feedback_handler import FeedbackProcessor
from app.core.validation.data_quality_validator import DataQualityValidator
import sys
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Database Core Fixtures
@pytest.fixture(scope="session")
def test_db_engine():
    """Session-wide test database engine with clean schema"""
    engine = create_engine(settings.TEST_DATABASE_URL)
    
    # Drop all tables to ensure clean state
    Base.metadata.drop_all(engine)
    
    # Create all tables defined in the models
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Clean up after all tests
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def db_session(test_db_engine):
    """Function-scoped database session with rollback"""
    connection = test_db_engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()
    yield session
    session.close()
    transaction.rollback()
    connection.close()

# Security Fixtures
@pytest.fixture
def mock_security():
    """Create a mock security manager that allows all operations by default"""
    # Create mock components
    policy_engine = PolicyEngine(
        role_assignments={"test_user": "admin"},
        resource_policies={
            "catalog": {
                "admin": {
                    "ingest": True,
                    "process": True,
                    "curate": True,
                    "read": True,
                    "write": True
                }
            }
        }
    )
    
    encryption_service = EncryptionService(Fernet.generate_key())
    audit_logger = AuditLogger()
    
    # Create the security manager with real components
    security_manager = SecurityManager(
        policy_engine=policy_engine,
        encryption_service=encryption_service,
        audit_logger=audit_logger
    )
    
    return security_manager

# External Service Mocks
@pytest.fixture
def mock_elasticsearch(monkeypatch):
    """Mock Elasticsearch client for search tests"""
    from elasticsearch import Elasticsearch
    
    class MockES:
        def __init__(self):
            self.indexed = {}
        
        def index(self, index, id, body):
            self.indexed[id] = body
            
        def search(self, **kwargs):
            return {"hits": {"hits": list(self.indexed.values())}}
    
    monkeypatch.setattr(Elasticsearch, '__init__', lambda *args, **kwargs: MockES())
    return Elasticsearch()

@pytest.fixture
def mock_kafka(monkeypatch):
    """Mock Kafka producer for lineage tracking"""
    class MockProducer:
        def __init__(self, config):
            self.messages = []
            
        def produce(self, topic, key, value):
            self.messages.append((topic, key, value))
    
    monkeypatch.setattr('confluent_kafka.Producer', MockProducer)
    return MockProducer({})

# Test Data Factories
@pytest.fixture
def sample_data():
    """Common test data templates"""
    return {
        "data_catalog": {
            "name": "Test Dataset",
            "data_lake_id": "dl_123",
            "schema_definition": {"fields": [{"name": "test", "type": "string"}]}
        },
        "data_lake": {
            "name": "Test Entry",
            "data": b"test_data",
            "meta_info": {"source": "test", "level": "user", "data_layer": "raw"}
        }
    }
    


@pytest.fixture
def mock_es():
    """Create a mock Elasticsearch client"""
    mock = MagicMock()
    
    # Mock index existence check
    mock.indices.exists.return_value = False
    
    # Mock index operation
    def mock_index(index, document, **kwargs):
        return {"_index": index, "_id": "test_id", "result": "created"}
    mock.index = MagicMock(side_effect=mock_index)
    
    # Mock bulk operation
    def mock_bulk(body, **kwargs):
        return {
            "took": 30,
            "errors": False,
            "items": [{"index": {"_id": "test_id", "status": 201}} for _ in range(len(body)//2)]
        }
    mock.bulk = MagicMock(side_effect=mock_bulk)
    
    # Mock search results
    mock.search.return_value = {
        "hits": {
            "total": {"value": 1},
            "hits": [{
                "_source": {
                    "name": "Test Dataset",
                    "description": "Test description",
                    "meta_info": {"level": "campaign"},
                    "level_context": {"date": "2024-01-01"},
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-02T00:00:00"
                }
            }]
        },
        "aggregations": {
            "level": {"buckets": [{"key": "campaign", "doc_count": 1}]},
            "data_layer": {"buckets": [{"key": "raw", "doc_count": 1}]}
        }
    }
    
    return mock

@pytest.fixture
def search_service(mock_es):
    """Create search service instance"""
    return DataCatalogSearch(es_client=mock_es)

@pytest.fixture
def feedback_processor(db_session):
    """Create feedback processor instance"""
    return FeedbackProcessor(db_session)

@pytest.fixture
def data_validator(db_session):
    """Create data validator instance"""
    return DataQualityValidator(db_session)

@pytest.fixture
def sample_lake_entry(db_session):
    """Create a sample data lake entry"""
    entry = DataLakeModel(
        name="test_lake_entry",
        data=b"test_data",
        meta_info={
            "level": "campaign",
            "data_layer": "raw"
        }
    )
    db_session.add(entry)
    db_session.flush()
    return entry

@pytest.fixture
def sample_catalog_entry(db_session, sample_lake_entry):
    """Create a sample catalog entry for testing"""
    entry = DataCatalogModel(
        name="Test Dataset",
        description="Test description",
        data_lake_id=sample_lake_entry.id,
        schema_definition={
            "fields": [
                {"name": "id", "type": "string", "description": "Primary identifier"},
                {"name": "timestamp", "type": "datetime", "description": "Event timestamp"}
            ]
        },
        meta_info={
            "level": "campaign",
            "data_layer": "raw",
            "owner": "data_team",
            "sensitivity": "low",
            "data_quality": {
                "completeness": 0.95,
                "accuracy": 0.98
            }
        },
        level_context={
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "target_demographic": "all"
        }
    )
    return entry

@pytest.fixture
def lineage_entries(db_session):
    """Create source and derived entries for lineage testing"""
    source = DataCatalogModel(
        name="Source Dataset",
        data_lake_id="dl_123",
        meta_info={
            "level": "campaign",
            "data_layer": "raw"
        },
        level_context={
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "target_demographic": "all"
        }
    )
    db_session.add(source)
    
    derived = DataCatalogModel(
        name="Derived Dataset",
        data_lake_id="dl_456",
        meta_info={
            "level": "campaign",
            "data_layer": "processed"
        },
        level_context={
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "target_demographic": "all"
        }
    )
    db_session.add(derived)
    db_session.commit()
    return {"source": source, "derived": derived}

# ML Testing Fixtures

@pytest.fixture(scope="session")
def synthetic_tabular_data():
    """Generate synthetic tabular data for model testing"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate feature matrix with some correlated features
    X = np.random.randn(n_samples, n_features)
    
    # Add correlation between some features
    X[:, 5] = 0.7 * X[:, 0] + 0.3 * np.random.randn(n_samples)
    X[:, 6] = 0.8 * X[:, 1] + 0.2 * np.random.randn(n_samples)
    
    # Generate target with relationship to features
    y = 2 * X[:, 0] + 1.5 * X[:, 1] - 1 * X[:, 2] + 0.5 * X[:, 3] + np.random.randn(n_samples) * 0.5
    
    # Convert to pandas DataFrame with column names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    
    return df

@pytest.fixture(scope="session")
def demographic_data():
    """Generate synthetic demographic data for fairness testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create balanced demographic attributes
    gender = np.random.choice(["male", "female", "non_binary"], n_samples, p=[0.48, 0.48, 0.04])
    age_group = np.random.choice(["18-24", "25-34", "35-44", "45-54", "55+"], n_samples)
    ethnicity = np.random.choice(
        ["group_a", "group_b", "group_c", "group_d", "group_e"], 
        n_samples, 
        p=[0.6, 0.15, 0.15, 0.05, 0.05]
    )
    region = np.random.choice(["north", "south", "east", "west", "central"], n_samples)
    income = np.random.choice(["low", "medium", "high"], n_samples, p=[0.3, 0.5, 0.2])
    
    # Create DataFrame with demographics
    df = pd.DataFrame({
        "gender": gender,
        "age_group": age_group,
        "ethnicity": ethnicity,
        "region": region,
        "income": income
    })
    
    return df

@pytest.fixture(scope="session")
def fairness_test_data(synthetic_tabular_data, demographic_data):
    """Combine synthetic features with demographic data for fairness testing"""
    # Combine dataframes
    data = pd.concat([synthetic_tabular_data.reset_index(drop=True), 
                     demographic_data.reset_index(drop=True)], axis=1)
    
    # Add some bias to make testing meaningful
    # Lower scores for certain demographic groups
    mask_condition = (data["ethnicity"] == "group_b") & (data["gender"] == "female")
    data.loc[mask_condition, "target"] = data.loc[mask_condition, "target"] * 0.8
    
    # Higher scores for another demographic group
    mask_condition = (data["ethnicity"] == "group_a") & (data["income"] == "high")
    data.loc[mask_condition, "target"] = data.loc[mask_condition, "target"] * 1.2
    
    # Create train/test split
    np.random.seed(42)
    indices = np.random.permutation(len(data))
    train_size = int(0.7 * len(data))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]
    
    # Format data as expected by test functions
    result = {
        "X_train": train_data.drop(["target"], axis=1),
        "y_train": train_data["target"],
        "X_test": test_data.drop(["target"], axis=1),
        "y_test": test_data["target"],
        "protected_attributes": ["gender", "ethnicity", "age_group", "income"],
        "full_train_data": train_data,
        "full_test_data": test_data
    }
    
    return result

@pytest.fixture(scope="session")
def drift_test_data():
    """Generate data for drift detection testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create reference dataset with known distribution
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.exponential(2, n_samples),
        'feature3': np.random.uniform(-1, 1, n_samples),
        'feature4': np.random.poisson(5, n_samples),
        'feature5': np.random.normal(5, 2, n_samples),
        'cat1': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.6, 0.3, 0.1]),
        'cat2': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
    })
    
    # Create datasets with different types of drift
    np.random.seed(43)  # Different seed
    
    # Slight drift - minor changes to distributions
    slight_drift_data = pd.DataFrame({
        'feature1': np.random.normal(0.1, 1, n_samples),  # Small mean shift
        'feature2': np.random.exponential(2.2, n_samples),  # Small parameter shift
        'feature3': np.random.uniform(-1, 1, n_samples),  # No drift
        'feature4': np.random.poisson(5, n_samples),  # No drift
        'feature5': np.random.normal(5, 2, n_samples),  # No drift
        'cat1': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.58, 0.31, 0.11]),  # Small probability shift
        'cat2': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples, p=[0.4, 0.3, 0.2, 0.1])  # No drift
    })
    
    # Moderate drift - noticeable changes
    moderate_drift_data = pd.DataFrame({
        'feature1': np.random.normal(0.3, 1.1, n_samples),  # Medium mean and std shift
        'feature2': np.random.exponential(2.5, n_samples),  # Medium parameter shift
        'feature3': np.random.uniform(-1, 1, n_samples),  # No drift
        'feature4': np.random.poisson(5.5, n_samples),  # Medium parameter shift
        'feature5': np.random.normal(5, 2.2, n_samples),  # Medium std shift
        'cat1': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.55, 0.35, 0.1]),  # Medium probability shift
        'cat2': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples, p=[0.35, 0.35, 0.2, 0.1])  # Medium probability shift
    })
    
    # Severe drift - major distribution changes
    severe_drift_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, n_samples),  # Large mean and std shift
        'feature2': np.random.exponential(3, n_samples),  # Large parameter shift
        'feature3': np.random.uniform(-0.8, 1.2, n_samples),  # Range shift
        'feature4': np.random.poisson(7, n_samples),  # Large parameter shift
        'feature5': np.random.normal(6, 2.5, n_samples),  # Large mean and std shift
        'cat1': np.random.choice(['A', 'B', 'C'], n_samples, p=[0.4, 0.4, 0.2]),  # Large probability shift
        'cat2': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples, p=[0.25, 0.25, 0.25, 0.25])  # Uniform distribution
    })
    
    # Concept drift - feature relationships change
    np.random.seed(44)
    # Create data with different feature relationships for concept drift
    X = np.random.randn(n_samples, 5)
    # Original relationship: feature0 has most weight
    y_ref = 3 * X[:, 0] + 1 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(n_samples) * 0.2
    # New relationship: feature1 has most weight
    y_drift = 1 * X[:, 0] + 3 * X[:, 1] + 1.5 * X[:, 2] + np.random.randn(n_samples) * 0.2
    
    concept_ref_data = pd.DataFrame(
        np.column_stack([X, y_ref]), 
        columns=[f'feature{i}' for i in range(5)] + ['target']
    )
    
    concept_drift_data = pd.DataFrame(
        np.column_stack([X, y_drift]), 
        columns=[f'feature{i}' for i in range(5)] + ['target']
    )
    
    return {
        "reference_data": reference_data,
        "slight_drift_data": slight_drift_data,
        "moderate_drift_data": moderate_drift_data,
        "severe_drift_data": severe_drift_data,
        "concept_reference_data": concept_ref_data,
        "concept_drift_data": concept_drift_data
    }

@pytest.fixture(scope="session")
def golden_test_queries():
    """Provide a set of golden test queries for model validation"""
    return [
        {
            "id": "golden_1",
            "headline": "Amazing new product launches today",
            "description": "Our revolutionary product is now available to the public",
            "expected_score": 85,
            "tolerance": 5
        },
        {
            "id": "golden_2",
            "headline": "Limited time offer: 50% discount",
            "description": "Get this amazing deal before it expires",
            "expected_score": 78,
            "tolerance": 5
        },
        {
            "id": "golden_3",
            "headline": "Free shipping on all orders",
            "description": "No minimum purchase required",
            "expected_score": 72,
            "tolerance": 5
        },
        {
            "id": "golden_4",
            "headline": "Product recall announcement",
            "description": "Safety issue discovered in recent batch",
            "expected_score": 35,
            "tolerance": 7
        },
        {
            "id": "golden_5",
            "headline": "Join our loyalty program",
            "description": "Earn points with every purchase",
            "expected_score": 65,
            "tolerance": 5
        }
    ]

@pytest.fixture
def mock_model_registry():
    """Provide a mock model registry for testing model versioning and loading"""
    class MockModelRegistry:
        def __init__(self):
            self.models = {
                "ad_score_predictor": {
                    "v1.0.0": {
                        "path": "models/ad_score_predictor_v1.pkl",
                        "metrics": {
                            "rmse": 5.2,
                            "mae": 3.8,
                            "r2": 0.82
                        },
                        "training_date": "2023-01-15",
                        "status": "archived"
                    },
                    "v1.1.0": {
                        "path": "models/ad_score_predictor_v1_1.pkl",
                        "metrics": {
                            "rmse": 4.8,
                            "mae": 3.5,
                            "r2": 0.85
                        },
                        "training_date": "2023-03-20",
                        "status": "production"
                    },
                    "v2.0.0-beta": {
                        "path": "models/ad_score_predictor_v2_beta.pkl",
                        "metrics": {
                            "rmse": 4.5,
                            "mae": 3.3,
                            "r2": 0.87
                        },
                        "training_date": "2023-05-10",
                        "status": "staging"
                    }
                },
                "account_health_predictor": {
                    "v1.0.0": {
                        "path": "models/account_health_v1.pkl",
                        "metrics": {
                            "accuracy": 0.88,
                            "f1": 0.85,
                            "auc": 0.92
                        },
                        "training_date": "2023-02-05",
                        "status": "production"
                    }
                }
            }
        
        def get_model_info(self, model_name, version=None):
            if model_name not in self.models:
                return None
            
            if version is None:
                # Return info for production version
                for ver, info in self.models[model_name].items():
                    if info["status"] == "production":
                        return {"version": ver, **info}
                return None
            
            if version not in self.models[model_name]:
                return None
                
            return {"version": version, **self.models[model_name][version]}
        
        def get_model_path(self, model_name, version=None):
            info = self.get_model_info(model_name, version)
            if info:
                return info["path"]
            return None
            
        def register_model(self, model_name, version, path, metrics, status="staging"):
            if model_name not in self.models:
                self.models[model_name] = {}
                
            self.models[model_name][version] = {
                "path": path,
                "metrics": metrics,
                "training_date": datetime.now().strftime("%Y-%m-%d"),
                "status": status
            }
            
        def promote_model(self, model_name, version, target_status="production"):
            if model_name not in self.models or version not in self.models[model_name]:
                return False
                
            # If promoting to production, demote current production model
            if target_status == "production":
                for ver, info in self.models[model_name].items():
                    if info["status"] == "production":
                        info["status"] = "archived"
                        
            self.models[model_name][version]["status"] = target_status
            return True
    
    return MockModelRegistry()

@pytest.fixture
def shadow_deployment_logs():
    """Generate shadow deployment logs for testing"""
    logs = []
    np.random.seed(42)
    
    # Generate 100 log entries
    for i in range(100):
        timestamp = datetime.now() - timedelta(minutes=i*10)
        
        # Primary model score with normal distribution around 75
        primary_score = min(100, max(0, np.random.normal(75, 8)))
        
        # Shadow model score - slightly different distribution
        if i < 50:  # First half - similar performance
            shadow_score = min(100, max(0, primary_score + np.random.normal(0, 5)))
        else:  # Second half - shadow model tends to score higher
            shadow_score = min(100, max(0, primary_score + np.random.normal(5, 5)))
        
        # Add some metadata
        logs.append({
            "id": f"log_{i}",
            "timestamp": timestamp.isoformat(),
            "request_data": {
                "headline": f"Test headline {i}",
                "description": f"Test description {i}",
                "platform": "facebook" if i % 2 == 0 else "google",
            },
            "primary_prediction": {
                "model_id": "ad_score_predictor_v1.1.0",
                "score": primary_score,
                "confidence": 0.8 + np.random.random() * 0.15,
                "processing_time_ms": 25 + np.random.random() * 15
            },
            "shadow_prediction": {
                "model_id": "ad_score_predictor_v2.0.0-beta",
                "score": shadow_score,
                "confidence": 0.85 + np.random.random() * 0.1,
                "processing_time_ms": 22 + np.random.random() * 12
            }
        })
    
    return logs

@pytest.fixture
def monitoring_metrics_data():
    """Generate monitoring metrics data for testing"""
    # Create a dataset of monitoring metrics over time
    np.random.seed(42)
    data = []
    
    # Generate 7 days of data at 15-minute intervals
    start_time = datetime.now() - timedelta(days=7)
    for i in range(7 * 24 * 4):  # 7 days × 24 hours × 4 intervals per hour
        timestamp = start_time + timedelta(minutes=15 * i)
        
        # Add daily and weekly seasonality
        hour_of_day = timestamp.hour
        day_of_week = timestamp.weekday()
        
        # Traffic is higher during business hours and weekdays
        base_traffic = 100
        hour_factor = 1.5 if 9 <= hour_of_day <= 17 else 0.7
        day_factor = 1.2 if day_of_week < 5 else 0.6  # Lower on weekends
        
        traffic = base_traffic * hour_factor * day_factor
        
        # Add some random variation
        traffic_with_noise = max(1, int(traffic * (1 + np.random.normal(0, 0.1))))
        latency = max(1, 120 + np.random.normal(0, 15))  # Base 120ms latency
        error_rate = max(0, min(1, 0.01 + np.random.normal(0, 0.005)))  # ~1% error rate
        
        # Add a few anomalies
        if i == 24 * 4 * 3 + 10:  # Day 3, mid-day
            latency *= 3  # Latency spike
        if i == 24 * 4 * 5 + 30:  # Day 5, afternoon
            error_rate = 0.15  # Error spike
        if i == 24 * 4 * 6 + 8:  # Day 6, morning
            traffic_with_noise = int(traffic_with_noise * 0.1)  # Traffic drop
        
        data.append({
            "timestamp": timestamp.isoformat(),
            "metric_group": "ml_service",
            "service": "ad_score_predictor",
            "instance": "prod-1",
            "requests_per_minute": traffic_with_noise,
            "avg_latency_ms": latency,
            "p95_latency_ms": latency * 1.4,
            "p99_latency_ms": latency * 2.0,
            "error_rate": error_rate,
            "throughput": traffic_with_noise / 60,
            "memory_usage_mb": 2048 + np.random.normal(0, 100),
            "cpu_usage_percent": min(100, max(0, 30 + np.random.normal(0, 10)))
        })
    
    return pd.DataFrame(data)

@pytest.fixture(scope="session")
def sentiment_model():
    """Create a sentiment analysis model instance for testing."""
    model_name = "vinai/bertweet-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        trust_remote_code=True
    )
    return model, tokenizer

@pytest.fixture(scope="session")
def sample_sentiment_data():
    """Create sample sentiment data for testing."""
    # Load a small subset of Sentiment140
    dataset = load_dataset("stanfordnlp/sentiment140", trust_remote_code=True)
    df = pd.DataFrame(dataset['train'])
    
    # Take a small sample for testing
    sample_df = df.sample(n=100, random_state=42)
    
    # Convert sentiment from 0/4 to 0/1
    sample_df['sentiment'] = sample_df['sentiment'].map({0: 0, 4: 1})
    
    return {
        'texts': sample_df['text'].tolist(),
        'labels': sample_df['sentiment'].values
    }

@pytest.fixture(scope="session")
def training_args():
    """Create training arguments for testing."""
    class Args:
        def __init__(self):
            self.output_dir = "test_output"
            self.batch_size = 32
            self.learning_rate = 2e-5
            self.epochs = 3
            self.max_seq_length = 128
            self.model_type = "bertweet"
            self.use_full_dataset = False
            self.fairness_evaluation = False
            self.bias_mitigation = False
            self.use_ensemble = True
    
    return Args()

@pytest.fixture(scope="session")
def mock_tensorboard():
    """Create a mock TensorBoard writer."""
    mock_writer = Mock()
    mock_writer.add_text = Mock()
    mock_writer.add_scalar = Mock()
    mock_writer.close = Mock()
    return mock_writer

@pytest.fixture(scope="session")
def synthetic_demographics():
    """Generate synthetic demographic data for fairness testing."""
    np.random.seed(42)
    n_samples = 100
    
    demographics = {
        'age_group': np.random.choice(
            ['18-24', '25-34', '35-44', '45-54', '55+'],
            size=n_samples,
            p=[0.2, 0.3, 0.25, 0.15, 0.1]
        ),
        'gender': np.random.choice(['M', 'F'], size=n_samples, p=[0.48, 0.52]),
        'race': np.random.choice(
            ['White', 'Black', 'Hispanic', 'Asian', 'Other'],
            size=n_samples,
            p=[0.6, 0.13, 0.18, 0.06, 0.03]
        ),
        'education': np.random.choice(
            ['High School', 'Some College', 'Bachelor', 'Graduate'],
            size=n_samples,
            p=[0.3, 0.25, 0.3, 0.15]
        )
    }
    
    return pd.DataFrame(demographics)

@pytest.fixture(scope="session")
def mock_gpu():
    """Create a mock GPU environment."""
    with patch('torch.cuda.is_available') as mock_cuda:
        mock_cuda.return_value = True
        with patch('torch.cuda.get_device_name') as mock_device:
            mock_device.return_value = "NVIDIA A100-SXM4-80GB"
            yield

@pytest.fixture(scope="session")
def mock_cpu():
    """Create a mock CPU environment."""
    with patch('torch.cuda.is_available') as mock_cuda:
        mock_cuda.return_value = False
        yield

@pytest.fixture(scope="session")
def test_output_dir():
    """Create a temporary output directory for tests."""
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    yield output_dir
    # Cleanup after tests
    import shutil
    shutil.rmtree(output_dir)

@pytest.fixture(scope="session")
def mock_model_registry():
    """Create a mock model registry for testing."""
    class MockRegistry:
        def __init__(self):
            self.models = {}
        
        def save_model(self, model_name, model, metrics):
            self.models[model_name] = {
                'model': model,
                'metrics': metrics,
                'timestamp': pd.Timestamp.now()
            }
        
        def load_model(self, model_name):
            return self.models.get(model_name)
    
    return MockRegistry()

@pytest.fixture(scope="session")
def mock_metrics():
    """Create mock metrics for testing."""
    return {
        'accuracy': 0.85,
        'precision': 0.83,
        'recall': 0.87,
        'f1': 0.85,
        'auc': 0.92,
        'training_time': 120.5,
        'inference_time': 0.15
    }

@pytest.fixture(scope="session")
def mock_training_data():
    """Create mock training data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic text data
    texts = [f"Sample text {i} with some sentiment words" for i in range(n_samples)]
    
    # Generate synthetic labels (0 or 1)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    
    return {
        'texts': texts,
        'labels': labels,
        'n_samples': n_samples
    }

@pytest.fixture(scope="session")
def mock_validation_data():
    """Create mock validation data for testing."""
    np.random.seed(42)
    n_samples = 200
    
    # Generate synthetic text data
    texts = [f"Validation text {i} with sentiment" for i in range(n_samples)]
    
    # Generate synthetic labels (0 or 1)
    labels = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    
    return {
        'texts': texts,
        'labels': labels,
        'n_samples': n_samples
    }