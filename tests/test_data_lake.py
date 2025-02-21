# tests/test_data_lake.py
import pytest
from typing import Dict, Any
from sqlalchemy.exc import IntegrityError, DataError
from cryptography.fernet import Fernet
from app.models.domain.data_lake_model import DataLakeModel
from app.core.database import SessionLocal, Base, engine
from app.models.domain.data_catalog_model import DataCatalogModel
from app.core.data_lake.security_manager import PolicyEngine, EncryptionService, AuditLogger, PIIProcessor
import logging
from datetime import datetime

@pytest.fixture(scope="session")
def database_setup():
    """One-time database setup and teardown for the entire test session"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def db_session(database_setup):
    """Create a new database session for each test function"""
    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)
    
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()

@pytest.fixture
def valid_sample_data() -> Dict[str, Any]:
    """Provide a standard valid data sample for testing"""
    return {
        "name": "test_entry",
        "data": b"initial_test_data",
        "meta_info": {
            "source": "pytest",
            "level": "user",
            "data_layer": "raw"
        }
    }

class TestDataLakeModel:
    def test_basic_creation(self, db_session, valid_sample_data):
        """Verify successful data lake entry creation"""
        entry = DataLakeModel(**valid_sample_data)
        db_session.add(entry)
        db_session.commit()
        
        retrieved_entry = db_session.get(DataLakeModel, entry.id)
        
        assert retrieved_entry is not None
        assert retrieved_entry.name == valid_sample_data["name"]
        assert retrieved_entry.data == valid_sample_data["data"]

    def test_uuid_uniqueness(self, db_session, valid_sample_data):
        """Ensure unique UUID generation for multiple entries"""
        entries = [
            DataLakeModel(**{**valid_sample_data, "name": f"entry_{i}"}) 
            for i in range(5)
        ]
        
        db_session.add_all(entries)
        db_session.commit()
        
        unique_ids = {entry.id for entry in entries}
        assert len(unique_ids) == 5, "UUIDs must be unique"

    def test_id_constraint(self, db_session, valid_sample_data):
        """Verify database prevents duplicate IDs"""
        # Create initial entry
        initial_entry = DataLakeModel(**valid_sample_data)
        db_session.add(initial_entry)
        db_session.commit()
        
        # Attempt to create entry with same ID
        duplicate_entry = DataLakeModel(
            id=initial_entry.id,
            name="duplicate_entry",
            data=b"duplicate_data",
            meta_info=valid_sample_data["meta_info"]
        )
        
        db_session.add(duplicate_entry)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_data_size_constraints(self, db_session, valid_sample_data):
        """Test data size limitations"""
        # Valid data size (10MB)
        valid_large_data = {
            **valid_sample_data, 
            "data": b"x" * (10 * 1024 * 1024)
        }
        valid_entry = DataLakeModel(**valid_large_data)
        db_session.add(valid_entry)
        db_session.commit()
        
        # Invalid data size (exceeding 10MB)
        invalid_data = {
            **valid_sample_data, 
            "data": b"x" * (10 * 1024 * 1024 + 1)
        }
        invalid_entry = DataLakeModel(**invalid_data)
        db_session.add(invalid_entry)
        with pytest.raises((DataError, IntegrityError)):
            db_session.commit()

    def test_multi_level_data_input(self, db_session):
        """Validate support for multiple data input levels"""
        test_scenarios = [
            {
                "name": "campaign_entry",
                "data": b"campaign_data",
                "meta_info": {
                    "level": "campaign",
                    "campaign_id": "camp_001",
                    "data_layer": "raw",
                    "source": "marketing"
                }
            },
            {
                "name": "user_entry", 
                "data": b"user_data",
                "meta_info": {
                    "level": "user",
                    "user_id": "user_001",
                    "data_layer": "processed",
                    "source": "tracking"
                }
            },
            {
                "name": "environment_entry",
                "data": b"env_data", 
                "meta_info": {
                    "level": "environment",
                    "env_id": "env_001",
                    "data_layer": "curated",
                    "source": "system_logs"
                }
            }
        ]
        
        entries = [DataLakeModel(**scenario) for scenario in test_scenarios]
        db_session.add_all(entries)
        db_session.commit()
        
        for entry, scenario in zip(entries, test_scenarios):
            assert entry.meta_info["level"] == scenario["meta_info"]["level"]
            assert entry.meta_info["data_layer"] == scenario["meta_info"]["data_layer"]

# Add to TestDataLakeModel class

    def test_rbac_enforcement(self):
        """Verify role-based access control for data operations"""
        # Setup test policies
        test_policies = {
            "data_lake": {
                "admin": {"read": True, "write": True},
                "researcher": {"read": True, "write": False}
            }
        }
        policy_engine = PolicyEngine(
            role_assignments={"user1": "admin", "user2": "researcher"},
            resource_policies=test_policies
        )
        
        # Admin should have write access
        assert policy_engine.evaluate("user1", "data_lake", "write") is True
        
        # Researcher should only have read access
        assert policy_engine.evaluate("user2", "data_lake", "write") is False

    def test_catalog_lake_integration(self, db_session, valid_sample_data):
        """Validate data lake <> catalog relationship"""
        # Create lake entry first
        lake_entry = DataLakeModel(
            name="test_entry",
            data=b"test_data",
            meta_info={
                "level": "user",
                "data_layer": "raw",
                "source": "test"
            }
        )
        db_session.add(lake_entry)
        db_session.flush()  # Get the ID without committing
        
        # Now create catalog entry with lake_entry.id
        catalog_entry = DataCatalogModel(
            name="user_analytics",
            data_lake_id=lake_entry.id,
            schema_definition={"fields": ["user_id", "timestamp"]}
        )
        
        db_session.add(catalog_entry)
        db_session.commit()
        
        assert catalog_entry.data_lake_entry.id == lake_entry.id
        assert lake_entry.catalog_entries[0].name == "user_analytics"

    def test_data_catalog_lineage(self, db_session):
        """Verify data lineage tracking in catalog"""
        # Create a data lake entry first
        lake_entry = DataLakeModel(
            name="test_entry",
            data=b"test_data",
            meta_info={
                "level": "user",
                "data_layer": "raw",
                "source": "test"
            }
        )
        db_session.add(lake_entry)
        db_session.flush()

        catalog_entry = DataCatalogModel(
            name="clinical_trial_data",
            data_lake_id=lake_entry.id,  # Set the required foreign key
            schema_definition={"columns": ["patient_id", "treatment"]},
            meta_info={"level": "user"},  # Add required meta_info
            level_context={  # Add required level_context
                "user_segments": ["clinical_trial"],
                "age_range": "18-65",
                "geography": "US"
            },
            lineage={
                "source": {
                    "id": "redcap_123",
                    "name": "REDCap Clinical Trial Database",
                    "type": "external_system"
                },
                "transformations": [
                    {
                        "type": "anonymization",
                        "timestamp": datetime.utcnow().isoformat(),
                        "description": "Patient data anonymization"
                    },
                    {
                        "type": "normalization",
                        "timestamp": datetime.utcnow().isoformat(),
                        "description": "Data normalization and standardization"
                    }
                ]
            }
        )

        db_session.add(catalog_entry)
        db_session.commit()

        # Verify lineage is properly stored
        retrieved = db_session.get(DataCatalogModel, catalog_entry.id)
        assert retrieved.lineage["source"]["name"] == "REDCap Clinical Trial Database"
        assert len(retrieved.lineage["transformations"]) == 2

    def test_raw_layer_immutability(self, db_session, valid_sample_data):
        """Ensure raw data cannot be modified after ingestion"""
        raw_entry = DataLakeModel(**{
            **valid_sample_data,
            "meta_info": {"level": "user", "data_layer": "raw"}
        })
        db_session.add(raw_entry)
        db_session.commit()
        
        # Attempt modification
        with pytest.raises(ValueError, match="Cannot modify data in raw layer after creation"):
            raw_entry.data = b"modified_data"

    def test_processed_layer_transformations(self):
        """Validate PII redaction in processed layer"""
        test_data = b'{"user": "John Doe", "email": "john@example.com"}'
        processed_data = PIIProcessor.redact_pii(test_data)  # Assume PII redaction function
        
        assert b"John Doe" not in processed_data
        assert b"REDACTED" in processed_data

    def test_curated_layer_aggregation(self, db_session):
        """Verify curated layer contains aggregated metrics"""
        curated_entry = DataLakeModel(
            name="campaign_metrics",
            data=b'{"participants": 1500, "conversion_rate": 0.25}',
            meta_info={
                "level": "campaign",
                "data_layer": "curated",
                "metrics_version": "v2.1"
            }
        )
        db_session.add(curated_entry)
        db_session.commit()
        
        assert curated_entry.meta_info["data_layer"] == "curated"
        assert "conversion_rate" in curated_entry.data.decode()
    def test_data_encryption(self):
        """Validate end-to-end encryption of sensitive data"""
        test_data = b"sensitive_patient_data"
        secret_key = Fernet.generate_key()
        encryption_service = EncryptionService(secret_key)
        
        encrypted = encryption_service.encrypt(test_data)
        decrypted = encryption_service.decrypt(encrypted)
        
        assert decrypted == test_data
        assert encrypted != test_data  # Ensure actual encryption occurred

    def test_audit_logging(self, caplog):
        """Verify security events are properly logged"""
        caplog.set_level(logging.INFO)  # Set logging level to capture INFO logs
        audit_logger = AuditLogger()
        audit_logger.record_access_attempt("user3", "data_lake:123", "read", False)
        
        assert "DENIED" in caplog.text
        assert "data_lake:123" in caplog.text
    def test_metadata_validation(self, valid_sample_data):
        """Test metadata validation mechanisms"""
        invalid_scenarios = [
            # Completely empty metadata
            {**valid_sample_data, "meta_info": {}},
            
            # Missing critical fields
            {**valid_sample_data, "meta_info": {
                "source": "test"  # Missing level and data_layer
            }},
            
            # Invalid level
            {**valid_sample_data, "meta_info": {
                "level": "invalid", 
                "data_layer": "raw"
            }},
            
            # Invalid data layer
            {**valid_sample_data, "meta_info": {
                "level": "user", 
                "data_layer": "unknown"
            }}
        ]
        
        for scenario in invalid_scenarios:
            with pytest.raises(ValueError):
                DataLakeModel(**scenario)

    def test_metadata_flexibility(self, db_session):
        """Ensure flexible metadata storage"""
        flexible_entry = DataLakeModel(
            name="flexible_test",
            data=b"flexible_data",
            meta_info={
                "level": "user",
                "data_layer": "raw",
                "source": "test",
                "custom_fields": {
                    "nested_data": {
                        "key1": "value1",
                        "key2": 42
                    }
                }
            }
        )
        
        db_session.add(flexible_entry)
        db_session.commit()
        
        retrieved_entry = db_session.get(DataLakeModel, flexible_entry.id)
        assert retrieved_entry.meta_info.get("custom_fields", {}).get("nested_data", {}).get("key2") == 42

    def test_name_constraints(self, valid_sample_data):
        """Verify name field constraints"""
        invalid_names = [
            "",  # Empty name
            "a" * 256  # Overly long name
        ]
        
        for invalid_name in invalid_names:
            with pytest.raises(AssertionError):
                DataLakeModel(**{**valid_sample_data, "name": invalid_name})